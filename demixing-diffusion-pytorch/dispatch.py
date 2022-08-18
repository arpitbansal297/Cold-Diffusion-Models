"""dispatch.py
   for dispatching jobs on CML
   July 2021
"""
import argparse
import getpass
import os
import random
import subprocess
import time
import warnings

parser = argparse.ArgumentParser(description="Dispatch python jobs on the CML cluster")
parser.add_argument("file", type=argparse.FileType())
parser.add_argument("--qos", default="scav", type=str,
                    help="QOS, choose default, medium, high, scav")
parser.add_argument("--name", default=None, type=str,
                    help="Name that will be displayed in squeue. Default: file name")
parser.add_argument("--gpus", default="1", type=int, help="Requested GPUs per job")
parser.add_argument("--mem", default="64", type=int, help="Requested memory per job")
parser.add_argument("--time", default=10, type=int, help="Requested hour limit per job")
args = parser.parse_args()

# get username for use in email and squeue below (do crazy things for arjun)
cmluser_str = getpass.getuser()
username = cmluser_str

# Parse and validate input:
if args.name is None:
    dispatch_name = args.file.name
else:
    dispatch_name = args.name

# Usage warnings:
if args.mem > 385:
    raise ValueError("Maximal node memory exceeded.")
if args.gpus > 8:
    raise ValueError("Maximal node GPU number exceeded.")
if args.qos == "high" and args.gpus > 4:
    warnings.warn("QOS only allows for 4 GPUs, GPU request has been reduced to 4.")
    args.gpus = 4
if args.qos == "medium" and args.gpus > 2:
    warnings.warn("QOS only allows for 2 GPUs, GPU request has been reduced to 2.")
    args.gpus = 2
if args.qos == "default" and args.gpus > 1:
    warnings.warn("QOS only allows for 1 GPU, GPU request has been reduced to 1.")
    args.gpus = 1
if args.mem / args.gpus > 48:
    warnings.warn("You are oversubscribing to memory. "
                  "This might leave some GPUs idle as total node memory is consumed.")

# 1) Stripping file of comments and blank lines
content = args.file.readlines()
jobs = [c.strip().split("#", 1)[0] for c in content if "python" in c and c[0] != "#"]

print(f"Detected {len(jobs)} jobs.")

# Write file list
authkey = random.randint(10**5, 10**6 - 1)
with open(f".cml_job_list_{authkey}.temp.sh", "w") as file:
    file.writelines(chr(10).join(job for job in jobs))
    file.write("\n")

# 2) Prepare environment
if not os.path.exists("cmllogs"):
    os.makedirs("cmllogs")

# 3) Construct launch file
SBATCH_PROTOTYPE = \
    f"""#!/bin/bash
# Lines that begin with #SBATCH specify commands to be used by SLURM for scheduling
#SBATCH --job-name={"".join(e for e in dispatch_name if e.isalnum())}
#SBATCH --array={f"1-{len(jobs)}"}
#SBATCH --output=cmllogs/%x_%A_%a.log
#SBATCH --error=cmllogs/%x_%A_%a.log
#SBATCH --time={args.time}:00:00
#SBATCH --account={"tomg" if args.qos != "scav" else "scavenger"}
#SBATCH --qos={args.qos if args.qos != "scav" else "scavenger"}
#SBATCH --gres=gpu:{args.gpus}
#SBATCH --cpus-per-task=4
#SBATCH --partition={"dpart" if args.qos != "scav" else "scavenger"}
#SBATCH --mem={args.mem}gb
#SBATCH --mail-user={username}@umd.edu
#SBATCH --mail-type=END,TIME_LIMIT,FAIL,ARRAY_TASKS
#SBATCH --exclude=cmlgrad05,cmlgrad02,cml12,cml17,cml18,cml19,cml20,cml21,cml22,cml23,cml24
srun $(head -n $((${{SLURM_ARRAY_TASK_ID}} + 0)) .cml_job_list_{authkey}.temp.sh | tail -n 1)
"""

# Write launch commands to file
with open(f".cml_launch_{authkey}.temp.sh", "w") as file:
    file.write(SBATCH_PROTOTYPE)
print("Launch prototype is ...")
print("---------------")
print(SBATCH_PROTOTYPE)
print("---------------")
print(chr(10).join("srun " + job for job in jobs))
print(f"Preparing {len(jobs)} jobs ")
print("Terminate if necessary ...")
for _ in range(10):
    time.sleep(1)

# Execute file with sbatch
subprocess.run(["/usr/bin/sbatch", f".cml_launch_{authkey}.temp.sh"])
print("Subprocess launched ...")
time.sleep(1)
os.system(f"watch squeue -u {cmluser_str}")