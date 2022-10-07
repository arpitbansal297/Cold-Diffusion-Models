# Longer step experiments
python train.py --time_steps 20 --forward_process_type 'Decolorization'  --dataset_folder <path_to_cifar10_train_set> --exp_name 'cifar_exp' --decolor_total_remove --decolor_routine 'Linear'
python train.py --time_steps 20 --forward_process_type 'Decolorization' --exp_name 'celeba_exp' --decolor_ema_factor 0.9 --decolor_total_remove --decolor_routine 'Linear' --dataset celebA --dataset_folder <path_to_celeba_train_set> --resolution 64 --resume_training
