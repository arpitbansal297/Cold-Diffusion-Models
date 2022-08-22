# Cold Diffusion: Inverting Arbitrary Image Transforms Without Noise
<img src="./all_transform_cover.png" width="1000px"></img>

This repository is the official PyTorch implementation of Cold-Diffusion. Find [the paper on arxiv](https://arxiv.org/abs/2208.09392)

Download the CelebA-HQ and AFHQ dataset.
Use the following script to create data and use them as path to data for MNIST, Cifar10 and CelebA. 
```
python create_data.py
```

## Denoise

#### Training 
```
cd denoising-diffusion-pytorch
python celebA_noise_128.py --time_steps 200 --sampling_routine x0_step_down --save_folder <Path to save model> --data_path <Path to data folder>
python AFHQ_noise_128.py --time_steps 200 --sampling_routine x0_step_down --save_folder <Path to save model> --data_path <Path to data folder>
```

sampling_routine with estimated noise
```
python celebA_noise_128_test.py --time_steps 200 --sampling_routine ddim --save_folder <Path to save images> --load_path <Path to load model> --data_path <Path to data folder> --test_type test_sample_and_save_for_fid
python AFHQ_noise_128_test.py --time_steps 200 --sampling_routine ddim --save_folder <Path to save images> --load_path <Path to load model> --data_path <Path to data folder> --test_type test_sample_and_save_for_fid
```

sampling_routine with fixed noise
```
python celebA_noise_128_test.py --time_steps 200 --sampling_routine x0_step_down --save_folder <Path to save images> --load_path <Path to load model> --data_path <Path to data folder> --test_type test_sample_and_save_for_fid
python AFHQ_noise_128_test.py --time_steps 200 --sampling_routine x0_step_down --save_folder <Path to save images> --load_path <Path to load model> --data_path <Path to data folder> --test_type test_sample_and_save_for_fid
```

## Deblur

```
cd deblurring-diffusion-pytorch
```

#### Transformation
Training
```
python mnist_train.py --time_steps 20 --blur_size 11 --blur_std 7.0 --blur_routine 'Constant' --sampling_routine x0_step_down --data_path <Path to data folder> --save_folder <Path to save model> 
python cifar10_train.py --time_steps 50 --blur_routine 'Special_6_routine' --sampling_routine x0_step_down --data_path <Path to data folder> --save_folder <Path to save model> 
python celebA_128.py --time_steps 200 --blur_size 15 --blur_std 0.01 --blur_routine Exponential_reflect --sampling_routine x0_step_down --data_path <Path to data folder> --save_folder <Path to save model> 
```

Testing
```
python mnist_test.py --time_steps 20 --blur_size 11 --blur_std 7.0 --blur_routine 'Constant' --sampling_routine 'x0_step_down' --save_folder <Path to save results> --data_path <Path to data folder> --test_type test_data
python cifar10_test.py --time_steps 50 --blur_routine 'Special_6_routine' --sampling_routine 'x0_step_down' --save_folder <Path to save results> --data_path <Path to data folder> --test_type test_data
python celebA_128_test.py --time_steps 200 --blur_size 15 --blur_std 0.01 --blur_routine Exponential_reflect --sampling_routine x0_step_down --save_folder <Path to save results> --data_path <Path to data folder> --test_type test_data
```

#### Generation

Training
```
python celebA_128.py --discrete --time_steps 300 --blur_size 27 --blur_std 0.01 --blur_routine Exponential --sampling_routine x0_step_down --data_path <Path to data folder> --save_folder <Path to save models>
python AFHQ_128.py --discrete --time_steps 300 --blur_size 27 --blur_std 0.01 --blur_routine Exponential --sampling_routine x0_step_down --data_path <Path to data folder> --save_folder <Path to save models>
```

Sampling with Perfect Symmetry
```
python celebA_128_test.py --gmm_cluster 1 --noise 0.000 --discrete --time_steps 300 --blur_size 27 --blur_std 0.01 --blur_routine Exponential --sampling_routine x0_step_down --save_folder <Path to save results> --load_path <Path to load models> --data_path <Path to data folder> --test_type train_distribution_mean_blur_torch_gmm_ablation
python AFHQ_128_test.py --gmm_cluster 1 --noise 0.000 --discrete --time_steps 300 --blur_size 27 --blur_std 0.01 --blur_routine Exponential --sampling_routine x0_step_down --save_folder <Path to save results> --load_path <Path to load models> --data_path <Path to data folder> --test_type train_distribution_mean_blur_torch_gmm_ablation
```


## Animorph

#### Generation

Training
```
cd demixing-diffusion-pytorch
python AFHQ_128_to_celebA_128.py --time_steps 200 --sampling_routine x0_step_down --save_folder <path to save models> --data_path_start <Path to starting data manifold> --data_path_end <Path to ending data manifold>
```
Sampling
```
python AFHQ_128_to_celebA_128_test.py --time_steps 200 --sampling_routine x0_step_down --save_folder <Path to save images> --load_path <Path to load model> --data_path_start <Path to starting data manifold> --data_path_end <Path to ending data manifold> --test_type test_sample_and_save_for_fid
```

## Inpaint

#### Transformation
Training
```
cd defading-diffusion-pytorch
python mnist_train.py --time_steps 50 --save_folder <path to save models> --discrete --sampling_routine x0_step_down --train_steps 700000 --blur_std 0.1 --fade_routine Random_Incremental --data_path <Path to data folder>
python cifar10_train.py --time_steps 50 --save_folder <path to save models> --discrete --sampling_routine x0_step_down --train_steps 700000 --blur_std 0.1 --fade_routine Random_Incremental --data_path <Path to data folder>
python celebA_train.py --time_steps 100 --fade_routine Incremental --save_folder <path to save models> --sampling_routine x0_step_down --train_steps 350000 --kernel_std 0.2 --initial_mask 1 --image_size 128 --dataset celebA --data_path <Path to data folder>
```

Testing
```
python mnist_test.py --time_steps 50 --save_folder test_mnist --discrete --sampling_routine x0_step_down --kernel_std 0.1 --initial_mask 1 --image_size 28 --fade_routine Random_Incremental --load_path <Path to load model> --data_path <Path to data folder> --test_type test_data 
python cifar10_test.py --time_steps 50 --save_folder test_cifar10 --discrete --sampling_routine x0_step_down --kernel_std 0.1 --initial_mask 1 --image_size 32 --fade_routine Random_Incremental --load_path <Path to load model> --data_path <Path to data folder> --test_type test_data
python celebA_test.py --time_steps 100 --fade_routine Incremental --save_folder test_celebA --sampling_routine x0_step_down --kernel_std 0.2 --initial_mask 1 --image_size 128 --dataset celebA --load_path <Path to load model> --data_path <Path to data folder> --test_type test_data
```

#### Generation

Training
```
cd defading-generation-diffusion-pytorch
python celebA_128.py --reverse --kernel_std 0.05 --initial_mask 1 --time_steps 750 --sampling_routine x0_step_down --save_folder <Path to save models> --data_path <Path to data folder>
```

Sampling
```
python celebA_constant_128_test.py --noise 0 --reverse --kernel_std 0.05 --initial_mask 1 --time_steps 750 --sampling_routine x0_step_down --save_folder <Path to save images> --data_path <Path to data folder> --load_path <Path to load model> --test_type test_sample_and_save_for_fid
```



## Super-Resolution

Training
```
cd resolution-diffusion-pytorch
python mnist_train.py --time_steps 3 --resolution_routine 'Incremental_factor_2' --save_folder <Path to save models>
python cifar10_train.py --time_steps 3 --resolution_routine 'Incremental_factor_2' --save_folder <Path to save models>
python celebA_128.py --time_steps 4 --resolution_routine 'Incremental_factor_2' --save_folder <Path to save models>
```

Testing
```
python mnist_test.py --time_steps 3 --train_routine 'Final' --sampling_routine 'x0_step_down' --resolution_routine 'Incremental_factor_2' --save_folder <Path to save images> --load_path <Path to load model> --test_type test_data
python cifar10_test.py --time_steps 3 --train_routine 'Final' --sampling_routine 'x0_step_down' --resolution_routine 'Incremental_factor_2' --save_folder <Path to save images> --load_path <Path to load model> --test_type test_data
python celebA_test.py --time_steps 4 --train_routine 'Final' --sampling_routine 'x0_step_down' --resolution_routine 'Incremental_factor_2' --save_folder <Path to save images> --load_path <Path to load model> --test_type test_data
```

#### Generation
```
python celebA_test.py --time_steps 4 --train_routine 'Final' --sampling_routine 'x0_step_down' --resolution_routine 'Incremental_factor_2' --save_folder <Path to save images> --load_path <Path to load model> --test_type test_data
```

## Snowify

```
cd snowification
```

Training
```
python train.py --dataset cifar10 --time_steps 200 --forward_process_type ‘Snow’ --snow_level 3 --exp_name <exp_name>  --dataset_folder <path-to-dataset> --random_snow --fix_brightness  --sampling_routine x0_step_down
python train.py --dataset celebA --time_steps 200 --forward_process_type ‘Snow’ --snow_level 4 --exp_name <exp_name> --dataset_folder <path-to-dataset> --random_snow --fix_brightness  --sampling_routine x0_step_down
```

Testing
```
python test.py --dataset cifar10 --time_steps 200 --forward_process_type ‘Snow’ --snow_level 3 --exp_name <exp_name> --dataset_folder <path-to-dataset> --random_snow --fix_brightness --resume_training --sampling_routine x0_step_down --test_type test_data --order_seed 1
python test.py --dataset celebA --time_steps 200 --forward_process_type ‘Snow’ --snow_level 4 --exp_name <exp_name> --dataset_folder <path-to-dataset> --random_snow --fix_brightness --resume_training --sampling_routine x0_step_down --test_type test_data --order_seed 1
```

## Colorization

```
cd decolor-diffusion
```

Training
```
python train.py --dataset cifar10 --time_steps 20 --forward_process_type ‘Decolorization’ --exp_name <exp_name> --decolor_total_remove --decolor_routine ‘Linear’ --dataset_folder <path-to-dataset>
python train.py --dataset celebA --time_steps 20 --forward_process_type ‘Decolorization’ --exp_name <exp_name> --decolor_total_remove --decolor_routine ‘Linear’ --dataset_folder <path-to-dataset>
```

Testing
```
python test.py --dataset cifar10 --time_steps 20 --forward_process_type ‘Decolorization’ --exp_name <exp-name>  --decolor_total_remove --decolor_routine ‘Linear’ --dataset_folder <path-to-dataset> --sampling_routine x0_step_down --test_type test_data --order_seed 1
python test.py --dataset celebA --time_steps 20 --forward_process_type ‘Decolorization’ --exp_name <exp-name>  --decolor_total_remove --decolor_routine ‘Linear’ --dataset_folder <path-to-dataset> --sampling_routine x0_step_down --test_type test_data --order_seed 1
```


## BibTeX Citation
```
@misc{https://doi.org/10.48550/arxiv.2208.09392,
  doi = {10.48550/ARXIV.2208.09392},
  url = {https://arxiv.org/abs/2208.09392},
  author = {Bansal, Arpit and Borgnia, Eitan and Chu, Hong-Min and Li, Jie S. and Kazemi, Hamid and Huang, Furong and Goldblum, Micah and Geiping, Jonas and Goldstein, Tom},
  keywords = {Computer Vision and Pattern Recognition (cs.CV), Machine Learning (cs.LG), FOS: Computer and information sciences, FOS: Computer and information sciences},
  title = {Cold Diffusion: Inverting Arbitrary Image Transforms Without Noise},
  publisher = {arXiv},
  year = {2022},
  copyright = {arXiv.org perpetual, non-exclusive license}
}
```
