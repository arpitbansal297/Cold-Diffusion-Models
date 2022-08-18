# Cold Diffusion: Inverting Arbitrary Image Transforms Without Noise
<img src="./all_transform_cover.png" width="1000px"></img>

This repository is the official PyTorch implementation of Cold-Diffusion. Find the paper on arxiv



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
```
cd defading-diffusion-pytorch
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
```
cd resolution-diffusion-pytorch
```

#### Transformation
```

```

#### Generation
```

```

## Snowify
```

```

## Colorization
```

```
