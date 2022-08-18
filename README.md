This repository is the official PyTorch implementation of Cold-Diffusion. Find the paper on arxiv

# Cold Diffusion: Inverting Arbitrary Image Transforms Without Noise

## Denoise

#### Generation

sampling_routine with fixed noise
```
cd denoising-diffusion-pytorch
python celebA_noise_128.py --time_steps 200 --sampling_routine x0_step_down --save_folder ./results_celebA_128_200_steps_noise --load_path ./results_celebA_128_200_steps_noise/model.pt
python AFHQ_noise_128.py --time_steps 200 --sampling_routine x0_step_down --save_folder ./results_AFHQ_128_200_steps_noise --load_path ./results_AFHQ_128_200_steps_noise/model.pt
```

sampling_routine with estimated noise
```
cd denoising-diffusion-pytorch
python celebA_noise_128.py --time_steps 200 --sampling_routine ddim --save_folder ./results_celebA_128_200_steps_noise --load_path ./results_celebA_128_200_steps_noise/model.pt
python AFHQ_noise_128.py --time_steps 200 --sampling_routine ddim --save_folder ./results_AFHQ_128_200_steps_noise --load_path ./results_AFHQ_128_200_steps_noise/model.pt
```

## Deblur

```
cd deblurring-diffusion-pytorch
```

#### Transformation
```
python mnist_train.py --time_steps 20 --blur_size 11 --blur_std 7.0 --blur_routine 'Constant' --sampling_routine 'x0_step_down' --save_folder './results_mnist_constant_11_20'
python cifar10_train.py --time_steps 50 --blur_routine 'Special_6_routine' --sampling_routine 'x0_step_down' --save_folder './results_cifar10_special_6' 
python celebA_128.py --time_steps 200 --blur_size 15 --blur_std 0.01 --blur_routine Exponential_reflect --sampling_routine x0_step_down --save_folder ./results_celebA_128_final_less
```

#### Generation
```
python celebA_128.py --discrete --time_steps 300 --blur_size 27 --blur_std 0.01 --blur_routine Exponential --sampling_routine x0_step_down --save_folder ./results_celebA_128_final_extreme_circular_discrete
python AFHQ_128.py --discrete --time_steps 300 --blur_size 27 --blur_std 0.01 --blur_routine Exponential --sampling_routine x0_step_down --save_folder ./results_AFHQ_128_final_extreme_circular_discrete
```

## Animorph

#### Generation

```
cd demixing-diffusion-pytorch
python AFHQ_128_to_celebA_128.py --time_steps 200 --sampling_routine x0_step_down --save_folder ./results_AFHQ_128_to_celebA_128_200_steps
```

## Inpaint

#### Transformation
```
cd defading-diffusion-pytorch
```

#### Generation

```
cd defading-generation-diffusion-pytorch
python celebA_constant_128.py --reverse --kernel_std 0.05 --initial_mask 1 --time_steps 750 --sampling_routine x0_step_down --save_folder ./results_celebA_128_1_0_05_300_steps_reverse_constant
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
