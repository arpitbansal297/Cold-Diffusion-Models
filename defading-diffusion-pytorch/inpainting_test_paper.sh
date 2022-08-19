#python defading-diffusion-pytorch/cifar10_test.py --time_steps 50 --save_folder ./cifar_inpainting_test --discrete --sampling_routine x0_step_down --blur_std 0.1 --fade_routine Random_Incremental --test_type test_paper_invert_section_images

#python defading-diffusion-pytorch/celebA_128_test.py --load_path /cmlscratch/eborgnia/cold_diffusion/celebA_64x64_100_step_11_init/model.pt --time_steps 100 --save_folder ./celebA_inpainting_test --discrete --sampling_routine x0_step_down --blur_std 0.1 --initial_mask 11 --fade_routine Random_Incremental --test_type test_data

#python defading-diffusion-pytorch/mnist_test.py --time_steps 50 --save_folder ./mnist_inpainting_test --discrete --sampling_routine x0_step_down --blur_std 0.1 --fade_routine Random_Incremental --test_type test_paper_invert_section_images

python defading-diffusion-pytorch/celebA_test.py