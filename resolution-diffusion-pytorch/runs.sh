

python celebA_128.py --time_steps 4 --resolution_routine 'Incremental_factor_2' --save_folder './celebA_4_steps_fac2_train'
python celebA_test.py --time_steps 4 --train_routine 'Final' --sampling_routine 'x0_step_down' --resolution_routine 'Incremental_factor_2' --save_folder './celebA_test' --load_path './celebA_final_ckpt/model.pt' --test_type 'test_fid_distance_decrease_from_manifold'


python mnist_train.py --time_steps 3 --resolution_routine 'Incremental_factor_2' --save_folder './mnist_3_steps_fac2_train'
python mnist_test.py --time_steps 3 --train_routine 'Final' --sampling_routine 'x0_step_down' --resolution_routine 'Incremental_factor_2' --save_folder './mnist_test' --load_path './mnist_final_ckpt/model.pt' --test_type 'test_fid_distance_decrease_from_manifold'


python cifar10_train.py --time_steps 3 --resolution_routine 'Incremental_factor_2' --save_folder './cifar10Aug_3_steps_fac2_train'
python cifar10_test.py --time_steps 3 --train_routine 'Final' --sampling_routine 'x0_step_down' --resolution_routine 'Incremental_factor_2' --save_folder './cifar10_test' --load_path './cifar10_final_ckpt/model.pt' --test_type 'test_fid_distance_decrease_from_manifold'


