export CUDA_VISIBLE_DEVICES=$2

python run_training.py --cfg configs/mat/syn/$1.yaml ${@:3}

python eval_mat.py --cfg configs/mat/syn/$1.yaml --env_dir nerf_data/tensoSDF/env_maps_exr/envmaps_1k_exr ${@:3}