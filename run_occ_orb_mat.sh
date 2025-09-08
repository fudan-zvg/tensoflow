export CUDA_VISIBLE_DEVICES=$2

python run_training.py --cfg configs/mat/orb/$1.yaml ${@:3}