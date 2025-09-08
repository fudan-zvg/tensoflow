export CUDA_VISIBLE_DEVICES=$2

python run_training.py --cfg configs/shape/syn/$1.yaml ${@:3}

python eval_geo.py --cfg configs/shape/syn/$1.yaml ${@:3}

python extract_mesh.py --cfg configs/shape/syn/$1.yaml ${@:3}
