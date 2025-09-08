export CUDA_VISIBLE_DEVICES=$1

python eval_mat.py --cfg configs/mat/orb/teapot.yaml --orb_relight_env teapot_scene001 ${@:2}
python eval_mat.py --cfg configs/mat/orb/teapot.yaml --orb_relight_env teapot_scene002 ${@:2}
python eval_orb_relight.py --cfg configs/mat/orb/teapot.yaml --relight_dir teapot_scene006_relighting_teapot_scene001 --gt_dir nerf_data/orb/blender_LDR/teapot_scene001 ${@:2}
python eval_orb_relight.py --cfg configs/mat/orb/teapot.yaml --relight_dir teapot_scene006_relighting_teapot_scene002 --gt_dir nerf_data/orb/blender_LDR/teapot_scene002 ${@:2}
