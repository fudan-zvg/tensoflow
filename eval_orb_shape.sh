export CUDA_VISIBLE_DEVICES=$2

python eval_orb_shape.py --out_mesh_path data/meshes/teapot_scene006_shape_$1-180000.ply --target_mesh_path nerf_data/orb/ground_truth/teapot_scene006/mesh_blender/mesh.obj
python eval_orb_shape.py --out_mesh_path data/meshes/gnome_scene003_shape_$1-180000.ply --target_mesh_path nerf_data/orb/ground_truth/gnome_scene003/mesh_blender/mesh.obj
python eval_orb_shape.py --out_mesh_path data/meshes/cactus_scene001_shape_$1-180000.ply --target_mesh_path nerf_data/orb/ground_truth/cactus_scene001/mesh_blender/mesh.obj
python eval_orb_shape.py --out_mesh_path data/meshes/car_scene004_shape_$1-180000.ply --target_mesh_path nerf_data/orb/ground_truth/car_scene004/mesh_blender/mesh.obj
python eval_orb_shape.py --out_mesh_path data/meshes/grogu_scene001_shape_$1-180000.ply --target_mesh_path nerf_data/orb/ground_truth/grogu_scene001/mesh_blender/mesh.obj