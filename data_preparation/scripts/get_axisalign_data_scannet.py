import os 
import argparse
from pathlib import Path

import pickle as pkl 
import numpy as np
import open3d as o3d

import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('-i', '--run-idx', type=int, default=0)
parser.add_argument('--val', action='store_true', default=False)
parser.add_argument('--data-root', type=Path, required=True)
parser.add_argument('--output-dir', type=Path, required=True)
args = parser.parse_args()
run_idx = args.run_idx

split = 'train'
if args.val:
    split = 'val'
    num_scenes = 243
else:
    assert split == 'train'
    num_scenes = 958

data_root = args.data_root.expanduser()
output_dir = args.output_dir.expanduser()
if not data_root.is_dir():
    raise FileNotFoundError(f'Dataset root does not exist: {data_root}')
output_dir.mkdir(parents=True, exist_ok=True)

metadata_path = Path(__file__).resolve().parents[1] / "metadata" / f"holocap_{split}.pkl"
anno_idxs = list(range(500*(run_idx), min(500*(run_idx+1), num_scenes)))

with metadata_path.open('rb') as f:
    anno_dct = pkl.load(f)

anno_list = anno_dct['data_list']

for anno_idx in tqdm.tqdm(anno_idxs, desc='scenes'):
    aa_matrix = anno_list[anno_idx]['axis_align_matrix']
    
    scene_id = anno_list[anno_idx]['sample_idx'].split('/')[-1]
    ply_path = 'scans/{}/{}_vh_clean.ply'.format(scene_id, scene_id)          # ply path in the original dataset

    mesh = o3d.io.read_triangle_mesh(os.path.join(data_root, ply_path), True)
    mesh.transform(aa_matrix)

    verts = np.asarray(mesh.vertices)
    colors = np.asarray(mesh.vertex_colors)
    
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(verts)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    save_path = output_dir / f'{scene_id}_vh_clean_axisaligned_points.ply'
    o3d.io.write_point_cloud(str(save_path), pcd)


