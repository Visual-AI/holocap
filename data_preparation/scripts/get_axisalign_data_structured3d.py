import os 
import argparse
from pathlib import Path

import pickle as pkl 
import numpy as np
import open3d as o3d

import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('-i', '--run-idx', type=int, default=0)
parser.add_argument('--data-root', type=Path, required=True)
parser.add_argument('--output-dir', type=Path, required=True)
args = parser.parse_args()
run_idx = args.run_idx

data_root = args.data_root.expanduser()
output_dir = args.output_dir.expanduser()
if not data_root.is_dir():
    raise FileNotFoundError(f'Dataset root does not exist: {data_root}')
output_dir.mkdir(parents=True, exist_ok=True)

metadata_path = Path(__file__).resolve().parents[1] / "metadata" / "structured3d.pkl"

with metadata_path.open('rb') as f:
    anno_dct = pkl.load(f)

anno_list = anno_dct['data_list']

num_scenes = len(anno_list)
anno_idxs = list(range(500*(run_idx), min(500*(run_idx+1), num_scenes)))

for anno_idx in tqdm.tqdm(anno_idxs, desc='scenes'):
    aa_matrix = anno_list[anno_idx]['axis_align_matrix']
    
    scene_id = anno_list[anno_idx]['sample_idx'].split('/')[0]
    room_id = anno_list[anno_idx]['sample_idx'].split('/')[2]
    ply_path = '{}/{}_{}.ply'.format(anno_list[anno_idx]['sample_idx'], scene_id, room_id)

    save_path = output_dir / f'{scene_id}_room_{room_id}_axisaligned_points.ply'
    if os.path.exists(save_path):
        continue

    mesh = o3d.io.read_point_cloud(os.path.join(data_root, ply_path))
    mesh.transform(aa_matrix)

    verts = np.asarray(mesh.points)
    colors = np.asarray(mesh.colors)

    assert verts.shape == colors.shape
    
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(verts)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    o3d.io.write_point_cloud(str(save_path), pcd)

