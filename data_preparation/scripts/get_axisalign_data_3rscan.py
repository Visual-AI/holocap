import os 
import argparse
from pathlib import Path

import pickle as pkl 
import numpy as np
import open3d as o3d
import trimesh

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
    scene_idx_max = 477
    scene_idx_min = 243
else:
    assert split == 'train'
    scene_idx_max = 1900
    scene_idx_min = 958

data_root = args.data_root.expanduser()
output_dir = args.output_dir.expanduser()
if not data_root.is_dir():
    raise FileNotFoundError(f'Dataset root does not exist: {data_root}')
output_dir.mkdir(parents=True, exist_ok=True)

metadata_path = Path(__file__).resolve().parents[1] / "metadata" / f"holocap_{split}.pkl"
anno_idxs = list(range(1000*(run_idx)+scene_idx_min, min(1000*(run_idx+1)+scene_idx_min, scene_idx_max+1)))

with metadata_path.open('rb') as f:
    anno_dct = pkl.load(f)

anno_list = anno_dct['data_list']

for anno_idx in tqdm.tqdm(anno_idxs, desc='scenes'):
    aa_matrix = anno_list[anno_idx]['axis_align_matrix']
    
    scene_id = anno_list[anno_idx]['sample_idx'].split('/')[-1]
    ply_path = 'scans/{}/mesh.refined.v2.obj'.format(scene_id)          # ply path in the original dataset

    save_path = output_dir / f'{scene_id}_axisaligned_points.ply'
    if os.path.exists(save_path):
        continue

    mesh1 = o3d.io.read_triangle_mesh(os.path.join(data_root, ply_path), False)
    pts1 = np.array(mesh1.vertices)
    mesh2 = trimesh.load(os.path.join(data_root, ply_path), process=False)
    pts2 = np.array(mesh2.vertices)

    # Verify that Open3D and trimesh load identical vertex sets.
    pts1_sorted = pts1[np.lexsort(pts1.T)]
    pts2_sorted = pts2[np.lexsort(pts2.T)]
    if pts1.shape == pts2.shape:
        assert np.allclose(pts1_sorted, pts2_sorted, atol=1e-8)
    else:
        raise ValueError(
            f"Open3D and trimesh vertex shapes differ: "
            f"{pts1_sorted.shape} vs {pts2_sorted.shape}"
        )

    colors = trimesh.visual.uv_to_color(mesh2.visual.uv, mesh2.visual.material.image)
    unique_colors = set(tuple(c[:3]) for c in colors.tolist())
    assert len(unique_colors) >= 10     # ensure there are more than 10 colors

    colors = colors[:, :3] / 255.0

    # transform to axis_aligned points
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts2)
    pcd.transform(aa_matrix)
    verts = np.asarray(pcd.points)

    assert verts.shape[0] == colors.shape[0]
    
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(verts)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    o3d.io.write_point_cloud(str(save_path), pcd)

