"""Build neighboring entity candidates for HoloCap relation inference."""

import os
import json
import tqdm
import argparse
import numpy as np
import torch
import open3d as o3d
from pytorch3d.ops import box3d_overlap
import trimesh

parser = argparse.ArgumentParser("Build HoloCap relation candidates")
parser.add_argument('--model-name', type=str, required=True)
parser.add_argument('--attribute-path', type=str, default='filtered_entities')
parser.add_argument('--relation-path', type=str, default='relation_candidates')
args = parser.parse_args()
model_name = args.model_name

num_thr = 5
iou_thr = 0.0
dist_thr = 0.2

attr_root = os.path.join('results', model_name, args.attribute_path)
rel_root = os.path.join('results', model_name, args.relation_path)
os.makedirs(rel_root, exist_ok=True)

# transform 9-dof box (list) to o3d.geometry.OrientedBoundingBox
def _9dof_to_box(box, label=None, color_selector=None, color=None):
    if isinstance(box, list):
        box = np.array(box)
    if isinstance(box, torch.Tensor):
        box = box.cpu().numpy()
    center = box[:3].reshape(3, 1)
    scale = box[3:6].reshape(3, 1)
    rot = box[6:].reshape(3, 1)
    rot_mat = o3d.geometry.OrientedBoundingBox.get_rotation_matrix_from_zxy(
        rot)
    geo = o3d.geometry.OrientedBoundingBox(center, rot_mat, scale)

    if color is not None:
        geo.color = [x / 255.0 for x in color]
        return geo

    if label is not None and color_selector is not None:
        color = color_selector.get_color(label)
        color = [x / 255.0 for x in color]
        geo.color = color
    return geo

def _9dof_to_trimesh(box):
    if isinstance(box, list):
        box = np.array(box)
    if isinstance(box, torch.Tensor):
        box = box.cpu().numpy()
    center = box[:3]
    extent = box[3:6]          # Open3D uses HALF sizes
    euler_angles = box[6:9]          # [rx, ry, rz] in radians, ZXY order
    rot_mat = o3d.geometry.OrientedBoundingBox.get_rotation_matrix_from_zxy(euler_angles)
    mesh = trimesh.creation.box(extents=extent)
    transform = np.eye(4)
    transform[:3, :3] = rot_mat
    transform[:3, 3] = center
    mesh.apply_transform(transform)
    return mesh

def boxiou_3d(ibox, jbox):
    if ibox[3] <= 0 or ibox[4] <= 0 or ibox[5] <= 0:
        return 0
    if jbox[3] <= 0 or jbox[4] <= 0 or jbox[5] <= 0:
        return 0
    ivert_o3d = np.asarray(_9dof_to_box(ibox).get_box_points())
    jvert_o3d = np.asarray(_9dof_to_box(jbox).get_box_points())
    o2p_idxs = [0, 1, 7, 2, 3, 6, 4, 5]
    ivert_p3d = torch.from_numpy(ivert_o3d[o2p_idxs])[None, :].float() 
    jvert_p3d = torch.from_numpy(jvert_o3d[o2p_idxs])[None, :].float() 
    _, iou_3d = box3d_overlap(ivert_p3d, jvert_p3d)
    return iou_3d.item()

json_list = sorted(os.listdir(attr_root))
for json_name in tqdm.tqdm(json_list):
    attr_path = os.path.join(attr_root, json_name)
    save_path = os.path.join(rel_root, json_name)
    if os.path.exists(save_path):
        continue
    with open(attr_path) as f:
        inst_list = json.load(f)

    num_inst = len(inst_list)
    # add virtual instances
    if num_inst < 5:
        centers = np.array([dct['bbox_3d'][:3] for dct in inst_list])
        max_coord = centers.max(axis=0)
        for i in range(1, num_thr - num_inst + 1):
            cx, cy, cz = max_coord + i
            dummy_bbox = [float(cx), float(cy), float(cz), 0.2, 0.2, 0.2, 0.0, 0.0, 0.0]
            inst_list.append({
                'flag': 'None', 
                'bbox_3d': dummy_bbox, 
                'class_label': 'object', 
                'language_string': 'Bbox(object,{},{},{},{},{},{},{})'.format(float(cx), float(cy), float(cz), 0.2, 0.2, 0.2, 0.0)
            })
        num_inst = len(inst_list)
        
    ret_dct = {}
    inst_dct = {}
    for iidx, dct in enumerate(inst_list):
        inst_dct[iidx] = dct
    ret_dct['instances'] = inst_dct

    ious_mat = np.array([[1.0 for _ in range(num_inst)] for _ in range(num_inst)])
    for iidx in range(num_inst):
        for jidx in range(iidx+1, num_inst):
            ious_mat[iidx, jidx] = boxiou_3d(inst_dct[iidx]['bbox_3d'], inst_dct[jidx]['bbox_3d'])
            ious_mat[jidx, iidx] = ious_mat[iidx, jidx]
    dist_mat = np.array([[0.0 for _ in range(num_inst)] for _ in range(num_inst)])
    for iidx in range(num_inst):
        for jidx in range(iidx+1, num_inst):
            mesh1 = _9dof_to_trimesh(inst_dct[iidx]['bbox_3d'])
            bbox1 = _9dof_to_box(inst_dct[iidx]['bbox_3d'])
            pts = bbox1.get_box_points()
            assert np.allclose(np.stack((pts[0],pts[3],pts[2],pts[5],pts[1],pts[6],pts[7],pts[4])), mesh1.vertices)
            mesh2 = _9dof_to_trimesh(inst_dct[jidx]['bbox_3d'])
            bbox2 = _9dof_to_box(inst_dct[jidx]['bbox_3d'])
            pts = bbox2.get_box_points()
            assert np.allclose(np.stack((pts[0],pts[3],pts[2],pts[5],pts[1],pts[6],pts[7],pts[4])), mesh2.vertices)
            # calculate distance
            d1 = mesh1.nearest.on_surface(mesh2.vertices)[1].min()
            d2 = mesh2.nearest.on_surface(mesh1.vertices)[1].min()
            dist_mat[iidx, jidx] = min(d1, d2)
            dist_mat[jidx, iidx] = dist_mat[iidx, jidx]

    neighbors = [[] for _ in range(num_inst)]
    for i in range(num_inst):
        for j in range(num_inst):
            if i != j and (ious_mat[i,j] > iou_thr or dist_mat[i,j] < dist_thr):
                neighbors[i].append(j)
    ret_dct['instance_neighbors'] = neighbors

    with open(save_path, 'w') as f:
        json.dump(ret_dct, f, indent=4)



