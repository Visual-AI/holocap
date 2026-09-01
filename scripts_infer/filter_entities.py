"""Filter duplicate HoloCap entity predictions."""

import os
import json
import tqdm
import argparse
from collections import defaultdict
import numpy as np
import torch
import open3d as o3d
from pytorch3d.ops import box3d_overlap

parser = argparse.ArgumentParser("Filter duplicate HoloCap entities")
parser.add_argument('--model-name', type=str, required=True)
parser.add_argument('--old-attribute-path', type=str, default='entities')
parser.add_argument('--new-attribute-path', type=str, default='filtered_entities')
args = parser.parse_args()
model_name = args.model_name

org_root = os.path.join('results', model_name, args.old_attribute_path)
new_root = os.path.join('results', model_name, args.new_attribute_path)
os.makedirs(new_root, exist_ok=True)

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

def judge_boxtext(ibox, jbox, num=2):
    tot = 0
    for ib, jb in zip(ibox[:7], jbox[:7]):
        if abs(ib-jb) < 1e-6:
            tot += 1
    return tot >= num

class UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))

    def find(self, x):
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, x, y):
        rx, ry = self.find(x), self.find(y)
        if rx != ry:
            self.parent[ry] = rx

def find_representative(group_indices, all_boxes, group_id_map):
    best_idx = group_indices[0]
    best_score = float('inf') 

    for idx in group_indices:
        current_box = all_boxes[idx]
        max_iou_with_other_groups = 0.0
        for j, other_box in enumerate(all_boxes):
            if group_id_map[j] == group_id_map[idx]:
                continue  
            iou = boxiou_3d(current_box, other_box)
            if iou > max_iou_with_other_groups:
                max_iou_with_other_groups = iou
        if max_iou_with_other_groups < best_score:
            best_score = max_iou_with_other_groups
            best_idx = idx

    return best_idx

max_dump = 0
max_rate = 0
dump_list = []
json_list = sorted(os.listdir(org_root))
for json_name in tqdm.tqdm(json_list):
    json_path = os.path.join(org_root, json_name)
    save_path = os.path.join(new_root, json_name)
    if os.path.exists(save_path):
        continue
    with open(json_path) as f:
        org_list = json.load(f)
    
    n = len(org_list)
    classes = [ent['class_label'] for ent in org_list]
    boxes = [ent['bbox_3d'] for ent in org_list]

    uf = UnionFind(n)
    for i in range(n):
        for j in range(i + 1, n):
            if classes[i] != classes[j]:
                continue
            if boxiou_3d(boxes[i], boxes[j]) > 0.95 or judge_boxtext(boxes[i], boxes[j]):
                uf.union(i, j)

    group_ids = [uf.find(i) for i in range(n)]
    groups = defaultdict(list)
    for idx, gid in enumerate(group_ids):
        groups[gid].append(idx)

    keep_flags = [False] * n
    for group_indices in groups.values():
        if len(group_indices) == 1:
            keep_flags[group_indices[0]] = True
        else:
            rep_idx = find_representative(group_indices, boxes, group_ids)
            keep_flags[rep_idx] = True

    new_list = [org_list[i] for i in range(n) if keep_flags[i]]
    num_dump = n - len(new_list)

    max_dump = max(max_dump, num_dump)
    max_rate = max(max_rate, num_dump / n if n > 0 else 0)
    dump_list.append(num_dump)

    with open(save_path, 'w') as f:
        json.dump(new_list, f, indent=4)

print('Average number of removed instances:', sum(dump_list)/len(dump_list))
print('Max number of removed instances:', max_dump)
print('Max removal rate in one scene:', max_rate)


