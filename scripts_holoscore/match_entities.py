import os
import json
import tqdm
import argparse
import numpy as np
import open3d as o3d
import torch
from pytorch3d.ops import box3d_overlap

from scipy.optimize import linear_sum_assignment
from nltk.corpus import wordnet as wn
from sentence_transformers import SentenceTransformer
from factual_scene_graph.evaluation.soft_spice_evaluation import encode_phrases
text_encoder = SentenceTransformer("sentence-transformers/all-mpnet-base-v2").to('cuda:0').eval()

parser = argparse.ArgumentParser()
parser.add_argument('-m', '--map-dir', type=str)
parser.add_argument('-p', '--prediction-dir', type=str)
parser.add_argument('-l', '--list', type=str, default='data/splits/val.json')
parser.add_argument('-a', '--anno', type=str, default='data/annotations/val.json')
parser.add_argument('--entity-thr', type=float, default=0.5)
parser.add_argument('--location-thr', type=float, default=0.3)
args = parser.parse_args()

save_root = args.map_dir
os.makedirs(save_root, exist_ok=True)

val_listfile = args.list
anno_jsonfile = args.anno
pred_root = args.prediction_dir

entity_thr = args.entity_thr
locate_thr = args.location_thr

with open(val_listfile) as f:
    val_list = json.load(f)

anno_dct = {}
with open(anno_jsonfile) as f:
    anno_list = json.load(f)
    for anno in anno_list:
        scene_id = anno['scene_id']
        if 'scene' in scene_id:
            scene_name = f'scannet/{scene_id}'
        elif '-' in scene_id:
            scene_name = f'3rscan/{scene_id}'
        elif 'region' in scene_id:
            scene_name = f'matterport3d/{scene_id}'
        else:
            assert '4' in scene_id
            scene_name = f'arkitscenes/{scene_id}'
        anno_dct[scene_name] = anno 

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

def get_synonyms(word, pos=None):
    if pos is not None:
        synsets = wn.synsets(word, pos)
    else:
        synsets = wn.synsets(word)
    synonyms = set()
    for synset in synsets:
        for lemma in synset.lemmas():
            synonyms.add(lemma.name())
    return synonyms

def compute_synonyms_score(word1, word2, pos=None):
    word1 = word1.strip()
    word2 = word2.strip()
    if not word1 or not word2:
        return 0.0
    if word1 in word2 or word2 in word1:
        return 1

    synonyms1 = set({})
    for w in word1.split():
        synonyms1 |= get_synonyms(w, pos)
    synonyms2 = set({})
    for w in word2.split():
        synonyms2 |= get_synonyms(w, pos)
    iou = len(synonyms1.intersection(synonyms2)) / (len(synonyms1.union(synonyms2)) + 1e-6)
    return iou

def eval_entity(cand_dct, gt_dct):
    if len(cand_dct) == 0:
        return 0, 0, 0, 0, {}

    # calculate sentence embeddings and similarities for soft matching
    gval_list = [gt_dct[gidx+1]['class_label'] for gidx in range(len(gt_dct))]
    cval_list = [cand_dct[cidx+1]['class_label'] for cidx in range(len(cand_dct))]
    gfeats, cfeats = encode_phrases(text_encoder, gval_list, cval_list, batch_size=1)
    sent_sim = gfeats.dot(cfeats.T)
    # calculate one-to-one mapping
    gval_list = [(gidx+1, gt_dct[gidx+1]['class_label']) for gidx in range(len(gt_dct))]
    cval_list = [(cidx+1, cand_dct[cidx+1]['class_label']) for cidx in range(len(cand_dct))]
    cost_mat = np.zeros((len(gval_list), len(cval_list)))
    loc_flag = np.zeros((len(gval_list), len(cval_list)))
    for ii, (gidx, gval) in enumerate(gval_list):
        gloc = gt_dct[gidx]['bbox_3d']
        for jj, (cidx, cval) in enumerate(cval_list):
            cloc = cand_dct[cidx]['bbox_3d']
            if gval == cval:                                                    # entity name
                cost_mat[ii, jj] = -1000
            else:
                syn_score = compute_synonyms_score(gval, cval, wn.NOUN)
                if syn_score > 0:                                               # entity name synonyms
                    cost_mat[ii, jj] = -100 * syn_score
                elif sent_sim[ii, jj] >= entity_thr:                                  # similarity between sentence embeddings larger than a threshold
                    cost_mat[ii, jj] = -10 * sent_sim[ii, jj]
                else:                                                           # not matched
                    cost_mat[ii, jj] = 1e6
            if cost_mat[ii, jj] < 0:      # location
                biou = boxiou_3d(gloc, cloc)
                if biou >= locate_thr:
                    cost_mat[ii, jj] += -1 * biou
                    loc_flag[ii, jj] = 1
    gt_idx, cand_idx = linear_sum_assignment(cost_mat)

    # calculate entity score based on one-to-one mapping
    erec = 0
    eprec = 0
    lrec = 0
    lprec = 0
    emap = {}
    for ii, jj in zip(gt_idx, cand_idx):
        if cost_mat[ii, jj] < 0:
            erec += 1
            eprec += 1
            if loc_flag[ii, jj] > 0.5:
                lrec += 1
                lprec += 1
            emap[gval_list[ii][0]] = cval_list[jj][0]

    erec /= len(gt_dct)
    eprec /= len(cand_dct)
    lrec /= len(gt_dct)
    lprec /= len(cand_dct)

    return erec, eprec, lrec, lprec, emap


def cal_f1(rec, prec):
    if rec == 0 or prec == 0:
        return 0.0
    return 2 * rec * prec / (rec+prec)

num_good = 0
good_list = []

erec_list = []
eprec_list = []
ef1_list = []
lrec_list = []
lprec_list = []
lf1_list = []
for scene_name in tqdm.tqdm(val_list):
    anno = anno_dct[scene_name]

    save_path = os.path.join(save_root, '__'.join(scene_name.split('/'))+'.json')
    if os.path.exists(save_path):
        continue

    pred_file = os.path.join(pred_root, '__'.join(scene_name.split('/'))+'.json')
    with open(pred_file) as f:
        pred_list = json.load(f)
    pred_dct = {}
    inst_idx = 0
    for pred in pred_list:
        pred_dct[inst_idx+1] = pred
        inst_idx += 1
        assert 1 <= inst_idx <= len(pred_list)
    assert len(pred_list) == len(pred_dct)

    gt_dct = {}
    for str_idx, dct in anno['instances'].items():
        inst_idx = int(str_idx)
        gt_dct[inst_idx] = dct
        assert 1 <= inst_idx <= len(anno['instances'])
    assert len(gt_dct) == len(anno['instances'])

    erec, eprec, lrec, lprec, emap = eval_entity(pred_dct, gt_dct)
    ef1 = cal_f1(erec, eprec)
    lf1 = cal_f1(lrec, lprec)

    save_dct = {
        'pred': pred_dct,
        'gt': gt_dct,
        'e-rec': erec,
        'e-prec': eprec, 
        'e-f1': ef1, 
        'l-rec': lrec,
        'l-prec': lprec, 
        'l-f1': lf1, 
        'gt-to-pred mapping': emap,
    }
    with open(save_path, 'w') as f:
        json.dump(save_dct, f, indent=4)

    erec_list.append(erec)
    eprec_list.append(eprec)
    ef1_list.append(ef1)
    lrec_list.append(lrec)
    lprec_list.append(lprec)
    lf1_list.append(lf1)

    if lf1 > 0.5:
        num_good += 1
        good_list.append(scene_name)

print('erec:', sum(erec_list)/len(erec_list))
print('eprec:', sum(eprec_list)/len(eprec_list))
print('ef1:', sum(ef1_list)/len(ef1_list))
print('lrec:', sum(lrec_list)/len(lrec_list))
print('lprec:', sum(lprec_list)/len(lprec_list))
print('lf1:', sum(lf1_list)/len(ef1_list))

print('number of good samples:', num_good)


