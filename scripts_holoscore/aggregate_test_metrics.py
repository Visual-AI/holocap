import os 
import json
import argparse

import tqdm
import torch
import numpy as np

from scipy.optimize import linear_sum_assignment

parser = argparse.ArgumentParser()
parser.add_argument('-m', '--map-dir', type=str)
parser.add_argument('-ap', '--attr-pred-dir', type=str)
parser.add_argument('-rp', '--rel-pred-dir', type=str)
parser.add_argument('-as', '--attr-save-dir', type=str, required=True)
parser.add_argument('-rs', '--rel-save-dir', type=str, required=True)
parser.add_argument('-l', '--list', type=str, default='data/splits/test.json')
parser.add_argument('-a', '--anno', type=str, default='data/annotations/test.json')
parser.add_argument('--attribute-thr', type=float, default=0.6)
parser.add_argument('--relation-thr', type=float, default=0.6)
args = parser.parse_args()

attr_thr = args.attribute_thr
attr_save_root = args.attr_save_dir
rel_thr = args.relation_thr
rel_save_root = args.rel_save_dir

val_listfile = args.list
anno_jsonfile = args.anno

rel_prompt = '<|point_start|><|point_pad|><|point_end|>Describe relation between the two boxes as follows: <|layout_s|>bbox_0={}\nbbox_1={}<|layout_e|>'

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


def cal_f1(rec, prec):
    if rec == 0 or prec == 0:
        return 0.0
    return 2 * rec * prec / (rec+prec)


def count_successful_matches(cost_matrix, threshold):
    cost_matrix = np.asarray(cost_matrix)
    N, M = cost_matrix.shape

    work_matrix = -cost_matrix
    row_indices, col_indices = linear_sum_assignment(work_matrix)
    values = cost_matrix[row_indices, col_indices]
    success_mask = values >= threshold

    num_success = np.sum(success_mask)
    matched_N = int(num_success)
    matched_M = int(num_success)

    return matched_N, matched_M, row_indices, col_indices


def worker(sidx_list, gpu_id):
    tot_ent_recs  = []
    tot_ent_precs = []
    tot_ent_f1s   = []
    tot_loc_recs  = []
    tot_loc_precs = []
    tot_loc_f1s   = []
    tot_macro_attr_recs = []
    tot_macro_attr_precs = []
    tot_macro_attr_f1s = []
    tot_rel_recs = []
    tot_rel_precs = []
    tot_rel_f1s = []
    tot_rels = []
    for sidx in tqdm.tqdm(sidx_list):
        scene_name = val_list[sidx]
        anno = anno_dct[scene_name]
    
        attr_save_path = os.path.join(attr_save_root, '__'.join(scene_name.split('/'))+'.json')
        with open(attr_save_path) as f:
            attr_save_dct = json.load(f)
        rel_save_path = os.path.join(rel_save_root, '__'.join(scene_name.split('/'))+'.json')
        with open(rel_save_path) as f:
            rel_save_dct = json.load(f)
        map_path = os.path.join(args.map_dir, '__'.join(scene_name.split('/'))+'.json')
        with open(map_path) as f:
            map_dct = json.load(f)
        attr_pred_path = os.path.join(args.attr_pred_dir, '__'.join(scene_name.split('/'))+'.json')
        with open(attr_pred_path) as f:
            attr_pred_dct = json.load(f)
        rel_pred_path = os.path.join(args.rel_pred_dir, '__'.join(scene_name.split('/'))+'.json')
        with open(rel_pred_path) as f:
            rel_pred_dct = json.load(f)

        tot_ent_recs.append( map_dct['e-rec'])
        tot_ent_precs.append(map_dct['e-prec'])
        tot_ent_f1s.append(  map_dct['e-f1'])
        tot_loc_recs.append( map_dct['l-rec'])
        tot_loc_precs.append(map_dct['l-prec'])
        tot_loc_f1s.append(  map_dct['l-f1'])

        gt2pred_dct = map_dct['gt-to-pred mapping']
        gt2pred_dct = {int(k):v for k,v in gt2pred_dct.items()}
        pred2gt_dct = {v:k for k,v in gt2pred_dct.items()}
        if len(attr_pred_dct) == 0:
            tot_attr_recs.append(0.0)
            tot_attr_precs.append(0.0)
            tot_attr_f1s.append(0.0)
        else:
            for cand_idx, gt_idx in pred2gt_dct.items():
                assert attr_pred_dct[str(cand_idx)]['match_flag']
                assert attr_save_dct['gt'][str(gt_idx)]['match_flag']

                pred2gt_scores = torch.tensor(attr_save_dct['prediction'][str(cand_idx)]['pred2gt_scores'])
                gt2pred_scores = torch.tensor(attr_save_dct['prediction'][str(cand_idx)]['gt2pred_scores'])

                match_pred, _, _, _ = count_successful_matches(pred2gt_scores, attr_thr)
                match_gt, _, _, _ = count_successful_matches(gt2pred_scores, attr_thr)
                single_attr_prec = match_pred / gt2pred_scores.shape[-1]
                single_attr_rec = match_gt / pred2gt_scores.shape[-1]
                single_attr_f1 = cal_f1(single_attr_rec, single_attr_prec)

                attr_save_dct['prediction'][str(cand_idx)]['attr_rec'] =  single_attr_rec 
                attr_save_dct['prediction'][str(cand_idx)]['attr_prec'] = single_attr_prec
                attr_save_dct['prediction'][str(cand_idx)]['attr_f1'] =   single_attr_f1
                attr_save_dct['gt'][str(gt_idx)]['attr_rec'] =  single_attr_rec 
                attr_save_dct['gt'][str(gt_idx)]['attr_prec'] = single_attr_prec
                attr_save_dct['gt'][str(gt_idx)]['attr_f1'] =   single_attr_f1

            macro_attr_rec_list = []
            macro_attr_prec_list = []
            macro_attr_f1_list = []
            for gt_idx, dct in attr_save_dct['gt'].items():
                if dct['match_flag']:
                    macro_attr_rec_list.append(dct['attr_rec'])
                    macro_attr_prec_list.append(dct['attr_prec'])
                    macro_attr_f1_list.append(dct['attr_f1'])
                else:
                    macro_attr_rec_list.append(0.0)
                    macro_attr_prec_list.append(0.0)
                    macro_attr_f1_list.append(0.0)

        macro_attr_rec  = sum(macro_attr_rec_list) / len(macro_attr_rec_list)
        macro_attr_prec = sum(macro_attr_prec_list)/ len(macro_attr_prec_list)
        macro_attr_f1   = sum(macro_attr_f1_list)  / len(macro_attr_f1_list)
        tot_macro_attr_recs.append( macro_attr_rec)
        tot_macro_attr_precs.append(macro_attr_prec)
        tot_macro_attr_f1s.append(  macro_attr_f1)

        inst_dct = rel_pred_dct['instances']
        rel_list = rel_save_dct['prediction']['relation']
        gt_list = anno_dct[scene_name]['relation']
        rel_gt_dct = {}
        for rel in gt_list:
            idx1 = rel['index1']
            idx2 = rel['index2']
            assert idx1 != idx2
            if idx1 < idx2:
                rel_gt_dct[(idx1, idx2)] = {'relation': rel['relation'], 'reverse_relation': rel['reverse_relation'], 'match_flag': False}
            else:
                rel_gt_dct[(idx2, idx1)] = {'relation': rel['reverse_relation'], 'reverse_relation': rel['relation'], 'match_flag': False}

        if len(inst_dct) == 0:
            tot_rel_recs.append(0.0)
            tot_rel_precs.append(0.0)
            tot_rel_f1s.append(0.0)
        elif len(rel_gt_dct) == 0:
            tot_rel_recs.append(1.0)
            tot_rel_precs.append(1.0)
            tot_rel_f1s.append(1.0)
        else:
            num_rec = 0
            num_prec = 0
            new_list = []
            for rel in rel_list:
                idx1 = rel['index1']
                idx2 = rel['index2']
                assert rel['prompt'].split('as follows: ')[1] == rel_prompt.format(inst_dct[str(idx1)]['language_string'], inst_dct[str(idx2)]['language_string']).split('as follows: ')[1]
                pidx1 = idx1+1
                pidx2 = idx2+1
                if pidx1 in pred2gt_dct.keys() and pidx2 in pred2gt_dct.keys():
                    gidx1 = min(pred2gt_dct[pidx1], pred2gt_dct[pidx2])
                    gidx2 = max(pred2gt_dct[pidx1], pred2gt_dct[pidx2])
                    if (gidx1,gidx2) in rel_gt_dct.keys():
                        assert rel['match_flag'] == True
                        pred_rels = [rel['relation']]
                        gt_rels = [rel_gt_dct[(gidx1,gidx2)]['relation'], rel_gt_dct[(gidx1,gidx2)]['reverse_relation']]
                        pred2gt_scores = torch.tensor(rel['pred2gt_scores'])
                        gt2pred_scores = torch.tensor(rel['gt2pred_scores'])
                        rel_gt_dct[(gidx1,gidx2)] = True
                        if gt2pred_scores.max() >= rel_thr:
                            num_rec += 1
                        if pred2gt_scores.max() >= rel_thr:
                            num_prec += 1
                new_list.append(rel)

            tot_match = 0
            for rkey, rval in rel_gt_dct.items():
                if rkey[0] in gt2pred_dct.keys() and rkey[1] in gt2pred_dct.keys():
                    tot_match += 1
            tot_rels.append(tot_match/len(rel_gt_dct))

            rel_rec =  num_rec / len(rel_gt_dct) if len(rel_gt_dct)>0 else 0
            rel_prec = num_prec / len(rel_list) if len(rel_list)>0 else 0
            rel_f1 =   cal_f1(rel_rec, rel_prec)
                
            tot_rel_recs.append( rel_rec)
            tot_rel_precs.append(rel_prec)
            tot_rel_f1s.append(  rel_f1)
                
    tot_ent_rec  = sum(tot_ent_recs) / len(tot_ent_recs) 
    tot_ent_prec = sum(tot_ent_precs)/ len(tot_ent_precs)
    tot_ent_f1   = sum(tot_ent_f1s)  / len(tot_ent_f1s)
    tot_loc_rec  = sum(tot_loc_recs) / len(tot_loc_recs) 
    tot_loc_prec = sum(tot_loc_precs)/ len(tot_loc_precs)
    tot_loc_f1   = sum(tot_loc_f1s)  / len(tot_loc_f1s)
    tot_macro_attr_rec  = sum(tot_macro_attr_recs) / len(tot_macro_attr_recs) 
    tot_macro_attr_prec = sum(tot_macro_attr_precs)/ len(tot_macro_attr_precs) 
    tot_macro_attr_f1   = sum(tot_macro_attr_f1s)  / len(tot_macro_attr_f1s) 
    tot_rel_rec  = sum(tot_rel_recs) / len(tot_rel_recs) 
    tot_rel_prec = sum(tot_rel_precs)/ len(tot_rel_precs)
    tot_rel_f1   = sum(tot_rel_f1s)  / len(tot_rel_f1s)
    tot_rel = sum(tot_rels) / len(tot_rels)

    print('Ent Rec: {:.6f}'.format( tot_ent_rec))
    print('Ent Prec: {:.6f}'.format(tot_ent_prec))
    print('Ent F1: {:.6f}'.format(  tot_ent_f1))
    print('Loc Rec: {:.6f}'.format( tot_loc_rec))
    print('Loc Prec: {:.6f}'.format(tot_loc_prec))
    print('Loc F1: {:.6f}'.format(  tot_loc_f1))
    print('Att Rec: {:.6f}'.format( tot_macro_attr_rec))
    print('Att Prec: {:.6f}'.format(tot_macro_attr_prec))
    print('Att F1: {:.6f}'.format(  tot_macro_attr_f1))
    print('Rel Rec: {:.6f}'.format( tot_rel_rec))
    print('Rel Prec: {:.6f}'.format(tot_rel_prec))
    print('Rel F1: {:.6f}'.format(  tot_rel_f1))
    print('Max Rel: {:.6f}'.format(  tot_rel))


if __name__ == "__main__":
    sidx_list = list(range(len(val_list)))
    worker(sidx_list, 0)

