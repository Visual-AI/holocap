import os 
import json
import argparse

import tqdm
import torch
import numpy as np
import multiprocessing as mp

import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from scipy.optimize import linear_sum_assignment

parser = argparse.ArgumentParser()
parser.add_argument('-m', '--map-dir', type=str)
parser.add_argument('-gt', '--gt-dir', type=str)
parser.add_argument('-p', '--prediction-dir', type=str)
parser.add_argument('-s', '--save-dir', type=str, required=True)
parser.add_argument('-l', '--list', type=str, default='data/splits/val.json')
parser.add_argument('-a', '--anno', type=str, default='data/annotations/val.json')
parser.add_argument('--sim-thr', type=float, default=0.6)
args = parser.parse_args()

sim_thr = args.sim_thr
save_root = args.save_dir
os.makedirs(save_root, exist_ok=True)

val_listfile = args.list
anno_jsonfile = args.anno

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

# Each query must come with a one-sentence instruction that describes the task
task = 'Given a short search query describing an entity attribute, retrieve passages that describe the same entity with the same attribute'
def get_detailed_instruct(task_description: str, query: str) -> str:
    return f'Instruct: {task_description}\nQuery:{query}'

def last_token_pool(last_hidden_states: torch.Tensor,
                 attention_mask: torch.Tensor) -> torch.Tensor:
    left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
    if left_padding:
        return last_hidden_states[:, -1]
    else:
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_states.shape[0]
        return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]

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
    # initialize MLLM and image processor
    tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen3-Embedding-0.6B', padding_side='left')
    embed_model_name = 'Qwen/Qwen3-Embedding-0.6B'
    embed_model = AutoModel.from_pretrained(embed_model_name, 
        attn_implementation="flash_attention_2", 
        torch_dtype=torch.float16,
        device_map='cuda:{}'.format(gpu_id))

    def qwen3embed_similarity(cand_strings, gt_strings):
        max_length = 8192
        ns = len(cand_strings)
        queries = [get_detailed_instruct(task, string) for string in cand_strings]
        input_texts = queries + gt_strings
        
        # Tokenize the input texts
        batch_dict = tokenizer(
            input_texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        batch_dict.to(embed_model.device)
        outputs = embed_model(**batch_dict)
        embeddings = last_token_pool(outputs.last_hidden_state, batch_dict['attention_mask'])
        
        embeddings = F.normalize(embeddings, p=2, dim=1)
        scores = (embeddings[:ns] @ embeddings[ns:].T)
        return scores

    tot_recs = []
    tot_precs = []
    tot_f1s = []
    for sidx in tqdm.tqdm(sidx_list):
        scene_name = val_list[sidx]
        anno = anno_dct[scene_name]
    
        save_path = os.path.join(save_root, '__'.join(scene_name.split('/'))+'.json')
        if os.path.exists(save_path):
            continue
        
        map_path = os.path.join(args.map_dir, '__'.join(scene_name.split('/'))+'.json')
        with open(map_path) as f:
            map_dct = json.load(f)
        gt_path = os.path.join(args.gt_dir, '__'.join(scene_name.split('/'))+'.json')
        with open(gt_path) as f:
            gt_dct = json.load(f)
        pred_path = os.path.join(args.prediction_dir, '__'.join(scene_name.split('/'))+'.json')
        with open(pred_path) as f:
            pred_dct = json.load(f)

        gt2pred_dct = map_dct['gt-to-pred mapping']
        gt2pred_dct = {int(k):v for k,v in gt2pred_dct.items()}
        pred2gt_dct = {v:k for k,v in gt2pred_dct.items()}

        save_dct = {}
        save_dct['prediction'] = pred_dct
        for cand_idx, _ in save_dct['prediction'].items():
            save_dct['prediction'][cand_idx]['attr_rec'] = 0.0
            save_dct['prediction'][cand_idx]['attr_prec'] = 0.0
            save_dct['prediction'][cand_idx]['attr_f1'] = 0.0
        save_dct['gt'] = gt_dct
        for gt_idx, _ in gt_dct.items():
            save_dct['gt'][gt_idx]['match_flag'] = False 

        if len(pred_dct) == 0 or len(gt_dct) == 0:
            tot_recs.append(0.0)
            tot_precs.append(0.0)
            tot_f1s.append(0.0)
            with open(save_path, 'w') as f:
                json.dump(save_dct, f, indent=4)
            continue
            
        for cand_idx, gt_idx in pred2gt_dct.items():
            label = pred_dct[str(cand_idx)]['class_label']
            assert pred_dct[str(cand_idx)]['match_flag']
            assert save_dct['prediction'][str(cand_idx)]['match_flag']
            save_dct['gt'][str(gt_idx)]['match_flag'] = True
            save_dct['prediction'][str(cand_idx)]['match_gtidx'] = gt_idx

            pred_atom_attr = pred_dct[str(cand_idx)]['atom_attribute']
            gt_atom_attr = gt_dct[str(gt_idx)]['atom_attribute']

            def get_attr_list(gt_atom_attr):
                attrs = gt_atom_attr.strip('\n').split('\n')
                attrs = [') '.join(attr.split(') ')[1:]) for attr in attrs]
                return attrs
            
            # Retained for compatibility with the published HoloScore results.
            pred_atom_attr_list = get_attr_list(pred_atom_attr)[:-1]
            gt_atom_attr_list = get_attr_list(gt_atom_attr)
            pred2gt_scores = qwen3embed_similarity(pred_atom_attr_list, gt_atom_attr_list)
            gt2pred_scores = qwen3embed_similarity(gt_atom_attr_list, pred_atom_attr_list)

            match_pred, _, m_cidxs, m_gidxs = count_successful_matches(pred2gt_scores.cpu().detach().numpy(), sim_thr)
            match_gt, _, _, _ = count_successful_matches(gt2pred_scores.cpu().detach().numpy(), sim_thr)
            single_prec = match_pred / gt2pred_scores.shape[-1] if gt2pred_scores.shape[-1] else 0.0
            single_rec = match_gt / pred2gt_scores.shape[-1]
            single_f1 = cal_f1(single_rec, single_prec)

            save_dct['prediction'][str(cand_idx)]['attr_rec'] = single_rec 
            save_dct['prediction'][str(cand_idx)]['attr_prec'] = single_prec
            save_dct['prediction'][str(cand_idx)]['attr_f1'] = single_f1
            save_dct['prediction'][str(cand_idx)]['num_pred_atom_attr'] = pred2gt_scores.shape[0]
            save_dct['prediction'][str(cand_idx)]['num_match_gt_atom_attr'] = pred2gt_scores.shape[1]
            save_dct['prediction'][str(cand_idx)]['pred2gt_scores'] = pred2gt_scores.tolist()
            save_dct['prediction'][str(cand_idx)]['gt2pred_scores'] = gt2pred_scores.tolist()
            save_dct['gt'][str(gt_idx)]['attr_rec'] =  single_rec 
            save_dct['gt'][str(gt_idx)]['attr_prec'] = single_prec
            save_dct['gt'][str(gt_idx)]['attr_f1'] =   single_f1

        rec_list = []
        prec_list = []
        f1_list = []
        for gt_idx, dct in save_dct['gt'].items():
            if dct['match_flag']:
                rec_list.append(dct['attr_rec'])
                prec_list.append(dct['attr_prec'])
                f1_list.append(dct['attr_f1'])
            else:
                rec_list.append(0.0)
                prec_list.append(0.0)
                f1_list.append(0.0)

        rec  = sum(rec_list) / len(rec_list)
        prec = sum(prec_list)/ len(prec_list)
        f1   = sum(f1_list)  / len(f1_list)
        tot_recs.append( rec)
        tot_precs.append(prec)
        tot_f1s.append(  f1)
                
        with open(save_path, 'w') as f:
            json.dump(save_dct, f, indent=4)

    tot_rec = sum(tot_recs)/len(tot_recs) if len(tot_recs) != 0 else 0.0
    tot_prec = sum(tot_precs)/len(tot_precs) if len(tot_precs) != 0 else 0.0
    tot_f1 = sum(tot_f1s)/len(tot_f1s) if len(tot_f1s) != 0 else 0.0
    print('Rec: {:.6f}'.format(tot_rec))
    print('Prec: {:.6f}'.format(tot_prec))
    print('F1: {:.6f}'.format(tot_f1))

if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)  

    num_gpus = torch.cuda.device_count()

    sidx_list = list(range(len(val_list)))
    chunks = [[] for _ in range(num_gpus)]
    for i, item in enumerate(sidx_list):
        chunks[i % num_gpus].append(item)

    worker(sidx_list, 0)

