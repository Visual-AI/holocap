import os 
import json
import argparse

import tqdm
import torch

import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel

parser = argparse.ArgumentParser()
parser.add_argument('-m', '--map-dir', type=str)
parser.add_argument('-p', '--prediction-dir', type=str)
parser.add_argument('-s', '--save-dir', type=str, required=True)
parser.add_argument('-l', '--list', type=str, default='data/splits/test.json')
parser.add_argument('-a', '--anno', type=str, default='data/annotations/test.json')
parser.add_argument('--sim-thr', type=float, default=0.6)
args = parser.parse_args()

sim_thr = args.sim_thr
save_root = args.save_dir
os.makedirs(save_root, exist_ok=True)

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

# Each query must come with a one-sentence instruction that describes the task
task = 'Given a short search query describing the relationships, interactions or connections between two entities, retrieve passages that describe the same relationships, interactions or connections'
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
        pred_path = os.path.join(args.prediction_dir, '__'.join(scene_name.split('/'))+'.json')
        with open(pred_path) as f:
            pred_dct = json.load(f)

        gt2pred_dct = map_dct['gt-to-pred mapping']
        gt2pred_dct = {int(k):v for k,v in gt2pred_dct.items()}
        pred2gt_dct = {v:k for k,v in gt2pred_dct.items()}

        inst_dct = pred_dct['instances']
        rel_list = pred_dct['relation']

        save_dct = {}
        save_dct['prediction'] = pred_dct

        gt_list = anno_dct[scene_name]['relation']
        gt_dct = {}
        for rel in gt_list:
            idx1 = rel['index1']
            idx2 = rel['index2']
            assert idx1 != idx2
            if idx1 < idx2:
                gt_dct[(idx1, idx2)] = {'relation': rel['relation'], 'reverse_relation': rel['reverse_relation'], 'match_flag': False}
            else:
                gt_dct[(idx2, idx1)] = {'relation': rel['reverse_relation'], 'reverse_relation': rel['relation'], 'match_flag': False}

        if len(inst_dct) == 0:
            tot_recs.append(0.0)
            tot_precs.append(0.0)
            tot_f1s.append(0.0)
            with open(save_path, 'w') as f:
                json.dump(save_dct, f, indent=4)
            continue
        if len(gt_dct) == 0:
            tot_recs.append(1.0)
            tot_precs.append(1.0)
            tot_f1s.append(1.0)
            with open(save_path, 'w') as f:
                json.dump(save_dct, f, indent=4)
            continue

        num_rec = 0
        num_prec = 0
        new_list = []
        for ridx, rel in enumerate(rel_list):
            idx1 = rel['index1']
            idx2 = rel['index2']
            assert rel['prompt'].split('as follows: ')[1] == rel_prompt.format(inst_dct[str(idx1)]['language_string'], inst_dct[str(idx2)]['language_string']).split('as follows: ')[1]
            pidx1 = idx1+1
            pidx2 = idx2+1
            rel['match_flag'] = False
            if pidx1 in pred2gt_dct.keys() and pidx2 in pred2gt_dct.keys():
                gidx1 = min(pred2gt_dct[pidx1], pred2gt_dct[pidx2])
                gidx2 = max(pred2gt_dct[pidx1], pred2gt_dct[pidx2])
                if (gidx1,gidx2) in gt_dct.keys():
                    pred_rels = [rel['relation']]
                    gt_rels = [gt_dct[(gidx1,gidx2)]['relation'], gt_dct[(gidx1,gidx2)]['reverse_relation']]
                    pred2gt_scores = qwen3embed_similarity(pred_rels, gt_rels)
                    gt2pred_scores = qwen3embed_similarity(gt_rels, pred_rels)
                    gt_dct[(gidx1,gidx2)] = True
                    if gt2pred_scores.max() >= sim_thr:
                        num_rec += 1
                    if pred2gt_scores.max() >= sim_thr:
                        num_prec += 1
                    rel['pred2gt_scores'] = pred2gt_scores.tolist()
                    rel['gt2pred_scores'] = gt2pred_scores.tolist()
                    rel['match_flag'] = True
            new_list.append(rel)

        save_dct['prediction']['relation'] = new_list

        rec = num_rec / len(gt_dct) if len(gt_dct)>0 else 0
        prec = num_prec / len(rel_list) if len(rel_list)>0 else 0
        f1 = cal_f1(rec, prec)
            
        tot_recs.append(rec)
        tot_precs.append(prec)
        tot_f1s.append(f1)
                
        with open(save_path, 'w') as f:
            json.dump(save_dct, f, indent=4)

    tot_rec = sum(tot_recs)/len(tot_recs) 
    tot_prec = sum(tot_precs)/len(tot_precs) 
    tot_f1 = sum(tot_f1s)/len(tot_f1s) 
    print('Rec: {:.6f}'.format(tot_rec))
    print('Prec: {:.6f}'.format(tot_prec))
    print('F1: {:.6f}'.format(tot_f1))

if __name__ == "__main__":
    num_gpus = torch.cuda.device_count()

    sidx_list = list(range(len(val_list)))
    chunks = [[] for _ in range(num_gpus)]
    for i, item in enumerate(sidx_list):
        chunks[i % num_gpus].append(item)

    worker(sidx_list, 0)

