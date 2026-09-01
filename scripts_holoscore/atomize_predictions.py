import os 
import json
import argparse

import tqdm
import torch
import copy
import multiprocessing as mp

from transformers import AutoTokenizer, Gemma3ForCausalLM, BitsAndBytesConfig

torch.set_float32_matmul_precision('high')

parser = argparse.ArgumentParser()
parser.add_argument('-m', '--map-dir', type=str)
parser.add_argument('-gt', '--gt-dir', type=str)
parser.add_argument('-p', '--prediction-dir', type=str)
parser.add_argument('-pa', '--prediction-atom-dir', type=str)
parser.add_argument('-l', '--list', type=str, default='data/splits/val.json')
parser.add_argument('-a', '--anno', type=str, default='data/annotations/val.json')
parser.add_argument('-i', '--run-idx', type=int, default=0)
parser.add_argument('-n', '--num-idx', type=int, default=100)
args = parser.parse_args()

save_root = args.prediction_atom_dir
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

text_prompt = "Break down an input paragraph describing a specific entity's attributes into a list of short, atomic sentences. These attributes include but are not limited to color, material, shape, texture, state, style and the entity's constituent parts. Each sentence should express a single, self-contained attribute according to the given paragraph. Please avoid combining multiple attributes in one sentence." + \
"\nAn example is given as follows: " + \
"\nGiven a paragraph describing a window entity: " + \
"\"The window is a multi-paned structure featuring a grid of rectangular glass panes. The window frame is composed of a light-colored wood material, exhibiting a smooth texture.\"" + \
", you should transform the above paragraph into a list of sentences following the below format:\n" + \
"\"" + \
"1) The window is a multi-paned structure.\n" + \
"2) The window has a grid of rectangular glass panes.\n" + \
"3) The window has a light-colored frame.\n" + \
"4) The window has a wood frame.\n" + \
"5) The window\'s frame has a smooth texture.\n" + \
"\". " + \
"The input paragraph describing a/an {} entity is given as follows: {}\n" + \
"Do not add, infer, or include any information that is not explicitly stated in the original paragraph. You should fully understand this example to learn the required way to extract information and the formats of outputted lines. Please output the breakdown of the paragraph directly, that is the output text should only consist of the breakdown of the paragraph and should not include other things like prompting words."

def worker(sidx_list, gpu_id):
    # initialize MLLM and image processor
    model_name = "google/gemma-3-1b-it"
    quantization_config = BitsAndBytesConfig(load_in_8bit=True)
    model = Gemma3ForCausalLM.from_pretrained(
        model_name, quantization_config=quantization_config,
        device_map={"": "cuda:{}".format(gpu_id)}
    ).eval()
    tokenizer = AutoTokenizer.from_pretrained(model_name)

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
            pred_list = json.load(f)
        pred_dct = {}
        inst_idx = 0
        for pred in pred_list:
            pred_dct[inst_idx+1] = pred
            inst_idx += 1
            assert 1 <= inst_idx <= len(pred_list)
        assert len(pred_list) == len(pred_dct)

        gt2pred_dct = map_dct['gt-to-pred mapping']
        gt2pred_dct = {int(k):v for k,v in gt2pred_dct.items()}
        pred2gt_dct = {v:k for k,v in gt2pred_dct.items()}

        save_dct = copy.deepcopy(pred_dct)
        for cand_idx, _ in save_dct.items():
            save_dct[cand_idx]['match_flag'] = False
        for cand_idx, gt_idx in pred2gt_dct.items():
            label = pred_dct[cand_idx]['class_label']
            pred_caption = pred_dct[cand_idx]['caption']
            messages = [
                {
                    "role": "system",
                    "content": [{"type": "text", "text": "You are a helpful assistant."}]
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": text_prompt.format(label, pred_caption)}
                    ]
                }
            ]
            
            # Preparation for inference
            inputs = tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=True,
                return_dict=True, return_tensors="pt"
            ).to(model.device)#.to(torch.bfloat16)
            
            input_len = inputs["input_ids"].shape[-1]
            with torch.inference_mode():
                generation = model.generate(**inputs, max_new_tokens=500)
                generation = generation[0][input_len:]
            
            output_text = tokenizer.decode(generation)
            output_text = output_text.replace('<end_of_turn>', '')
    
            save_dct[cand_idx]['match_flag'] = True
            save_dct[cand_idx]['atom_attribute'] = output_text
        
        with open(save_path, 'w') as f:
            json.dump(save_dct, f, indent=4)


if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)  

    num_gpus = torch.cuda.device_count()

    sidx_list = list(range(args.run_idx*args.num_idx, (args.run_idx+1)*args.num_idx))
    chunks = [[] for _ in range(num_gpus)]
    for i, item in enumerate(sidx_list):
        chunks[i % num_gpus].append(item)

    processes = []
    for gpu_id in range(num_gpus):
        sub_sidx_list = [ii for ii in chunks[gpu_id] if ii < len(val_list)]
        if len(sub_sidx_list) == 0:
            continue
        p = mp.Process(target=worker, args=(sub_sidx_list, gpu_id))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

