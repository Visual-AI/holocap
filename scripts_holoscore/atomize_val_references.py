import os 
import json
import argparse

import tqdm
import torch
import multiprocessing as mp

torch.set_float32_matmul_precision('high')

val_listfile = 'data/splits/val.json'
anno_jsonfile = 'data/annotations/val.json'
save_root = './results_eval/val/reference_atomic_attributes/'
os.makedirs(save_root, exist_ok=True)


def get_save_path(scene_name):
    return os.path.join(save_root, '__'.join(scene_name.split('/')) + '.json')

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
    from transformers import AutoTokenizer, Gemma3ForCausalLM, BitsAndBytesConfig

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
    
        save_path = get_save_path(scene_name)
        if os.path.exists(save_path):
            continue
        save_dct = {}
        for str_idx, dct in tqdm.tqdm(anno['instances'].items()):
            label = dct['class_label']
            caption = dct['attribute']
            messages = [
                {
                    "role": "system",
                    "content": [{"type": "text", "text": "You are a helpful assistant."}]
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": text_prompt.format(label, caption)}
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
    
            dct['atom_attribute'] = output_text
            save_dct[str_idx] = dct
        
        with open(save_path, 'w') as f:
            json.dump(save_dct, f, indent=4)


if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)  

    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--run-idx', type=int, default=0)
    parser.add_argument('-n', '--num-idx', type=int, default=100)
    args = parser.parse_args()
    sidx_list = list(range(args.run_idx*args.num_idx, (args.run_idx+1)*args.num_idx))
    sidx_list = [
        sidx for sidx in sidx_list
        if sidx < len(val_list) and not os.path.exists(get_save_path(val_list[sidx]))
    ]
    if not sidx_list:
        print(f'Reference atomic attributes are complete: {save_root}')
        raise SystemExit(0)

    num_gpus = torch.cuda.device_count()
    if num_gpus == 0:
        raise RuntimeError('Reference atomization requires at least one CUDA device.')

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

    failed_exit_codes = [p.exitcode for p in processes if p.exitcode != 0]
    if failed_exit_codes:
        raise RuntimeError(f'Reference atomization workers failed: {failed_exit_codes}.')

