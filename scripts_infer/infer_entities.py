"""Generate HoloCap entity and attribute predictions."""

import os
import json
import argparse

import torch
import numpy as np
from tqdm import tqdm
from threading import Thread
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import TextIteratorStreamer, set_seed

from spatiallm import Layout
from spatiallm.pcd import load_o3d_pcd, get_points_and_colors, cleanup_pcd, Compose

from transformers import logging
logging.set_verbosity_error()

detect_prompts = [
    '<|point_start|><|point_pad|><|point_end|>Detect floor and ceiling instances.',
    '<|point_start|><|point_pad|><|point_end|>Detect wall, door and window instances.',
    '<|point_start|><|point_pad|><|point_end|>Detect basic instances.',
    '<|point_start|><|point_pad|><|point_end|>Detect more instances except basic categories.',
]

attr_prompt = "<|point_start|><|point_pad|><|point_end|>Comprehensively describe the attribute of the instance as follows: <|layout_s|>bbox_0={}<|layout_e|>. Please output the description:"

def preprocess_point_cloud(points, colors, grid_size, num_bins):
    transform = Compose(
        [
            dict(type="PositiveShift"),
            dict(type="NormalizeColor"),
            dict(
                type="GridSample",
                grid_size=grid_size,
                hash_type="fnv",
                mode="test",
                keys=("coord", "color"),
                return_grid_coord=True,
                max_grid_coord=num_bins,
            ),
        ]
    )
    point_cloud = transform(
        {
            "name": "pcd",
            "coord": points.copy(),
            "color": colors.copy(),
        }
    )
    coord = point_cloud["grid_coord"]
    xyz = point_cloud["coord"]
    rgb = point_cloud["color"]
    point_cloud = np.concatenate([coord, xyz, rgb], axis=1)
    return torch.as_tensor(np.stack([point_cloud], axis=0))


def generate_layout(
    model,
    point_cloud,
    tokenizer,
    prompt,
    top_k=10,
    top_p=0.95,
    temperature=0.6,
    num_beams=1,
    seed=-1,
    max_new_tokens=4096,
):
    if seed >= 0:
        set_seed(seed)

    conversation = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt},
    ]

    input_ids = tokenizer.apply_chat_template(
        conversation, add_generation_prompt=True, return_tensors="pt"
    )
    input_ids = input_ids.to(model.device)

    streamer = TextIteratorStreamer(
        tokenizer, timeout=20.0, skip_prompt=True, skip_special_tokens=True
    )

    generate_kwargs = dict(
        {"input_ids": input_ids, "point_clouds": point_cloud},
        streamer=streamer,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        use_cache=True,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        num_beams=num_beams,
    )
    t = Thread(target=model.generate, kwargs=generate_kwargs)
    t.start()

    generate_texts = []
    for text in streamer:
        generate_texts.append(text)

    output_str = "".join(generate_texts)
    layout = Layout(output_str)
    layout.undiscretize_and_unnormalize(num_bins=model.config.point_config["num_bins"])
    return layout, output_str


def layout2entities(layout):
    dct_list = []
    for box in layout.bboxes:
        dct = {}
        dct['class_label'] = box.class_name
        dct['bbox_3d'] = [
            float(box.position_x), float(box.position_y), float(box.position_z),
            float(box.scale_x), float(box.scale_y), float(box.scale_z),
            float(box.angle_z), 0.0, 0.0,
        ]
        box_str = box.to_language_string()
        dct['language_string'] = 'Bbox' + box_str.split('=Bbox')[1]
        dct_list.append(dct)
    return dct_list


def worker(pcd_list, chunk_idx=0, args=None):
    # load the model
    from peft import PeftModel
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    
    # lora
    if os.path.exists(os.path.join(args.model_path, "adapter_config.json")):
        with open(os.path.join(args.model_path, "adapter_config.json")) as f:
            adapter_config = json.load(f)
        base_model_path = adapter_config.get("base_model_name_or_path", None)
        if base_model_path is None:
            raise ValueError("adapter_config.json does not contain 'base_model_name_or_path'")
    
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            torch_dtype=getattr(torch, args.inference_dtype),
            trust_remote_code=True,  
        )
        model = PeftModel.from_pretrained(base_model, args.model_path)
    else:
        raise NotImplementedError

    model.to("cuda:{}".format(0))
    model.set_point_backbone_dtype(torch.float32)
    model.eval()

    # number of bins used for discretization
    num_bins = model.config.point_config["num_bins"]

    for pcd_name, point_cloud_file in tqdm(pcd_list, desc=f'Chunk {chunk_idx}'):
        save_path = os.path.join(args.output, '__'.join(pcd_name.split('/'))+'.json')
        if os.path.exists(save_path):
            continue
        
        # load the point cloud
        point_cloud = load_o3d_pcd(point_cloud_file)
        grid_size = Layout.get_grid_size(num_bins)

        if not args.no_cleanup:
            point_cloud = cleanup_pcd(point_cloud, voxel_size=grid_size)

        points, colors = get_points_and_colors(point_cloud)
        min_extent = np.min(points, axis=0)

        # preprocess the point cloud to tensor features
        input_pcd = preprocess_point_cloud(points, colors, grid_size, num_bins)

        entity_list = []
        for prompt in detect_prompts:
            # generate the layout
            layout, _ = generate_layout(
                model, input_pcd, tokenizer, prompt,
                top_k=args.top_k, top_p=args.top_p,
                temperature=args.temperature, num_beams=args.num_beams,
                seed=args.seed,
            )
            layout.translate(min_extent)
            tmp_list = layout2entities(layout)
            entity_list += tmp_list

        entity_wattr_list = []
        for eidx, entity in enumerate(entity_list):
            _, attr_str = generate_layout(
                model, input_pcd, tokenizer, attr_prompt.format(entity['language_string']),
                top_k=args.top_k, top_p=args.top_p,
                temperature=args.temperature, num_beams=args.num_beams,
                seed=args.seed,
            )
            entity['caption'] = attr_str
            entity['entity_id'] = eidx+1
            entity_wattr_list.append(entity)

        with open(save_path, 'w') as f:
            json.dump(entity_wattr_list, f, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("HoloCap entity inference")
    parser.add_argument("--num-chunk", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument(
        "-l",
        "--list-file",
        type=str,
        required=True,
        help="Path to the input list file containing the names of point cloud files",
    )
    parser.add_argument(
        "-p",
        "--point-cloud",
        type=str,
        required=True,
        help="Path to the input point cloud file or a folder containing multiple point cloud files",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        required=True,
        help="Path to the output layout txt file or a folder to save multiple layout txt files",
    )
    parser.add_argument(
        "-m",
        "--model-path",
        type=str,
        required=True,
        help="Path to the model checkpoint",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="The number of highest probability vocabulary tokens to keep for top-k filtering",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.95,
        help="The smallest set of most probable tokens with probabilities that add up to top_p or higher are kept",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.6,
        help="The value used to module the next token probabilities",
    )
    parser.add_argument(
        "--num-beams",
        type=int,
        default=1,
        help="The number of beams for beam search",
    )
    parser.add_argument(
        "--inference-dtype",
        type=str,
        default="bfloat16",
        help="The torch dtype to use for inference, bfloat16 or float32",
    )
    parser.add_argument(
        "--no-cleanup",
        default=False,
        action="store_true",
        help="Whether to not cleanup the point cloud",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="The seed to use during inference, negative value means no seed",
    )
    args = parser.parse_args()
    os.makedirs(args.output, exist_ok=True)

    # check if the input is a single point cloud file or a folder containing multiple point cloud files
    if os.path.isfile(args.point_cloud):
        point_cloud_files = [args.point_cloud]
    else:
        with open(args.list_file) as f:
            pcd_list = json.load(f)
        point_cloud_files = []
        for pcd_name in pcd_list:
            if 'scannet' in pcd_name:
                ppath = os.path.join(args.point_cloud, 'scannet_pcds', '{}_vh_clean_axisaligned_points.ply'.format(pcd_name.split('/')[-1]))
            elif '3rscan' in pcd_name:
                ppath = os.path.join(args.point_cloud, '3rscan_pcds', '{}_axisaligned_points.ply'.format(pcd_name.split('/')[-1]))
            elif 'matterport3d' in pcd_name:
                ppath = os.path.join(args.point_cloud, 'matterport3d_pcds', '{}_axisaligned_points.ply'.format(pcd_name.split('/')[-1]))
            elif 'arkitscenes' in pcd_name:
                ppath = os.path.join(args.point_cloud, 'arkitscenes_pcds', '{}_axisaligned_points.ply'.format(pcd_name.split('/')[-1]))
            else:
                raise NotImplementedError
            point_cloud_files.append((pcd_name, ppath))
    
    chunks = [[] for _ in range(args.num_chunk)]
    for i, item in enumerate(point_cloud_files):
        chunks[i % args.num_chunk].append(item)
    worker(chunks[args.chunk_idx], args.chunk_idx, args)

