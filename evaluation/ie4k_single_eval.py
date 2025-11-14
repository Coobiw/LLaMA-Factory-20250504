import os
import json
from pathlib import Path
from tqdm import tqdm

from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info

import torch
from torch.utils.data import Dataset

import random
import numpy as np

from scipy import stats
import numpy as np
from scipy.optimize import curve_fit



def logistic_func(X, bayta1, bayta2, bayta3, bayta4):
    logisticPart = 1 + np.exp(np.negative(np.divide(X - bayta3, np.abs(bayta4))))
    yhat = bayta2 + np.divide(bayta1 - bayta2, logisticPart)
    return yhat

def fit_function(y_label, y_output):
    beta = [np.max(y_label), np.min(y_label), np.mean(y_output), 0.5]
    popt, _ = curve_fit(logistic_func, y_output, \
        y_label, p0=beta, maxfev=100000000)
    y_output_logistic = logistic_func(y_output, *popt)
    
    return y_output_logistic


def performance_fit(y_label, y_output, func_fit=True):
    if func_fit:
        y_output_logistic = fit_function(y_label, y_output)
    else:
        y_output_logistic = y_output
    PLCC = stats.pearsonr(y_output_logistic, y_label)[0]
    SRCC = stats.spearmanr(y_output, y_label)[0]

    return PLCC, SRCC, (PLCC+SRCC) / 2

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)

set_seed(42)

class IE4k(Dataset):

    def __init__(self, json_file: str):
        super().__init__()
        # Load data from JSON file
        with open(json_file, 'r', encoding='utf-8') as f:
            self.data = json.load(f)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Extract ground truth from assistant's response
        gt = float(item['messages'][1]['content'])
        
        # Get user message content (already formatted with prompt)
        user_content = item['messages'][0]['content']
        
        # Get image path
        image_path = item['images'][0]
        
        # Construct messages for model inference
        msgs = [
            {
                "role": "user", 
                "content": [
                    {"type": "image", "image": image_path},
                    {"type": "text", "text": user_content}
                ]
            },
            {
                "role": "assistant",
                "content": ""
            }
        ]
        
        return {
            "messages": msgs,
            "mos_perception": gt,
        }

def model_gen(model, processor, messages):
    # Preparation for inference
    texts = [
        processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=False)
        for msg in messages
    ]
    image_inputs, _ = process_vision_info(messages)
    inputs = processor(
        text=texts,
        images=image_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to("cuda")
    
    # Batch Inference: Generation of the output
    generated_ids = model.generate(**inputs, max_new_tokens=MAX_TOKENS, do_sample=False, temperature=0.)

    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_texts = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    return output_texts

if __name__ == "__main__":
    model_ckpt_dir = "/code/All-In-One/qbw/LLaMA-Factory-20250504/saves/qwen2p5_vl-7b/full/ie4k_single/checkpoint-141"
    
    model_name = "qwen25vl_ie_single_direct_e3"
    json_file = "./target_only_val.json"  # Path to the JSON file
    MAX_TOKENS=100
    
    print(model_ckpt_dir)
    print(model_name)
    print(f"JSON file: {json_file}")
    
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_ckpt_dir, 
        torch_dtype=torch.bfloat16, 
        device_map="cuda",
        attn_implementation="flash_attention_2",
    ).eval()
    
    max_pixels = 1048576 # 1024 x 1024
    min_pixels = 262144 # 512 x 512
    processor = AutoProcessor.from_pretrained("/code/All-In-One/qbw/EasyR1-20250410/cache/ckpt/Qwen2.5-VL-7B-Instruct", min_pixels=min_pixels, max_pixels=max_pixels)
    processor.tokenizer.padding_side  = 'left'

    ie_4k = IE4k(json_file=json_file)
    output = []
    output_fname = f"./{model_name}_ie4k.json"

    eval_bs = 1
    indices = list(range(len(ie_4k)))[::eval_bs]
    l = len(ie_4k)
    for start_idx in tqdm(indices):
        if start_idx + eval_bs > l:
            items = [ie_4k[idx] for idx in range(start_idx, l)]
        else:
            items = [ie_4k[idx] for idx in range(start_idx, start_idx + eval_bs)]

        batch_messages = [item['messages'] for item in items]
        model_responses = model_gen(model, processor, batch_messages)

        for response_idx, model_response in enumerate(model_responses):
            item = items[response_idx]
            item['model_response'] = model_response
            print(model_response)
            item.pop("messages")
            output.append(item)

    with open(output_fname, 'w') as fo:
        json.dump(output, fo, ensure_ascii=False, indent=4)
    
    y_label, y_out = [], []
    error_count = 0
    for i, item in enumerate(output):
        model_response = item['model_response']
        try:
            out = float(model_response.strip().rstrip("."))
            
            # Skip if extraction failed (returned None)
            if out is None:
                error_count += 1
                print(f"{i}th error:\t extraction returned None")
                continue
                
            y_out.append(out)
            y_label.append(float(item['mos_perception']))
        except Exception as e:
            error_count += 1
            print(f"{i}th error:\t", e)
            
    print(error_count)
    output1 = performance_fit(y_label, y_out, func_fit=True)
    output2 = performance_fit(y_label, y_out, func_fit=False)

    print(output1)
    print(output2)
    
    out_score = f"./{model_name}_ie4k_score.txt"
    with open(out_score, 'w') as fo:
        fo.write(f"PLCC: {output1[0]}\n")
        fo.write(f"SRCC: {output1[1]}\n")
        fo.write(f"MainScore: {output1[2]}\n")
        fo.write(f"PLCC: {output2[0]}\n")
        fo.write(f"SRCC: {output2[1]}\n")
        fo.write(f"MainScore: {output2[2]}\n")