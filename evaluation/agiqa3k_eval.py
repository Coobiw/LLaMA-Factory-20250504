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

class AGIQA3k(Dataset):

    def __init__(self, annos, sys_prompt, query_format):
        super().__init__()
        self.annos = []
        with open(annos) as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip()
                item = json.loads(line)
                self.annos.append(item)
            self.sys_prompt = sys_prompt
            self.user_query_format = query_format

    def __len__(self):
        return len(self.annos)

    def __getitem__(self,idx):
        item = self.annos[idx]
        mos_perception = item['messages'].pop(1)['content']
        return {
            "messages": [
                item['messages'][0]
            ],
            "mos_perception": float(mos_perception),
        }

def model_gen(model, processor, messages):
    # Preparation for inference
    texts = [
        processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
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
    generated_ids = model.generate(**inputs, max_new_tokens=2048, do_sample=False, temperature=0.)

    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_texts = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    return output_texts


if __name__ == "__main__":
    home_dir = "/code/All-In-One/qbw/LLaMA-Factory-20250504/saves/qwen2p5_vl-7b/full/agiqa3k_qual_sft_idx{}/checkpoint-76"
    
    for model_idx in range(10):
        model_path = home_dir.format(model_idx)
        model_name = f"agiqa3k_qual_sft_idx{model_idx}"
        os.makedirs(f"{model_path}/eval_results", exist_ok=True)
        print(model_path)
        print(model_name)
        
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path, 
            torch_dtype=torch.bfloat16, 
            device_map="cuda",
            attn_implementation="flash_attention_2",
        ).eval()
        
        
        max_pixels = 1048576 # 1024 x 1024
        min_pixels = 262144 # 512 x 512
        processor = AutoProcessor.from_pretrained("/code/All-In-One/qbw/EasyR1-20250410/cache/ckpt/Qwen2.5-VL-7B-Instruct", min_pixels=min_pixels, max_pixels=max_pixels)
        processor.tokenizer.padding_side  = 'left'

        annos = f"/code/All-In-One/qbw/LLaMA-Factory-20250504/data/agiqa3k/annos_idx{model_idx}/test.jsonl"
        sys_prompt = ''
    
        return_dtype, lower_bound, upper_bound  = "float", "1", "5"
        mid_prompt = " rounded to two decimal places," if return_dtype == "float" else ""
        query_format = 'What is your overall rating on the quality of this AI-generated picture?' + f' The rating should be a {return_dtype} between {lower_bound} and {upper_bound},{mid_prompt} with {lower_bound} representing very poor quality and {upper_bound} representing excellent quality. Return the final answer directly.\n\n'

        agiqa_3k = AGIQA3k(annos, sys_prompt, query_format)
        output = []
        output_fname = f"{model_path}/eval_results/agiqa3k_{model_name}_{return_dtype}_{lower_bound}_{upper_bound}.json"

        eval_bs = 128
        indices = list(range(len(agiqa_3k)))[::eval_bs]
        l = len(agiqa_3k)
        for start_idx in tqdm(indices):
            if start_idx + eval_bs > l:
                items = [agiqa_3k[idx] for idx in range(start_idx, l)]
            else:
                items = [agiqa_3k[idx] for idx in range(start_idx, start_idx + eval_bs)]

            batch_messages = [item['messages'] for item in items]
            model_responses = model_gen(model, processor, batch_messages)

            for response_idx, model_response in enumerate(model_responses):
                item = items[response_idx]
                item['model_response'] = model_response
                output.append(item)

        with open(output_fname, 'w') as fo:
            json.dump(output, fo, ensure_ascii=False, indent=4)
        
        y_label, y_out = [], []
        error_count = 0
        for i, item in enumerate(output):
            model_response = item['model_response']
            try:    
                out = float(model_response.strip())
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
        
        out_score = f"{model_path}/eval_results/agiqa3k_{model_name}_{return_dtype}_{lower_bound}_{upper_bound}_score.txt"
        with open(out_score, 'w') as fo:
            fo.write(f"PLCC: {output1[0]}\n")
            fo.write(f"SRCC: {output1[1]}\n")
            fo.write(f"MainScore: {output1[2]}\n")
            fo.write(f"PLCC: {output2[0]}\n")
            fo.write(f"SRCC: {output2[1]}\n")
            fo.write(f"MainScore: {output2[2]}\n")