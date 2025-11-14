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

from datasets import load_dataset

DIRECT_PROMPT = """\
<image>What is your overall rating on the quality of this AI-generated picture? The rating should be a float between 1 and 5, rounded to two decimal places, with 1 representing very poor quality and 5 representing excellent quality. Return the final answer directly.\n\n\
"""

CoT_PROMPT = """\
<image>What is your overall rating on the quality of this AI-generated picture? The rating should be a float between 1 and 5, rounded to two decimal places, with 1 representing very poor quality and 5 representing excellent quality. Return the final answer like: <answer> the score </answer>. A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think><answer> answer here </answer>\n\
"""

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

    def __init__(self, cot: bool = True):
        super().__init__()
        self.cot = cot
        self.qa_dataset = load_dataset("Coobiw/agiqa3k_finale_1013", split="test")

    def __len__(self):
        return len(self.qa_dataset)

    def __getitem__(self,idx):
        item = self.qa_dataset[idx]
        gt = item['answer']
        query_template = CoT_PROMPT if self.cot else DIRECT_PROMPT

        msgs = [
            {
                "role": "user", "content": [
                    {"type": "image", "image": item['images'][0]},
                    {"type": "text", "text": query_template},
                ]
            },
            {
                "role": "assistant",
                "content": "<think>" if self.cot else ""
            }
        ]
        return {
            "messages": msgs,
            "mos_perception": float(gt),
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
    generated_ids = model.generate(**inputs, max_new_tokens=2048, do_sample=False, temperature=0.)

    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_texts = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    return output_texts

def extract_answer_from_response(response: str):
    """
    Extract numeric answer from model response.
    Looks for content between <answer> and </answer> tags.
    
    Args:
        response: Model generated response string
        
    Returns:
        Extracted float value or None if extraction fails
    """
    try:
        answer_start = response.find("<answer>")
        answer_end = response.find("</answer>", answer_start + len("<answer>"))
        
        if answer_end == -1:
            if answer_start == -1:
                # No tags found, try to parse the whole response
                answer_text = response
            else:
                # Only start tag found, take everything after it
                answer_text = response[answer_start + len("<answer>"):]
        else:
            # Both tags found, extract content between them
            answer_text = response[answer_start + len("<answer>"):answer_end]
        
        return float(answer_text.strip())
    except Exception:
        return None

if __name__ == "__main__":
    model_ckpt_dir = "/code/All-In-One/qbw/LLaMA-Factory-20250504/saves/qwen2p5_vl-7b/full/agiqa_ie_merge/checkpoint-468"
    
    model_name = "qwen25vl_agiqa_ie_merge_cot_e3"
    cot = True
    print(model_ckpt_dir)
    print(model_name)
    print(cot)
    
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

    agiqa3k = AGIQA3k(cot=cot)
    output = []
    output_fname = f"./{model_name}_agiqa3k.json"

    eval_bs = 64
    indices = list(range(len(agiqa3k)))[::eval_bs]
    l = len(agiqa3k)
    for start_idx in tqdm(indices):
        if start_idx + eval_bs > l:
            items = [agiqa3k[idx] for idx in range(start_idx, l)]
        else:
            items = [agiqa3k[idx] for idx in range(start_idx, start_idx + eval_bs)]

        batch_messages = [item['messages'] for item in items]
        model_responses = model_gen(model, processor, batch_messages)

        for response_idx, model_response in enumerate(model_responses):
            item = items[response_idx]
            item['model_response'] = model_response
            item.pop("messages")
            output.append(item)

    with open(output_fname, 'w') as fo:
        json.dump(output, fo, ensure_ascii=False, indent=4)
    
    y_label, y_out = [], []
    error_count = 0
    for i, item in enumerate(output):
        model_response = item['model_response']
        try:
            if cot:
                out = extract_answer_from_response(model_response.strip().rstrip("."))
            else:
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
    
    out_score = f"./{model_name}_agiqa3k_score.txt"
    with open(out_score, 'w') as fo:
        fo.write(f"PLCC: {output1[0]}\n")
        fo.write(f"SRCC: {output1[1]}\n")
        fo.write(f"MainScore: {output1[2]}\n")
        fo.write(f"PLCC: {output2[0]}\n")
        fo.write(f"SRCC: {output2[1]}\n")
        fo.write(f"MainScore: {output2[2]}\n")