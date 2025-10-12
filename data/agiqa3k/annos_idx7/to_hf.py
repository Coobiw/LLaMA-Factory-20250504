import json
import os
from datasets import Dataset, DatasetDict, Sequence
from datasets import Image as ImageData
from PIL import Image
import numpy as np

from pathlib import Path

import torch

import random
random.seed(42)
np.random.seed(42)

from huggingface_hub import login

token = os.environ.get("HUGGINGFACE_TOKEN")
login(token=token)

def generate_data(jsonl_path, prompt_str):
    with open(jsonl_path) as f:
        for line in f:
            item = json.loads(line.strip())
            image = Image.open(item['messages'][0]['content'][0]['image'], "r").convert("RGB")
            yield {
                "images": [image],
                "problem": "<image>" + prompt_str,
                "answer": float(item['messages'][1]['content']),
            }


def main():
    return_dtype, lower_bound, upper_bound  = "float", "1", "5"
    mid_prompt = " rounded to two decimal places," if return_dtype == "float" else ""
    query_format = 'What is your overall rating on the quality of this AI-generated picture?' + f' The rating should be a {return_dtype} between {lower_bound} and {upper_bound},{mid_prompt} with {lower_bound} representing very poor quality and {upper_bound} representing excellent quality. Return the final answer like: <answer> the score </answer>.\n\n'
    
    trainset = Dataset.from_generator(generate_data, gen_kwargs={"jsonl_path": "/code/All-In-One/qbw/LLaMA-Factory-20250504/data/agiqa3k/annos_idx7/train.jsonl", "prompt_str": query_format})
    testset = Dataset.from_generator(generate_data, gen_kwargs={"jsonl_path": "/code/All-In-One/qbw/LLaMA-Factory-20250504/data/agiqa3k/annos_idx7/test.jsonl", "prompt_str": query_format})
    dataset = DatasetDict({"train": trainset, "test": testset}).cast_column("images", Sequence(ImageData()))
    dataset.push_to_hub("Coobiw/agiqa3k_finale_1013", private=False, token=True)


if __name__ == "__main__":
    main()
