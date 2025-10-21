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

base_anno = "/code/All-In-One/qbw/EasyR1-20250410/cache/data/AGIQA-3k/annos/data_jsonl.jsonl"
img_path2prompt = {}

with open(base_anno) as f:
    for line in f:
        item = json.loads(line.strip())
        image = item['image']
        assert image not in img_path2prompt
        img_path2prompt[image] = item['prompt']
        

def generate_data(jsonl_path, prompt_str):
    with open(jsonl_path) as f:
        for line in f:
            item = json.loads(line.strip())
            image = Image.open(item['messages'][0]['content'][0]['image'], "r").convert("RGB")
            prompt = img_path2prompt[item['messages'][0]['content'][0]['image']]
            yield {
                "images": [image],
                "problem": prompt_str.format(prompt),
                "answer": float(item['messages'][1]['content']),
                "prompt": prompt,
            }


def main():
    query_format = """
<image>What is your overall rating on the quality of this AI-generated picture with a textual prompt: {}?

Think step by step about the perceptual quality of the image, considering clarity, color fidelity, contrast, texture, structural consistency, and realism, etc. Integrate these aspects to form a final quality judgment.

The rating should be a float between 1 and 5, rounded to two decimal places, with 1 representing very poor quality and 5 representing excellent quality. Return the final answer like: <answer> the score </answer>.
""".strip()
    
    trainset = Dataset.from_generator(generate_data, gen_kwargs={"jsonl_path": "/code/All-In-One/qbw/LLaMA-Factory-20250504/data/agiqa3k/annos_idx7/train.jsonl", "prompt_str": query_format})
    testset = Dataset.from_generator(generate_data, gen_kwargs={"jsonl_path": "/code/All-In-One/qbw/LLaMA-Factory-20250504/data/agiqa3k/annos_idx7/test.jsonl", "prompt_str": query_format})
    dataset = DatasetDict({"train": trainset, "test": testset}).cast_column("images", Sequence(ImageData()))
    dataset.push_to_hub("Coobiw/agiqa3k_prompt_1013", private=False, token=True)


if __name__ == "__main__":
    main()
