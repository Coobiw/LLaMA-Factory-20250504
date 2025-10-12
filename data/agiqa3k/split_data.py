import json
import os
from PIL import Image
import numpy as np
from tqdm import tqdm, trange

from pathlib import Path

import torch

import random
random.seed(42)
np.random.seed(42)

class AGIQA3k(torch.utils.data.Dataset):

    def __init__(self, annos, query_format):
        super().__init__()
        self.annos = []
        with open(annos) as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip()
                item = json.loads(line)
                self.annos.append(item)
            self.user_query_format = query_format

    def __len__(self):
        return len(self.annos)

    def __getitem__(self,idx):
        item = self.annos[idx]
        return {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "image": item['image'],
                        },
                        {"type": "text", "text": self.user_query_format},
                    ],
                }
            ],
            "mos_perception": item['mos_perception'],
        }

def postprocess_data(item, min_mos, max_mos):
    mos_perception = item.pop('mos_perception')
    normalized_mos_perception = 1 + (mos_perception - min_mos) * 4 / (max_mos - min_mos)
    item['messages'].append(
        {
            "role": "assistant",
            "content": f"{normalized_mos_perception:.2f}",
        }
    )
    return item

def postprocess_data_format2(item, min_mos, max_mos):
    mos_perception = item.pop('mos_perception')
    normalized_mos_perception = 1 + (mos_perception - min_mos) * 4 / (max_mos - min_mos)
    item['images'] = [item['messages'][0]['content'][0]['image']]
    item['messages'][0] = {
        "role": "user",
        "content": "<image>" + item['messages'][0]['content'][1]['text'],
    }
    item['messages'].append(
        {
            "role": "assistant",
            "content": f"{normalized_mos_perception:.2f}",
        }
    )
    return item

def to_jsonl(datasource, indices, file_name, min_mos, max_mos):
    file_dir = Path(file_name).parent
    if not file_dir.exists():
        file_dir.mkdir(parents=True, exist_ok=True)
    with open(file_name, "w") as fo:
        for idx in tqdm(indices, desc=f"Processing {file_name}"):
            data_line = postprocess_data(datasource[idx], min_mos, max_mos)
            fo.write(json.dumps(data_line, ensure_ascii=False) + "\n")
    print(f"{file_name} has finished!!!")

def to_train_json(datasource, indices, file_name, min_mos, max_mos):
    file_dir = Path(file_name).parent
    if not file_dir.exists():
        file_dir.mkdir(parents=True, exist_ok=True)
    output = []
    for idx in tqdm(indices, desc=f"Processing {file_name}"):
        data_line = postprocess_data_format2(datasource[idx], min_mos, max_mos)
        output.append(data_line)
    with open(file_name, "w") as fo:
        json.dump(output, fo, ensure_ascii=False, indent=4)


def main(use_agiqa3k_split_fn=False, data_idx = 0):
    annos = "cache/data/AGIQA-3k/data_jsonl.jsonl"

    return_dtype, lower_bound, upper_bound  = "float", "1", "5"
    mid_prompt = " rounded to two decimal places," if return_dtype == "float" else ""
    query_format = 'What is your overall rating on the quality of this AI-generated picture?' + f' The rating should be a {return_dtype} between {lower_bound} and {upper_bound},{mid_prompt} with {lower_bound} representing very poor quality and {upper_bound} representing excellent quality. Return the final answer directly.\n\n'

    agiqa_3k = AGIQA3k(annos, query_format)

    min_mos = agiqa_3k[0]['mos_perception']
    max_mos = agiqa_3k[0]['mos_perception']
    for idx in range(1, len(agiqa_3k)):
        mos = agiqa_3k[idx]['mos_perception']

        if mos < min_mos:
            min_mos = mos
        if mos > max_mos:
            max_mos = mos
        

    def agiqa3k_split_fn():
        count = int(0.8 * 300)
    
        new_indices = np.random.permutation(300)
    
        train_indices, test_indices = [], []
    
        for i in range(len(agiqa_3k)):
            image_name = Path(agiqa_3k[i]["messages"][0]['content'][0]['image']).stem
            image_name_split = image_name.split("_")
            idx = int(image_name_split[-1])
    
            if idx in new_indices[:count]:
                train_indices.append(i)
            else:
                test_indices.append(i)
    
        return train_indices, test_indices
    
    def random_split_fn():
        train_count = int(0.8 * len(agiqa_3k))
        test_count = len(agiqa_3k) - train_count
        
        total_indices = list(range(len(agiqa_3k)))
        random.shuffle(total_indices)
        train_indices = total_indices[:train_count]
        test_indices = total_indices[train_count:]
        
        assert len(test_indices) == test_count
        assert len(train_indices) == train_count
        return train_indices, test_indices
    
    if use_agiqa3k_split_fn:
        split_fn = agiqa3k_split_fn
    else:
        split_fn = random_split_fn
    
    train_indices, test_indices = split_fn()

    to_jsonl(agiqa_3k, indices=train_indices, file_name=f'data/agiqa3k/annos_idx{data_idx}/train.jsonl', min_mos=min_mos, max_mos=max_mos)
    to_jsonl(agiqa_3k, indices=test_indices, file_name=f'data/agiqa3k/annos_idx{data_idx}/test.jsonl', min_mos=min_mos, max_mos=max_mos)
    to_train_json(agiqa_3k, indices=train_indices, file_name=f'data/agiqa3k/annos_idx{data_idx}/train.json', min_mos=min_mos, max_mos=max_mos)


if __name__ == "__main__":
    total_data_idx = 10
    for data_idx in trange(total_data_idx):
        if data_idx < 5:
            use_agiqa3k_split_fn = True
        else:
            use_agiqa3k_split_fn = False
        main(use_agiqa3k_split_fn, data_idx=data_idx)
        print(f"data_idx {data_idx} has finished!!!")

