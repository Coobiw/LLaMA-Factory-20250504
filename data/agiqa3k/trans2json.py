import json
import os

from pathlib import Path

import torch

from tqdm import tqdm

class AGIQA3k(torch.utils.data.Dataset):

    def __init__(self, annos, sys_prompt, query_format, use_prompt=False):
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
        self.use_prompt = use_prompt

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
                        {"type": "text", "text": self.user_query_format.format(item['prompt']) + self.sys_prompt if self.use_prompt else self.user_query_format + self.sys_prompt},
                    ],
                }
            ],
            "mos_perception": item['mos_perception'],
            "mos_align": item['mos_align']
        }

def generate_data(data_source_1, data_source_2, indices, min_mos, max_mos, min_align, max_align):
    for idx in indices:
        assert data_source_1[idx]['mos_align'] == data_source_2[idx]['mos_align']
        assert data_source_1[idx]['mos_perception'] == data_source_2[idx]['mos_perception']
        for source_id in range(2):
            if source_id == 0:
                item = data_source_1[idx]
                image = item['messages'][0]['content'][0]['image']
                
                yield {
                    "images": [image],
                    "messages": [
                        {"role": "user", "content": "<image>" + item['messages'][0]['content'][1]['text']},
                        {"role": "assistant", "content": str(1 + (item['mos_perception'] - min_mos) * 4 / (max_mos - min_mos))},
                    ]
                }
            else:
                item = data_source_2[idx]
                image = item['messages'][0]['content'][0]['image']
                
                yield {
                    "images": [image],
                    "messages": [
                        {"role": "user", "content": "<image>" + item['messages'][0]['content'][1]['text']},
                        {"role": "assistant", "content": str(1 + (item['mos_align'] - min_align) * 4 / (max_align - min_align))},
                    ]
                }

def to_jsonl(data_lines, indices, file_name):
    with open(file_name, "w") as fo:
        for idx in indices:
            data_line = data_lines[idx].strip()
            fo.write(data_line + "\n")
    print(f"{file_name} has finished!!!")

def main():
    annos = "/code/All-In-One/qbw/EasyR1-20250410/cache/data/AGIQA-3k/annos/train.jsonl"
    sys_prompt = ''

    return_dtype, lower_bound, upper_bound  = "float", "1", "5"
    mid_prompt = " rounded to two decimal places," if return_dtype == "float" else ""
    query_format = 'What is your overall rating on the quality of this AI-generated picture?' + f' The rating should be a {return_dtype} between {lower_bound} and {upper_bound},{mid_prompt} with {lower_bound} representing very poor quality and {upper_bound} representing excellent quality. Return the final answer directly.\n\n'

    query_format_2 = 'What is your overall rating on the image-text correspondence between this AI-generated picture and the textual prompt: "{}"?' + f' The rating should be a {return_dtype} between {lower_bound} and {upper_bound},{mid_prompt} with {lower_bound} representing very poor quality and {upper_bound} representing excellent quality. Return the final answer directly.\n\n'

    agiqa_3k = AGIQA3k(annos, sys_prompt, query_format)
    agiqa_3k_align = AGIQA3k(annos, sys_prompt, query_format_2, use_prompt=True)
    indices = list(range(len(agiqa_3k)))

    min_mos = None
    max_mos = None
    min_align = None
    max_align = None
    for idx in indices:
        mos = agiqa_3k[idx]['mos_perception']
        align = agiqa_3k[idx]['mos_align']
        if (min_mos is None) and (max_mos is None) and (min_align is None) and (max_align is None):
            min_mos = mos
            max_mos = mos
            min_align = align
            max_align = align
            continue

        if mos < min_mos:
            min_mos = mos
        if mos > max_mos:
            max_mos = mos
        if align < min_align:
            min_align = align
        if align > max_align:
            max_align = align

    output = []
    for item in tqdm(generate_data(agiqa_3k, agiqa_3k_align, indices, min_mos, max_mos, min_align, max_align)):
        output.append(item)

    with open("./agiqa3k_joint.json", 'w') as fo:
        json.dump(output, fo, ensure_ascii=False, indent=4)
    
    
    


if __name__ == "__main__":
    main()
