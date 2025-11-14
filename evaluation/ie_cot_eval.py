import os
import json
import base64
import asyncio
from pathlib import Path
from tqdm import tqdm, trange
from tqdm.asyncio import tqdm as atqdm
import aiohttp

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
    """Set random seed for reproducibility."""
    np.random.seed(seed)
    random.seed(seed)

set_seed(42)

def encode_image_to_base64(image_path: str) -> str:
    """
    Encode image file to base64 string.
    
    Args:
        image_path: Path to the image file
        
    Returns:
        Base64 encoded string of the image
    """
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


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
        image_path_2 = item['images'][1]
        
        # Construct messages for model inference
        msgs = [
            {
                "role": "user", 
                "content": [
                    {"type": "image", "image": image_path},
                    {"type": "image", "image": image_path_2},
                    {"type": "text", "text": user_content}
                ]
            },
            {
                "role": "assistant",
                "content": "<think>"
            }
        ]
        
        return {
            "messages": msgs,
            "mos_perception": gt,
        }

async def call_vllm_api_async(
    session: aiohttp.ClientSession,
    semaphore: asyncio.Semaphore,
    idx: int,
    messages: list,
    api_url: str,
    model_name: str,
    temperature: float,
    max_tokens: int,
    do_sample: bool
) -> tuple:
    """
    Async call to vLLM API for a single sample.
    
    Args:
        session: aiohttp ClientSession
        semaphore: asyncio Semaphore for concurrency control
        idx: Index of the sample (for maintaining order)
        messages: Message list for single inference
        api_url: Base URL of the vLLM service
        model_name: Model name
        temperature: Sampling temperature
        max_tokens: Maximum number of tokens to generate
        do_sample: Whether to use sampling
        
    Returns:
        Tuple of (index, generated_text)
    """
    # Convert messages format for API
    api_messages = []
    for msg in messages:
        if msg['role'] == 'user':
            content_list = []
            for item in msg['content']:
                if item['type'] == 'image':
                    # Encode image to base64
                    image_path = item['image']
                    base64_image = encode_image_to_base64(image_path)
                    content_list.append({
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{base64_image}"
                        }
                    })
                elif item['type'] == 'text':
                    content_list.append({
                        "type": "text",
                        "text": item['text']
                    })
            api_messages.append({
                "role": "user",
                "content": content_list
            })
        else:
            # Assistant message
            api_messages.append({
                "role": msg['role'],
                "content": msg['content']
            })
    
    # Prepare API request
    payload = {
        "model": model_name,
        "messages": api_messages,
        "temperature": temperature if do_sample else 0.0,
        "max_tokens": max_tokens,
        "top_p": 0.9 if do_sample else 1.0,
    }
    
    # Call API with concurrency control
    async with semaphore:
        try:
            timeout = aiohttp.ClientTimeout(total=300)  # 5 minutes timeout
            async with session.post(
                f"{api_url}/v1/chat/completions",
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=timeout
            ) as response:
                response.raise_for_status()
                result = await response.json()
                generated_text = result['choices'][0]['message']['content']
                return idx, generated_text
        except Exception as e:
            print(f"API call {idx} failed: {e}")
            return idx, ""  # Return empty string on error

async def run_evaluation_async(
    api_url: str,
    model_name: str,
    json_file: str,
    temperature: float,
    max_tokens: int,
    do_sample: bool,
    max_concurrent: int = 50
):
    """
    Run async evaluation with concurrent API calls.
    
    Args:
        api_url: vLLM API URL
        model_name: Model name
        json_file: Path to evaluation JSON file
        temperature: Sampling temperature
        max_tokens: Max tokens to generate
        do_sample: Whether to use sampling
        max_concurrent: Maximum number of concurrent requests
    """
    # Load dataset
    ie_4k = IE4k(json_file=json_file)
    
    # Create semaphore for concurrency control
    semaphore = asyncio.Semaphore(max_concurrent)
    
    # Create aiohttp session
    async with aiohttp.ClientSession() as session:
        # Create all tasks
        tasks = []
        for idx in range(len(ie_4k)):
            item = ie_4k[idx]
            task = call_vllm_api_async(
                session=session,
                semaphore=semaphore,
                idx=idx,
                messages=item['messages'],
                api_url=api_url,
                model_name=model_name,
                temperature=temperature,
                max_tokens=max_tokens,
                do_sample=do_sample
            )
            tasks.append(task)
        
        # Run all tasks concurrently with progress bar
        print(f"\nRunning {len(tasks)} API calls concurrently (max {max_concurrent} at a time)...")
        results = await atqdm.gather(*tasks, desc="API calls")
    
    # Sort results by index to maintain order
    results.sort(key=lambda x: x[0])
    
    # Build output with original items
    output = []
    for idx, model_response in results:
        item = ie_4k[idx]
        item['model_response'] = model_response
        output.append(item)
    
    return output


if __name__ == "__main__":
    # Configuration
    TEMPERATURE = float(os.environ.get("TEMPERATURE", "1.0"))
    API_URL = os.environ.get("VLLM_API_URL", "http://localhost:8000")
    MODEL_NAME = os.environ.get("MODEL_NAME", "ie-critic-r1")
    MAX_CONCURRENT = int(os.environ.get("MAX_CONCURRENT", "16"))
    
    if TEMPERATURE == 0:
        DOSAMPLE = False
        try_times = 1
    else:
        DOSAMPLE = True
        try_times = 5
    
    print(f"API URL: {API_URL}")
    print(f"Model Name: {MODEL_NAME}")
    print(f"Temperature: {TEMPERATURE}")
    print(f"Do Sample: {DOSAMPLE}")
    print(f"Max Concurrent Requests: {MAX_CONCURRENT}")

    for idx in trange(try_times):
        model_name = f"ie-critic-r1_temp{TEMPERATURE}_split{idx}"
        json_file = "./ie4k_cot_test.json"  # Path to the JSON file
        MAX_TOKENS = 2048
        
        print(f"\n{'='*60}")
        print(f"Run {idx+1}/{try_times}")
        print(f"Model name: {model_name}")
        print(f"JSON file: {json_file}")
        print(f"Max tokens: {MAX_TOKENS}")
        print(f"{'='*60}")
    
        # Run async evaluation
        output = asyncio.run(run_evaluation_async(
            api_url=API_URL,
            model_name=MODEL_NAME,
            json_file=json_file,
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            do_sample=DOSAMPLE,
            max_concurrent=MAX_CONCURRENT
        ))
        
        output_fname = f"./{model_name}_ie4k.json"
    
        with open(output_fname, 'w') as fo:
            json.dump(output, fo, ensure_ascii=False, indent=4)
        
        y_label, y_out = [], []
        error_count = 0
        for i, item in enumerate(output):
            model_response = item['model_response']
            try:
                out = extract_answer_from_response(model_response.strip().rstrip("."))
                
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