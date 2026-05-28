# count_tokens.py
import os
import json
import argparse
from pathlib import Path
from collections import defaultdict
import pandas as pd
from tqdm import tqdm

try:
    import tiktoken
    TIKTOKEN_AVAILABLE = True
except ImportError:
    TIKTOKEN_AVAILABLE = False


def get_tokenizer(model_name="gpt-4o"):
    if TIKTOKEN_AVAILABLE:
        try:
            if "gpt-4" in model_name or "gpt-3.5" in model_name:
                return tiktoken.encoding_for_model(model_name)
            else:
                return tiktoken.get_encoding("cl100k_base")
        except:
            return tiktoken.get_encoding("cl100k_base")
    return None


def count_tokens(text, tokenizer):
    if not text:
        return 0
    if tokenizer:
        try:
            return len(tokenizer.encode(text))
        except:
            return len(text) // 4
    else:
        return len(text) // 4


def load_jsonl_file(filepath: Path):
    samples = []
    
    encodings = ['utf-8', 'gbk', 'gb2312', 'latin-1', 'cp1252', 'iso-8859-1']
    
    for encoding in encodings:
        try:
            with open(filepath, 'r', encoding=encoding) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        sample = json.loads(line)
                        samples.append(sample)
                    except json.JSONDecodeError:
                        continue
            if samples:
                break
        except UnicodeDecodeError:
            continue
    
    if not samples:
        try:
            with open(filepath, 'rb') as f:
                content = f.read()
                decoded_content = content.decode('utf-8', errors='ignore')
                for line in decoded_content.split('\n'):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        sample = json.loads(line)
                        samples.append(sample)
                    except json.JSONDecodeError:
                        continue
        except Exception:
            pass
    
    return samples


def analyze_single_experiment(exp_dir: Path, model_name: str = "gpt-4o"):
    data_files = list(exp_dir.glob("adv_*.jsonl"))
    if not data_files:
        return None
    
    tokenizer = get_tokenizer(model_name)
    
    stats = {
        'total_samples': 0,
        'total_tokens': 0,
        'per_sample_tokens': [],
    }
    
    for data_file in data_files:
        samples = load_jsonl_file(data_file)
        
        if not samples:
            continue
        
        for sample in tqdm(samples, desc="  Samples", leave=False):
            agent_responses = sample.get('agent_responses', [])
            
            if not agent_responses:
                continue
            
            sample_tokens = 0
            
            for agent_conv in agent_responses:
                for msg in agent_conv:
                    content = msg.get('content', '')
                    sample_tokens += count_tokens(content, tokenizer)
            
            stats['total_tokens'] += sample_tokens
            stats['per_sample_tokens'].append(sample_tokens)
            stats['total_samples'] += 1
    
    if stats['total_samples'] == 0:
        return None
    
    results = {
        'experiment_dir': str(exp_dir),
        'total_samples': stats['total_samples'],
        'total_tokens': stats['total_tokens'],
        'avg_tokens_per_sample': stats['total_tokens'] / stats['total_samples'],
    }
    
    output_file = exp_dir / "token_analysis.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    return results


def batch_process(base_dir: str):
    base_path = Path(base_dir)
    
    if not base_path.exists():
        return
    
    exp_dirs = []
    for adv_dir in base_path.rglob("adv_*"):
        if adv_dir.is_dir():
            if len(list(adv_dir.glob("adv_*.jsonl"))) > 0:
                exp_dirs.append(adv_dir)
    
    if not exp_dirs:
        return
    
    all_results = {}
    
    for exp_dir in exp_dirs:
        model_name = "gpt-4o"
        if "qwen" in str(exp_dir):
            model_name = "qwen3.5-27b"
        
        results = analyze_single_experiment(exp_dir, model_name)
        if results:
            all_results[str(exp_dir.relative_to(base_path))] = results
    
    if not all_results:
        return
    
    summary_file = base_path / "token_analysis_summary.json"
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    
    return all_results


def main():
    parser = argparse.ArgumentParser(description='Token consumption evaluation for debate experiments')
    parser.add_argument('--base_dir', type=str, default='results/deepseek-r1',
                        help='Base directory containing experiment results')
    parser.add_argument('--single', type=str, default=None,
                        help='Single experiment directory to analyze')
    parser.add_argument('--model', type=str, default='deepseek-r1',
                        help='Model name for tokenization')
    
    args = parser.parse_args()
    
    if args.single:
        analyze_single_experiment(Path(args.single), args.model)
    else:
        batch_process(args.base_dir)


if __name__ == "__main__":
    main()