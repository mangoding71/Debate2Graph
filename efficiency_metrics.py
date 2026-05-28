# efficiency_metrics.py
import os
import json
import argparse
from pathlib import Path
import pandas as pd
from tqdm import tqdm


def get_file_size_kb(filepath: Path) -> float:
    if filepath.exists():
        return filepath.stat().st_size / 1024
    return 0.0


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
        except Exception:
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


def analyze_single_experiment(exp_dir: Path):
    data_files = list(exp_dir.glob("adv_*.jsonl"))
    if not data_files:
        return None
    
    per_sample_api_calls = []
    per_sample_storage_kb = []
    per_sample_latency = []
    
    audit_dir = exp_dir / "audit_reports"
    
    for data_file in data_files:
        samples = load_jsonl_file(data_file)
        
        if not samples:
            continue
        
        for sample in tqdm(samples, desc="  Samples", leave=False):
            agent_responses = sample.get('agent_responses', [])
            metadata = sample.get('metadata', {})
            
            if not agent_responses:
                continue
            
            sample_id = sample.get('id', 0)
            
            n_assistant_msgs = sum(1 for agent in agent_responses for msg in agent if msg.get('role') == 'assistant')
            per_sample_api_calls.append(n_assistant_msgs)
            
            latency = metadata.get('latency', None)
            if latency is None:
                start_time = metadata.get('start_time')
                end_time = metadata.get('end_time')
                if start_time and end_time:
                    latency = end_time - start_time
                else:
                    latency = n_assistant_msgs * 0.5
            per_sample_latency.append(latency)
            
            storage_kb = 0.0
            audit_file = audit_dir / f"audit_{sample_id}_0.json"
            storage_kb += get_file_size_kb(audit_file)
            graph_file = exp_dir / f"graph_{sample_id}_0.json"
            storage_kb += get_file_size_kb(graph_file)
            per_sample_storage_kb.append(storage_kb)
    
    if not per_sample_api_calls:
        return None
    
    results = {
        'experiment_dir': str(exp_dir),
        'total_samples': len(per_sample_api_calls),
        'avg_api_calls_per_sample': sum(per_sample_api_calls) / len(per_sample_api_calls),
        'avg_storage_kb_per_sample': sum(per_sample_storage_kb) / len(per_sample_storage_kb),
        'avg_latency_per_sample': sum(per_sample_latency) / len(per_sample_latency),
    }
    
    output_file = exp_dir / "efficiency_metrics.json"
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
        results = analyze_single_experiment(exp_dir)
        if results:
            all_results[str(exp_dir.relative_to(base_path))] = results
    
    if not all_results:
        return
    
    summary_file = base_path / "efficiency_metrics_summary.json"
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    
    return all_results


def main():
    parser = argparse.ArgumentParser(description='Efficiency metrics evaluation for debate experiments')
    parser.add_argument('--base_dir', type=str, default='results/qwen3.5-27b',
                        help='Base directory containing experiment results')
    parser.add_argument('--single', type=str, default=None,
                        help='Single experiment directory to analyze')
    
    args = parser.parse_args()
    
    if args.single:
        analyze_single_experiment(Path(args.single))
    else:
        batch_process(args.base_dir)


if __name__ == "__main__":
    main()