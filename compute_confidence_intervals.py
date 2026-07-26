# compute_confidence_intervals.py
import os
import json
import argparse
import numpy as np
from pathlib import Path
from collections import defaultdict
import pandas as pd
from tqdm import tqdm


class ConfidenceIntervalCalculator:
    def __init__(self, base_dir: str):
        self.base_dir = Path(base_dir)
        self.results = defaultdict(lambda: defaultdict(list))
        self.datasets = ['medmcqa', 'mmlu', 'truthfulqa']
        self.models = ['gpt-4o', 'deepseek-r1', 'qwen3.5-27b']
        
    def load_sample_stats(self, exp_dir: Path) -> dict:
        analysis_dir = exp_dir / "analysis"
        csv_file = analysis_dir / "debate_analysis_results.csv"
        
        if csv_file.exists():
            df = pd.read_csv(csv_file)
            return df.to_dict('records')
        
        graph_files = sorted(exp_dir.glob("graph_*.json"))
        audit_dir = exp_dir / "audit_reports"
        
        stats = []
        for graph_file in graph_files:
            sample_id = int(graph_file.stem.split('_')[1])
            audit_file = audit_dir / f"audit_{sample_id}_0.json"
            
            if not audit_file.exists():
                continue
            
            try:
                with open(audit_file, 'r', encoding='utf-8') as f:
                    audit_data = json.load(f)
                
                statistics = audit_data.get('statistics', {})
                audit_summary = audit_data.get('audit_summary', {})
                
                nodes_per_round = statistics.get('nodes_per_round', {})
                total_nodes = statistics.get('total_nodes', 0)
                total_edges = statistics.get('total_edges', 0)
                
                edges_by_type = statistics.get('edges_by_type_total', {})
                attack_edges = edges_by_type.get('attacks', 0)
                support_edges = edges_by_type.get('supports', 0)
                
                attack_density = attack_edges / total_edges if total_edges > 0 else 0
                support_density = support_edges / total_edges if total_edges > 0 else 0
                
                max_round = max(nodes_per_round.keys()) if nodes_per_round else 0
                final_round_nodes = nodes_per_round.get(max_round, 0)
                convergence_ratio = final_round_nodes / total_nodes if total_nodes > 0 else 0
                
                stats.append({
                    'sample_id': sample_id,
                    'total_nodes': total_nodes,
                    'total_edges': total_edges,
                    'attack_density': attack_density,
                    'support_density': support_density,
                    'convergence_ratio': convergence_ratio,
                })
                
            except Exception as e:
                continue
        
        return stats
    
    def compute_ci(self, data: list, confidence: float = 0.95):
        n_bootstrap = 1000
        n = len(data)
        
        if n < 2:
            return {'mean': None, 'ci_lower': None, 'ci_upper': None}
        
        bootstrap_means = []
        for _ in range(n_bootstrap):
            sample = np.random.choice(data, size=n, replace=True)
            bootstrap_means.append(np.mean(sample))
        
        mean = np.mean(data)
        ci_lower = np.percentile(bootstrap_means, (1 - confidence) / 2 * 100)
        ci_upper = np.percentile(bootstrap_means, (1 + confidence) / 2 * 100)
        
        return {
            'mean': mean,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'ci_range': ci_upper - ci_lower,
            'std': np.std(data),
            'n': n
        }
    
    def load_all_samples(self):
        print("Loading sample data...")
        
        for model_name in self.models:
            model_dir = self.base_dir / model_name
            if not model_dir.exists():
                continue
            
            for dataset_name in self.datasets:
                dataset_dir = model_dir / dataset_name
                if not dataset_dir.exists():
                    continue
                
                exp_dirs = list(dataset_dir.glob("adv_*"))
                if not exp_dirs:
                    continue
                
                for exp_dir in exp_dirs:
                    stats = self.load_sample_stats(exp_dir)
                    if stats:
                        for s in stats:
                            self.results[model_name][dataset_name].append(s)
                        print(f"  Loaded {len(stats)} samples from {model_name}/{dataset_name}")
        
        print(f"\nTotal loaded samples: {sum(len(v) for d in self.results.values() for v in d.values())}")
    
    def compute_all_ci(self):
        metrics = ['attack_density', 'support_density', 'convergence_ratio', 'total_nodes']
        
        results = {}
        
        for model_name, model_data in self.results.items():
            results[model_name] = {}
            for dataset_name, dataset_data in model_data.items():
                results[model_name][dataset_name] = {}
                
                metric_values = {m: [] for m in metrics}
                for sample in dataset_data:
                    for m in metrics:
                        if m in sample:
                            metric_values[m].append(sample[m])
                
                for m in metrics:
                    results[model_name][dataset_name][m] = self.compute_ci(metric_values[m])
        
        return results
    
    def generate_table(self, results: dict):
        print("\n" + "="*80)
        print("Confidence Intervals for Debate Quality Metrics")
        print("="*80)
        
        metrics_names = {
            'convergence_ratio': 'CR',
            'attack_density': 'AD',
            'support_density': 'SD',
            'total_nodes': 'Nodes'
        }
        
        for model_name, model_data in results.items():
            print(f"\n\\textbf{{{model_name.upper()}}}")
            print("-"*80)
            print(f"{'Dataset':<12} {'Metric':<8} {'Mean':<10} {'95% CI':<20} {'n':<6}")
            print("-"*80)
            
            for dataset_name in self.datasets:
                if dataset_name not in model_data:
                    continue
                for metric, ci_data in model_data[dataset_name].items():
                    if ci_data['mean'] is None:
                        continue
                    metric_short = metrics_names.get(metric, metric)
                    ci_str = f"[{ci_data['ci_lower']:.4f}, {ci_data['ci_upper']:.4f}]"
                    print(f"{dataset_name:<12} {metric_short:<8} {ci_data['mean']:<10.4f} {ci_str:<20} {ci_data['n']:<6}")
            print("-"*80)
    
    def generate_latex_table(self, results: dict):
        print("\n" + "="*80)
        print("LaTeX Table with Confidence Intervals")
        print("="*80)
        
        metrics_names = {
            'convergence_ratio': 'CR',
            'attack_density': 'AD',
            'support_density': 'SD',
            'total_nodes': 'Nodes'
        }
        
        for model_name, model_data in results.items():
            print(f"\n% Table for {model_name}")
            print("\\begin{table}[htbp]")
            print("\\centering")
            print(f"\\caption{{Debate quality metrics with 95\\% confidence intervals for {model_name}.}}")
            print("\\label{{tab:ci_" + model_name.replace('-', '_') + "}}")
            print("\\begin{tabular}{lcccc}")
            print("\\toprule")
            print("Dataset & Metric & Mean & 95\\% CI & n \\\\")
            print("\\midrule")
            
            for dataset_name in self.datasets:
                if dataset_name not in model_data:
                    continue
                for metric, ci_data in model_data[dataset_name].items():
                    if ci_data['mean'] is None:
                        continue
                    metric_short = metrics_names.get(metric, metric)
                    print(f"{dataset_name} & {metric_short} & "
                          f"{ci_data['mean']:.4f} & "
                          f"[{ci_data['ci_lower']:.4f}, {ci_data['ci_upper']:.4f}] & "
                          f"{ci_data['n']} \\\\")
            print("\\bottomrule")
            print("\\end{tabular}")
            print("\\end{table}")
    
    def save_results(self, results: dict, output_dir: Path):
        output_file = output_dir / "confidence_intervals.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\nResults saved to: {output_file}")


def main():
    parser = argparse.ArgumentParser(description='Compute confidence intervals for debate quality metrics')
    parser.add_argument('--base_dir', type=str, default='results/gpt-4o',
                        help='Base directory containing experiment results')
    parser.add_argument('--dataset', type=str, default=None,
                        choices=['mmlu', 'truthfulqa', 'medmcqa'],
                        help='Specific dataset to analyze (default: all)')
    parser.add_argument('--output_dir', type=str, default='results/aggregated_figures',
                        help='Output directory for results')
    
    args = parser.parse_args()
    
    calculator = ConfidenceIntervalCalculator(args.base_dir)
    
    if args.dataset:
        calculator.datasets = [args.dataset]
    
    calculator.load_all_samples()
    
    results = calculator.compute_all_ci()
    
    calculator.generate_table(results)
    
    calculator.generate_latex_table(results)
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    calculator.save_results(results, output_dir)


if __name__ == "__main__":
    main()