# plot_aggregated.py
import os
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple
import pandas as pd
import seaborn as sns

plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")


class AggregatedDebateAnalyzer:
    
    def __init__(self, base_dir: str = "results", models: List[str] = None, datasets: List[str] = None):
        self.base_dir = Path(base_dir)
        self.output_dir = self.base_dir / "aggregated_figures"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.models = models if models else []
        self.datasets = datasets if datasets else []
        self.summaries = {}
        
        self._scan_and_load()
    
    def _scan_and_load(self):
        if not self.models:
            return
        
        if not self.datasets:
            return
        
        for model_name in self.models:
            model_dir = self.base_dir / model_name
            if not model_dir.exists():
                continue
            
            for dataset_name in self.datasets:
                dataset_dir = model_dir / dataset_name
                if not dataset_dir.exists():
                    continue
                
                adv_dirs = list(dataset_dir.glob("adv_*"))
                if not adv_dirs:
                    continue
                
                adv_dir = adv_dirs[0]
                analysis_dir = adv_dir / "analysis"
                report_file = analysis_dir / "statistics_report.json"
                
                if report_file.exists():
                    with open(report_file, 'r', encoding='utf-8') as f:
                        self.summaries[(model_name, dataset_name)] = json.load(f)
    
    def get_metric_value(self, model: str, dataset: str, metric: str) -> float:
        summary = self.summaries.get((model, dataset), {})
        return summary.get('summary_stats', {}).get(metric, {}).get('mean', 0)
    
    def _get_color_list(self, n: int) -> List[str]:
        color_pool = ['#2E86AB', '#A23B72', '#F18F01', '#3D5A80', '#EE6C4D', 
                      '#98C1D9', '#293241', '#E0FBFC', '#C77DFF', '#FF9F1C']
        if n <= len(color_pool):
            return color_pool[:n]
        else:
            cmap = plt.cm.tab20
            return [cmap(i / n) for i in range(n)]
    
    def plot_1_grouped_bar_charts(self):
        if not self.models or not self.datasets:
            return
        
        metrics = [('convergence_ratio', 'Convergence Ratio', 0, 1),
                   ('attack_density', 'Attack Density', 0, 1),
                   ('support_density', 'Support Density', 0, 1),
                   ('total_nodes', 'Number of Nodes', None, None)]
        
        n_models = len(self.models)
        colors = self._get_color_list(n_models)
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes = axes.flatten()
        
        for idx, (metric, mname, ymin, ymax) in enumerate(metrics):
            ax = axes[idx]
            x = np.arange(len(self.datasets))
            width = 0.8 / n_models
            
            for i, model in enumerate(self.models):
                values = []
                for dataset in self.datasets:
                    val = self.get_metric_value(model, dataset, metric)
                    values.append(val)
                
                offset = (i - (n_models - 1) / 2) * width
                bars = ax.bar(x + offset, values, width, 
                            label=model, color=colors[i], alpha=0.8)
                
                for bar, val in zip(bars, values):
                    label_text = f'{val:.3f}' if metric != 'total_nodes' else f'{val:.1f}'
                    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                           label_text, ha='center', va='bottom', fontsize=8,
                           rotation=45 if len(self.models) > 4 else 0)
            
            ax.set_xlabel('Dataset', fontsize=12)
            ax.set_ylabel(mname, fontsize=12)
            ax.set_title(f'{mname} Comparison', fontsize=14, fontweight='bold')
            ax.set_xticks(x)
            ax.set_xticklabels(self.datasets)
            if ymin is not None:
                ax.set_ylim(ymin, ymax + 0.05)
            ax.legend(loc='upper right', fontsize=9)
            ax.grid(True, alpha=0.3, axis='y')
        
        plt.suptitle('Debate Quality Metrics Comparison Across Models and Datasets', fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout()
        plt.savefig(self.output_dir / '01_grouped_bar_charts.pdf', format='pdf', dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_2_attack_vs_convergence_scatter(self):
        if not self.models or not self.datasets:
            return
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        shapes = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*']
        shape_dict = {dataset: shapes[i % len(shapes)] for i, dataset in enumerate(self.datasets)}
        
        n_models = len(self.models)
        colors = self._get_color_list(n_models)
        color_dict = {model: colors[i] for i, model in enumerate(self.models)}
        
        for model in self.models:
            for dataset in self.datasets:
                attack = self.get_metric_value(model, dataset, 'attack_density')
                convergence = self.get_metric_value(model, dataset, 'convergence_ratio')
                
                ax.scatter(convergence, attack, 
                          s=200, c=[color_dict[model]], marker=shape_dict[dataset],
                          edgecolors='black', linewidth=1.5, alpha=0.8)
                
                ax.annotate(f'{model}\n{dataset}', 
                           xy=(convergence, attack),
                           xytext=(5, 5), textcoords='offset points',
                           fontsize=8, alpha=0.7)
        
        ax.set_xlabel('Convergence Ratio (Final Round Node Proportion)', fontsize=12)
        ax.set_ylabel('Attack Density (Attack Edge Proportion)', fontsize=12)
        ax.set_title('Attack Density vs Convergence Ratio: All Model-Dataset Combinations', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        legend_elements = []
        for model in self.models:
            legend_elements.append(plt.Line2D([0], [0], marker='o', color='w',
                                             markerfacecolor=color_dict[model],
                                             markersize=10, label=model))
        for dataset in self.datasets:
            legend_elements.append(plt.Line2D([0], [0], marker=shape_dict[dataset], color='w',
                                             markerfacecolor='gray', markersize=10, label=dataset))
        ax.legend(handles=legend_elements, loc='upper right', ncol=2)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / '02_attack_vs_convergence_scatter.pdf', format='pdf', dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_3_radar_charts(self):
        if not self.models or not self.datasets:
            return
        
        metrics = ['convergence_ratio', 'attack_density', 'support_density', 'total_nodes']
        metric_names = ['Convergence', 'Attack', 'Support', 'Nodes']
        
        max_vals = {}
        for metric in metrics:
            max_vals[metric] = 0
            for model in self.models:
                for dataset in self.datasets:
                    val = self.get_metric_value(model, dataset, metric)
                    max_vals[metric] = max(max_vals[metric], val)
        
        n_datasets = len(self.datasets)
        fig, axes = plt.subplots(1, n_datasets, figsize=(5 * n_datasets, 5), 
                                 subplot_kw=dict(projection='polar'))
        if n_datasets == 1:
            axes = [axes]
        
        n_models = len(self.models)
        colors = self._get_color_list(n_models)
        color_dict = {model: colors[i] for i, model in enumerate(self.models)}
        
        angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
        angles += angles[:1]
        
        for idx, dataset in enumerate(self.datasets):
            ax = axes[idx]
            
            for model in self.models:
                values = [self.get_metric_value(model, dataset, metric) for metric in metrics]
                norm_values = [values[i] / max_vals[metrics[i]] if max_vals[metrics[i]] > 0 else 0 
                              for i in range(len(values))]
                norm_values += norm_values[:1]
                
                ax.plot(angles, norm_values, 'o-', linewidth=2, label=model, color=color_dict[model])
                ax.fill(angles, norm_values, alpha=0.1, color=color_dict[model])
            
            ax.set_xticks(angles[:-1])
            ax.set_xticklabels(metric_names, fontsize=9)
            ax.set_ylim(0, 1)
            ax.set_title(f'{dataset}', fontsize=13, fontweight='bold', pad=20)
            ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0), fontsize=8)
            ax.grid(True)
        
        plt.suptitle('Model Performance Radar Chart (Normalized)', fontsize=15, fontweight='bold', y=1.05)
        plt.tight_layout()
        plt.savefig(self.output_dir / '03_radar_charts.pdf', format='pdf', dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_4_correlation_heatmap(self):
        if not self.models or not self.datasets:
            return
        
        all_data = []
        for model in self.models:
            for dataset in self.datasets:
                point = {
                    'model': model,
                    'dataset': dataset,
                    'Convergence Ratio': self.get_metric_value(model, dataset, 'convergence_ratio'),
                    'Attack Density': self.get_metric_value(model, dataset, 'attack_density'),
                    'Support Density': self.get_metric_value(model, dataset, 'support_density'),
                    'Number of Nodes': self.get_metric_value(model, dataset, 'total_nodes')
                }
                all_data.append(point)
        
        if len(all_data) < 2:
            return
        
        df_all = pd.DataFrame(all_data)
        corr_matrix = df_all[['Convergence Ratio', 'Attack Density', 'Support Density', 'Number of Nodes']].corr()
        
        fig, ax = plt.subplots(figsize=(8, 6))
        
        sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='RdBu_r', 
                   vmin=-1, vmax=1, center=0, ax=ax,
                   square=True, linewidths=0.5)
        ax.set_title('Correlation Heatmap of Debate Quality Metrics', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / '04_correlation_heatmap.pdf', format='pdf', dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_5_agent_contribution_aggregated(self):
        if not self.models or not self.datasets:
            return
        
        n_datasets = len(self.datasets)
        fig, axes = plt.subplots(1, n_datasets, figsize=(5 * n_datasets, 5))
        if n_datasets == 1:
            axes = [axes]
        
        n_models = len(self.models)
        colors = self._get_color_list(n_models)
        color_dict = {model: colors[i] for i, model in enumerate(self.models)}
        
        agent_labels = [f'Agent {i}' for i in range(3)]
        
        for idx, dataset in enumerate(self.datasets):
            ax = axes[idx]
            
            x = np.arange(len(agent_labels))
            width = 0.8 / n_models
            
            for i, model in enumerate(self.models):
                summary = self.summaries.get((model, dataset), {})
                agent_contrib = summary.get('agent_contributions', {})
                
                agent_means = []
                for agent in range(3):
                    agent_key = f'agent_{agent}'
                    if agent_key in agent_contrib:
                        agent_means.append(agent_contrib[agent_key].get('mean', 0))
                    else:
                        agent_means.append(0)
                
                offset = (i - (n_models - 1) / 2) * width
                bars = ax.bar(x + offset, agent_means, width, 
                            label=model, color=color_dict[model], alpha=0.8)
                
                if n_models <= 4:
                    for bar, val in zip(bars, agent_means):
                        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                               f'{val:.1f}', ha='center', va='bottom', fontsize=7)
            
            ax.set_xlabel('Agent', fontsize=11)
            ax.set_ylabel('Average Number of Arguments', fontsize=11)
            ax.set_title(f'{dataset}', fontsize=13, fontweight='bold')
            ax.set_xticks(x)
            ax.set_xticklabels(agent_labels)
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3, axis='y')
        
        plt.suptitle('Agent Contribution Comparison', fontsize=15, fontweight='bold', y=1.02)
        plt.tight_layout()
        plt.savefig(self.output_dir / '05_agent_contribution_aggregated.pdf', format='pdf', dpi=300, bbox_inches='tight')
        plt.close()
    
    def generate_latex_tables(self):
        if not self.models or not self.datasets:
            return
        
        latex_main = r"""\begin{table}[t]
\centering
\caption{Debate Quality Metrics Summary (Mean)}
\label{tab:quality_summary}
\begin{tabular}{llcccc}
\toprule
\multirow{2}{*}{Dataset} & \multirow{2}{*}{Model} & \multicolumn{4}{c}{Metrics} \\
\cmidrule(lr){3-6}
 & & Convergence & Attack & Support & \#Nodes \\
\midrule
"""
        
        for dataset in self.datasets:
            for i, model in enumerate(self.models):
                conv = self.get_metric_value(model, dataset, 'convergence_ratio')
                attack = self.get_metric_value(model, dataset, 'attack_density')
                support = self.get_metric_value(model, dataset, 'support_density')
                nodes = self.get_metric_value(model, dataset, 'total_nodes')
                
                if i == 0:
                    latex_main += f"\\multirow{{{len(self.models)}}}{{*}}{{{dataset}}} & {model} & {conv:.3f} & {attack:.3f} & {support:.3f} & {nodes:.1f} \\\\\n"
                else:
                    latex_main += f" & {model} & {conv:.3f} & {attack:.3f} & {support:.3f} & {nodes:.1f} \\\\\n"
            
            if dataset != self.datasets[-1]:
                latex_main += "\\addlinespace\n"
        
        latex_main += r"""
\bottomrule
\end{tabular}
\end{table}
"""
        
        with open(self.output_dir / '06_latex_table_main.tex', 'w', encoding='utf-8') as f:
            f.write(latex_main)
    
    def run_all(self):
        if not self.models:
            return
        
        if not self.datasets:
            return
        
        self.plot_1_grouped_bar_charts()
        self.plot_2_attack_vs_convergence_scatter()
        self.plot_3_radar_charts()
        self.plot_4_correlation_heatmap()
        self.plot_5_agent_contribution_aggregated()
        self.generate_latex_tables()


def main():
    parser = argparse.ArgumentParser(description='Aggregated Debate Quality Analysis Tool')
    parser.add_argument('--base_dir', type=str, default='results',
                       help='Root directory of experiment results')
    parser.add_argument('--models', type=str, nargs='+', 
                       default=['gpt-4o', 'deepseek-r1', 'qwen3.5-27b'],
                       help='Model folder names to read')
    parser.add_argument('--datasets', type=str, nargs='+',
                       default=['medmcqa', 'mmlu', 'truthfulqa'],
                       help='Dataset names to visualize')
    args = parser.parse_args()
    
    analyzer = AggregatedDebateAnalyzer(base_dir=args.base_dir, models=args.models, datasets=args.datasets)
    analyzer.run_all()


if __name__ == "__main__":
    main()