# plot.py
import os
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple, Any
import pandas as pd
from glob import glob

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif']
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 13
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
plt.rcParams['axes.unicode_minus'] = False


class DebateAnalyzer:
    
    def __init__(self, input_dir: str):
        self.input_dir = Path(input_dir)
        self.graph_files = sorted(self.input_dir.glob("graph_*.json"))
        audit_dir = self.input_dir / "audit_reports"
        if audit_dir.exists():
            self.audit_files = sorted(audit_dir.glob("audit_*.json"))
        else:
            self.audit_files = []
        self.data_files = list(self.input_dir.glob("adv_*.jsonl"))
        self.output_dir = self.input_dir / "analysis"
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        if len(self.graph_files) != len(self.audit_files):
            pass
    
    def load_graph(self, filepath: Path) -> Dict:
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def load_audit(self, filepath: Path) -> Dict:
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def extract_sample_id(self, filename: Path) -> int:
        parts = filename.stem.split('_')
        if len(parts) >= 2:
            return int(parts[1])
        return 0
    
    def analyze_single_sample(self, graph_data: Dict, audit_data: Dict, sample_id: int) -> Dict:
        nodes = graph_data.get('nodes', {})
        edges = graph_data.get('edges', [])
        
        total_nodes = len(nodes)
        total_edges = len(edges)
        
        nodes_by_round = defaultdict(int)
        nodes_by_agent = defaultdict(int)
        nodes_by_type = defaultdict(int)
        
        for node_id, node in nodes.items():
            round_num = node.get('round', 0)
            agent_id = node.get('agent_id', 0)
            node_type = node.get('node_type', 'unknown')
            
            nodes_by_round[round_num] += 1
            nodes_by_agent[agent_id] += 1
            nodes_by_type[node_type] += 1
        
        edges_by_round = defaultdict(int)
        edges_by_type = defaultdict(int)
        
        for edge in edges:
            source_id = edge.get('source', '')
            if source_id in nodes:
                source_round = nodes[source_id].get('round', 0)
                edges_by_round[source_round] += 1
            
            relation = edge.get('relation', 'unknown')
            edges_by_type[relation] += 1
        
        audit_summary = audit_data.get('audit_summary', {})
        statistics = audit_data.get('statistics', {})
        
        top_disputes = audit_summary.get('top_dispute_nodes', [])
        top_consensus = audit_summary.get('top_consensus_nodes', [])
        conclusions = audit_summary.get('conclusions', [])
        
        max_round = max(nodes_by_round.keys()) if nodes_by_round else 0
        final_round_nodes = nodes_by_round.get(max_round, 0)
        convergence_ratio = final_round_nodes / total_nodes if total_nodes > 0 else 0
        
        rounds = sorted(nodes_by_round.keys())
        if len(rounds) > 1:
            growth_rates = []
            for i in range(1, len(rounds)):
                prev = nodes_by_round[rounds[i-1]]
                curr = nodes_by_round[rounds[i]]
                growth_rates.append((curr - prev) / prev if prev > 0 else 0)
            avg_growth = np.mean(growth_rates) if growth_rates else 0
        else:
            avg_growth = 0
        
        attack_edges = edges_by_type.get('attacks', 0)
        attack_density = attack_edges / total_edges if total_edges > 0 else 0
        
        support_edges = edges_by_type.get('supports', 0)
        support_density = support_edges / total_edges if total_edges > 0 else 0
        
        return {
            'sample_id': sample_id,
            'total_nodes': total_nodes,
            'total_edges': total_edges,
            'nodes_by_round': dict(nodes_by_round),
            'nodes_by_agent': dict(nodes_by_agent),
            'nodes_by_type': dict(nodes_by_type),
            'edges_by_round': dict(edges_by_round),
            'edges_by_type': dict(edges_by_type),
            'top_disputes_count': len(top_disputes),
            'top_consensus_count': len(top_consensus),
            'conclusions_count': len(conclusions),
            'convergence_ratio': convergence_ratio,
            'avg_growth_rate': avg_growth,
            'attack_density': attack_density,
            'support_density': support_density,
            'max_round': max_round
        }
    
    def analyze_all(self) -> pd.DataFrame:
        results = []
        min_files = min(len(self.graph_files), len(self.audit_files))
        
        for i in range(min_files):
            try:
                graph_file = self.graph_files[i]
                audit_file = self.audit_files[i]
                
                graph_id = self.extract_sample_id(graph_file)
                audit_id = self.extract_sample_id(audit_file)
                
                if graph_id != audit_id:
                    continue
                
                graph_data = self.load_graph(graph_file)
                audit_data = self.load_audit(audit_file)
                
                result = self.analyze_single_sample(graph_data, audit_data, graph_id)
                results.append(result)
                    
            except Exception:
                continue
        
        return pd.DataFrame(results)
    
    def generate_statistics_report(self, df: pd.DataFrame) -> Dict:
        report = {
            'total_samples': len(df),
            'summary_stats': {
                'total_nodes': {
                    'mean': float(df['total_nodes'].mean()),
                    'std': float(df['total_nodes'].std()),
                    'min': int(df['total_nodes'].min()),
                    'max': int(df['total_nodes'].max()),
                    'median': float(df['total_nodes'].median()),
                },
                'total_edges': {
                    'mean': float(df['total_edges'].mean()),
                    'std': float(df['total_edges'].std()),
                    'min': int(df['total_edges'].min()),
                    'max': int(df['total_edges'].max()),
                    'median': float(df['total_edges'].median()),
                },
                'convergence_ratio': {
                    'mean': float(df['convergence_ratio'].mean()),
                    'std': float(df['convergence_ratio'].std()),
                    'median': float(df['convergence_ratio'].median()),
                },
                'attack_density': {
                    'mean': float(df['attack_density'].mean()),
                    'std': float(df['attack_density'].std()),
                    'median': float(df['attack_density'].median()),
                },
                'support_density': {
                    'mean': float(df['support_density'].mean()),
                    'std': float(df['support_density'].std()),
                    'median': float(df['support_density'].median()),
                }
            },
            'agent_contributions': {},
            'nodes_by_type_total': {},
            'edges_by_type_total': {}
        }
        
        for agent in range(3):
            contributions = df['nodes_by_agent'].apply(lambda x: x.get(agent, 0))
            if contributions.sum() > 0:
                report['agent_contributions'][f'agent_{agent}'] = {
                    'mean': float(contributions.mean()),
                    'std': float(contributions.std()),
                    'total': int(contributions.sum())
                }
        
        for node_type in ['conclusion', 'evidence', 'assumption', 'claim', 'question']:
            counts = df['nodes_by_type'].apply(lambda x: x.get(node_type, 0))
            if counts.sum() > 0:
                report['nodes_by_type_total'][node_type] = int(counts.sum())
        
        for edge_type in ['attacks', 'supports', 'cites']:
            counts = df['edges_by_type'].apply(lambda x: x.get(edge_type, 0))
            if counts.sum() > 0:
                report['edges_by_type_total'][edge_type] = int(counts.sum())
        
        return report


class DebateVisualizer:
    
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def plot_distributions(self, df: pd.DataFrame):
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.hist(df['total_nodes'], bins=20, edgecolor='black', alpha=0.7, color='steelblue')
        ax.set_xlabel('Number of Nodes')
        ax.set_ylabel('Number of Samples')
        ax.axvline(df['total_nodes'].mean(), color='red', linestyle='--', label=f"Mean: {df['total_nodes'].mean():.1f}")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(self.output_dir / 'nodes_distribution.pdf', dpi=300, bbox_inches='tight')
        plt.close()

        fig, ax = plt.subplots(figsize=(5, 4))
        ax.hist(df['total_edges'], bins=20, edgecolor='black', alpha=0.7, color='steelblue')
        ax.set_xlabel('Number of Edges')
        ax.set_ylabel('Number of Samples')
        ax.axvline(df['total_edges'].mean(), color='red', linestyle='--', label=f"Mean: {df['total_edges'].mean():.1f}")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(self.output_dir / 'edges_distribution.pdf', dpi=300, bbox_inches='tight')
        plt.close()

        fig, ax = plt.subplots(figsize=(5, 4))
        ax.hist(df['convergence_ratio'], bins=20, edgecolor='black', alpha=0.7, color='steelblue')
        ax.set_xlabel('Convergence Ratio')
        ax.set_ylabel('Number of Samples')
        ax.axvline(df['convergence_ratio'].mean(), color='red', linestyle='--', label=f"Mean: {df['convergence_ratio'].mean():.3f}")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(self.output_dir / 'convergence_distribution.pdf', dpi=300, bbox_inches='tight')
        plt.close()

        fig, ax = plt.subplots(figsize=(5, 4))
        ax.hist(df['attack_density'], bins=20, edgecolor='black', alpha=0.7, color='darkred')
        ax.set_xlabel('Attack Density')
        ax.set_ylabel('Number of Samples')
        ax.axvline(df['attack_density'].mean(), color='red', linestyle='--', label=f"Mean: {df['attack_density'].mean():.3f}")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(self.output_dir / 'attack_distribution.pdf', dpi=300, bbox_inches='tight')
        plt.close()

        fig, ax = plt.subplots(figsize=(5, 4))
        ax.scatter(df['total_nodes'], df['total_edges'], alpha=0.6, c='steelblue', s=30)
        ax.set_xlabel('Number of Nodes')
        ax.set_ylabel('Number of Edges')
        z = np.polyfit(df['total_nodes'], df['total_edges'], 1)
        p = np.poly1d(z)
        x_sorted = np.sort(df['total_nodes'])
        ax.plot(x_sorted, p(x_sorted), 'r--', alpha=0.8, label=f'y = {z[0]:.2f}x + {z[1]:.2f}')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(self.output_dir / 'node_edge_scatter.pdf', dpi=300, bbox_inches='tight')
        plt.close()

        fig, ax = plt.subplots(figsize=(5, 4))
        ax.scatter(df['convergence_ratio'], df['attack_density'], alpha=0.6, c='darkred', s=30)
        ax.set_xlabel('Convergence Ratio')
        ax.set_ylabel('Attack Density')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(self.output_dir / 'convergence_attack_scatter.pdf', dpi=300, bbox_inches='tight')
        plt.close()

        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes[0, 0].hist(df['total_nodes'], bins=20, edgecolor='black', alpha=0.7, color='steelblue')
        axes[0, 0].set_xlabel('Number of Nodes')
        axes[0, 0].set_ylabel('Number of Samples')
        axes[0, 0].axvline(df['total_nodes'].mean(), color='red', linestyle='--', label=f"Mean: {df['total_nodes'].mean():.1f}")
        axes[0, 0].legend(fontsize=10)
        axes[0, 0].grid(True, alpha=0.3)

        axes[0, 1].hist(df['total_edges'], bins=20, edgecolor='black', alpha=0.7, color='steelblue')
        axes[0, 1].set_xlabel('Number of Edges')
        axes[0, 1].set_ylabel('Number of Samples')
        axes[0, 1].axvline(df['total_edges'].mean(), color='red', linestyle='--', label=f"Mean: {df['total_edges'].mean():.1f}")
        axes[0, 1].legend(fontsize=10)
        axes[0, 1].grid(True, alpha=0.3)

        axes[0, 2].hist(df['convergence_ratio'], bins=20, edgecolor='black', alpha=0.7, color='steelblue')
        axes[0, 2].set_xlabel('Convergence Ratio')
        axes[0, 2].set_ylabel('Number of Samples')
        axes[0, 2].axvline(df['convergence_ratio'].mean(), color='red', linestyle='--', label=f"Mean: {df['convergence_ratio'].mean():.3f}")
        axes[0, 2].legend(fontsize=10)
        axes[0, 2].grid(True, alpha=0.3)

        axes[1, 0].hist(df['attack_density'], bins=20, edgecolor='black', alpha=0.7, color='darkred')
        axes[1, 0].set_xlabel('Attack Density')
        axes[1, 0].set_ylabel('Number of Samples')
        axes[1, 0].axvline(df['attack_density'].mean(), color='red', linestyle='--', label=f"Mean: {df['attack_density'].mean():.3f}")
        axes[1, 0].legend(fontsize=10)
        axes[1, 0].grid(True, alpha=0.3)

        axes[1, 1].scatter(df['total_nodes'], df['total_edges'], alpha=0.6, c='steelblue', s=30)
        axes[1, 1].set_xlabel('Number of Nodes')
        axes[1, 1].set_ylabel('Number of Edges')
        axes[1, 1].plot(x_sorted, p(x_sorted), 'r--', alpha=0.8, label=f'y = {z[0]:.2f}x + {z[1]:.2f}')
        axes[1, 1].legend(fontsize=10)
        axes[1, 1].grid(True, alpha=0.3)

        axes[1, 2].scatter(df['convergence_ratio'], df['attack_density'], alpha=0.6, c='darkred', s=30)
        axes[1, 2].set_xlabel('Convergence Ratio')
        axes[1, 2].set_ylabel('Attack Density')
        axes[1, 2].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.output_dir / 'distribution_plots.pdf', dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_round_progression(self, df: pd.DataFrame):
        round_data = defaultdict(list)
        for _, row in df.iterrows():
            nodes_by_round = row['nodes_by_round']
            for round_num, node_count in nodes_by_round.items():
                round_data[round_num].append(node_count)
        
        rounds = sorted(round_data.keys())
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        means = [np.mean(round_data[r]) for r in rounds]
        stds = [np.std(round_data[r]) for r in rounds]
        
        axes[0].errorbar(rounds, means, yerr=stds, marker='o', capsize=5, 
                        capthick=2, markersize=8, linewidth=2, color='steelblue')
        axes[0].set_xlabel('Debate Round')
        axes[0].set_ylabel('Average Number of Nodes')
        axes[0].grid(True, alpha=0.3)
        
        z = np.polyfit(rounds, means, 1)
        p = np.poly1d(z)
        axes[0].plot(rounds, p(rounds), 'r--', alpha=0.8, label=f'Trend Line (slope={z[0]:.2f})')
        axes[0].legend()
        
        bp = axes[1].boxplot([round_data[r] for r in rounds], labels=rounds, patch_artist=True)
        for box in bp['boxes']:
            box.set_facecolor('lightblue')
            box.set_alpha(0.7)
        axes[1].set_xlabel('Debate Round')
        axes[1].set_ylabel('Number of Nodes')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'round_progression.pdf', dpi=150, bbox_inches='tight')
        plt.close()
    
    def plot_agent_contributions(self, df: pd.DataFrame):
        agent_data = defaultdict(list)
        for _, row in df.iterrows():
            nodes_by_agent = row['nodes_by_agent']
            for agent_id, count in nodes_by_agent.items():
                agent_data[agent_id].append(count)
        
        agents = sorted(agent_data.keys())
        agent_names = [f'Agent {a}' for a in agents]
        
        color_palette = ['steelblue', 'darkorange', 'forestgreen', 'crimson', 
                        'purple', 'teal', 'goldenrod', 'navy', 'saddlebrown']
        colors = color_palette[:len(agents)]
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        means = [np.mean(agent_data[a]) for a in agents]
        stds = [np.std(agent_data[a]) for a in agents]
        
        bars = axes[0].bar(agent_names, means, alpha=0.7, color=colors)
        axes[0].set_ylim(0, max(means) * 1.5)
        axes[0].set_xlabel('Agent')
        axes[0].set_ylabel('Average Number of Arguments')
        axes[0].grid(True, alpha=0.3, axis='y')
        
        for bar, mean in zip(bars, means):
            axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                        f'{mean:.1f}', ha='center', va='bottom')
        
        parts = axes[1].violinplot([agent_data[a] for a in agents], showmeans=True, showmedians=True)
        for i, pc in enumerate(parts['bodies']):
            pc.set_facecolor(colors[i % len(colors)])
            pc.set_alpha(0.5)
        for partname in ('cbars', 'cmins', 'cmaxes', 'means', 'medians'):
            if partname in parts:
                plt.setp(parts[partname], color='black', linewidth=1.5)
        axes[1].set_xticks(range(1, len(agents) + 1))
        axes[1].set_xticklabels(agent_names)
        axes[1].set_xlabel('Agent')
        axes[1].set_ylabel('Number of Arguments')
        axes[1].grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'agent_contributions.pdf', dpi=150, bbox_inches='tight')
        plt.close()
    
    def plot_nodes_by_type(self, df: pd.DataFrame):
        type_counts = defaultdict(int)
        for _, row in df.iterrows():
            nodes_by_type = row['nodes_by_type']
            for node_type, count in nodes_by_type.items():
                type_counts[node_type] += count
        
        if not type_counts:
            return
        
        fig, ax = plt.subplots(figsize=(8, 6))
        
        types = list(type_counts.keys())
        counts = list(type_counts.values())
        
        type_names = {
            'conclusion': 'Conclusion',
            'evidence': 'Evidence',
            'assumption': 'Assumption',
            'claim': 'Claim',
            'question': 'Question'
        }
        labels = [type_names.get(t, t) for t in types]
        
        bars = ax.bar(labels, counts, alpha=0.7, color='steelblue')
        ax.set_xlabel('Argument Type')
        ax.set_ylabel('Total Count')
        ax.grid(True, alpha=0.3, axis='y')
        
        for bar, count in zip(bars, counts):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5, 
                   str(count), ha='center', va='bottom', fontsize=11)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'nodes_by_type.pdf', dpi=150, bbox_inches='tight')
        plt.close()
    
    def plot_relation_by_type(self, df: pd.DataFrame):
        relation_counts = defaultdict(int)
        for _, row in df.iterrows():
            edges_by_type = row['edges_by_type']
            for relation_type, count in edges_by_type.items():
                relation_counts[relation_type] += count
        
        if not relation_counts:
            return
        
        fig, ax = plt.subplots(figsize=(8, 6))
        
        relations = list(relation_counts.keys())
        counts = list(relation_counts.values())
        
        relation_names = {
            'attacks': 'Attacks',
            'supports': 'Supports',
            'cites': 'Cites'
        }
        labels = [relation_names.get(r, r) for r in relations]
        colors = ['darkred', 'forestgreen', 'steelblue'][:len(relations)]
        
        bars = ax.bar(labels, counts, alpha=0.7, color=colors)
        ax.set_xlabel('Relation Type')
        ax.set_ylabel('Total Count')
        ax.grid(True, alpha=0.3, axis='y')
        
        for bar, count in zip(bars, counts):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 3, 
                   str(count), ha='center', va='bottom', fontsize=11)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'relations_by_type.pdf', dpi=150, bbox_inches='tight')
        plt.close()
    
    def plot_heatmap(self, df: pd.DataFrame):
        numeric_cols = ['total_nodes', 'total_edges', 'convergence_ratio', 
                       'attack_density', 'support_density', 'top_disputes_count']
        available_cols = [col for col in numeric_cols if col in df.columns]
        
        if len(available_cols) < 2:
            return
        
        corr_matrix = df[available_cols].corr()
        
        col_names = {
            'total_nodes': 'Nodes',
            'total_edges': 'Edges',
            'convergence_ratio': 'Convergence',
            'attack_density': 'Attack Density',
            'support_density': 'Support Density',
            'top_disputes_count': 'Dispute Count'
        }
        
        fig, ax = plt.subplots(figsize=(8, 6))
        
        im = ax.imshow(corr_matrix, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
        
        labels = [col_names.get(col, col) for col in available_cols]
        ax.set_xticks(range(len(labels)))
        ax.set_yticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha='right')
        ax.set_yticklabels(labels)
        
        for i in range(len(available_cols)):
            for j in range(len(available_cols)):
                text = ax.text(j, i, f'{corr_matrix.iloc[i, j]:.2f}',
                             ha="center", va="center", 
                             color="black" if abs(corr_matrix.iloc[i, j]) < 0.5 else "white",
                             fontsize=10)
        
        plt.colorbar(im, ax=ax, label='Correlation Coefficient')
        plt.tight_layout()
        plt.savefig(self.output_dir / 'correlation_heatmap.pdf', dpi=150, bbox_inches='tight')
        plt.close()
    
    def create_case_study(self, df: pd.DataFrame, analyzer: DebateAnalyzer, sample_id: int = None):
        if sample_id is None:
            median_idx = df['total_nodes'].sub(df['total_nodes'].median()).abs().idxmin()
            sample_id = df.iloc[median_idx]['sample_id']
        
        if sample_id not in df['sample_id'].values:
            return None
        
        result = df[df['sample_id'] == sample_id].iloc[0]
        
        graph_file = analyzer.input_dir / f"graph_{sample_id}_0.json"
        if not graph_file.exists():
            return None
        
        graph_data = analyzer.load_graph(graph_file)
        nodes = graph_data.get('nodes', {})
        edges = graph_data.get('edges', [])
        
        key_claims = []
        for node_id, node in nodes.items():
            if node.get('round') == result['max_round'] and len(node.get('content', '')) > 30:
                key_claims.append({
                    'agent': node.get('agent_id'),
                    'content': node.get('content', '')[:200],
                    'type': node.get('node_type')
                })
        
        attack_counts = defaultdict(int)
        for edge in edges:
            if edge.get('relation') == 'attacks':
                attack_counts[edge.get('target')] += 1
        
        top_attacked = sorted(attack_counts.items(), key=lambda x: x[1], reverse=True)[:3]
        dispute_nodes = []
        for node_id, count in top_attacked:
            if node_id in nodes:
                dispute_nodes.append({
                    'content': nodes[node_id].get('content', '')[:150],
                    'attack_count': count,
                    'agent': nodes[node_id].get('agent_id'),
                    'round': nodes[node_id].get('round')
                })
        
        case_report = f"""
{'='*60}
Case Study: Sample {sample_id}
{'='*60}

[Sample Overview]
- Total Nodes: {result['total_nodes']}
- Total Edges: {result['total_edges']}
- Number of Rounds: {result['max_round'] + 1}
- Convergence Ratio: {result['convergence_ratio']:.3f}
- Attack Density: {result['attack_density']:.3f}
- Support Density: {result['support_density']:.3f}

[Argument Distribution]
- By Round: {dict(result['nodes_by_round'])}
- By Agent: 
  - Agent 0: {result['nodes_by_agent'].get(0, 0)} arguments
  - Agent 1: {result['nodes_by_agent'].get(1, 0)} arguments
  - Agent 2: {result['nodes_by_agent'].get(2, 0)} arguments

[Key Dispute Points]
"""
        for i, dispute in enumerate(dispute_nodes, 1):
            case_report += f"""
{i}. [Round {dispute['round']}, Agent {dispute['agent']}] 
   "{dispute['content']}..."
   - Attack Count: {dispute['attack_count']}
"""

        case_report += f"""
[Key Arguments in Final Round] (Round {result['max_round']})
"""
        for i, claim in enumerate(key_claims[:5], 1):
            case_report += f"""
{i}. [Agent {claim['agent']}] {claim['type']}: 
   "{claim['content']}..."
"""

        case_report += f"""
[Debate Graph Statistics]
- Node Type Distribution: {dict(result['nodes_by_type'])}
- Edge Type Distribution: {dict(result['edges_by_type'])}
- Number of Dispute Points: {result['top_disputes_count']}
- Number of Conclusions: {result['conclusions_count']}

{'='*60}
"""
        
        output_file = self.output_dir / f'case_study_sample_{sample_id}.txt'
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(case_report)
        
        return case_report


def process_single_experiment(input_dir: str, sample_id: int = None):
    analyzer = DebateAnalyzer(input_dir)
    
    if len(analyzer.graph_files) == 0:
        return None
    
    df = analyzer.analyze_all()
    
    if len(df) == 0:
        return None
    
    report = analyzer.generate_statistics_report(df)
    
    report_file = analyzer.output_dir / 'statistics_report.json'
    with open(report_file, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    visualizer = DebateVisualizer(analyzer.output_dir)
    visualizer.plot_distributions(df)
    visualizer.plot_round_progression(df)
    visualizer.plot_agent_contributions(df)
    visualizer.plot_nodes_by_type(df)
    visualizer.plot_relation_by_type(df)
    visualizer.plot_heatmap(df)
    
    visualizer.create_case_study(df, analyzer, sample_id)
    
    csv_file = analyzer.output_dir / 'debate_analysis_results.csv'
    csv_data = []
    for _, row in df.iterrows():
        csv_data.append({
            'sample_id': row['sample_id'],
            'total_nodes': row['total_nodes'],
            'total_edges': row['total_edges'],
            'convergence_ratio': row['convergence_ratio'],
            'attack_density': row['attack_density'],
            'support_density': row['support_density'],
            'avg_growth_rate': row['avg_growth_rate'],
            'top_disputes_count': row['top_disputes_count'],
            'top_consensus_count': row['top_consensus_count'],
            'conclusions_count': row['conclusions_count'],
            'max_round': row['max_round']
        })
    pd.DataFrame(csv_data).to_csv(csv_file, index=False, encoding='utf-8-sig')
    
    return report


def batch_process(base_dir: str, sample_id: int = None):
    base_path = Path(base_dir)
    
    adv_dirs = []
    for adv_dir in base_path.rglob("adv_*"):
        if adv_dir.is_dir():
            if len(list(adv_dir.glob("graph_*.json"))) > 0:
                adv_dirs.append(adv_dir)
    
    if not adv_dirs:
        return
    
    all_reports = {}
    
    for adv_dir in adv_dirs:
        report = process_single_experiment(str(adv_dir), sample_id)
        if report:
            all_reports[str(adv_dir.relative_to(base_path))] = report
    
    summary_file = base_path / "all_experiments_summary.json"
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(all_reports, f, indent=2, ensure_ascii=False)
    
    return all_reports


def main():
    parser = argparse.ArgumentParser(description='Debate Quality Analysis Tool - Batch Processing Mode')
    parser.add_argument('--base_dir', type=str, default='results/qwen3.5-27b')
    parser.add_argument('--single', type=str, default=None,
                       help='Single experiment directory path')
    parser.add_argument('--sample_id', type=int, default=None,
                       help='Sample ID for case study')
    
    args = parser.parse_args()
    
    if args.single:
        process_single_experiment(args.single, args.sample_id)
    else:
        batch_process(args.base_dir, args.sample_id)


if __name__ == "__main__":
    main()