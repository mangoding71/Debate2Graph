# audit_visualizer.py
import json
from pathlib import Path
from typing import Dict, List, Any
from argumentation_graph import ArgumentationGraph, NodeType, RelationType


def generate_audit_summary(graph: ArgumentationGraph) -> dict:
    
    if not graph.nodes:
        return {
            "total_nodes": 0,
            "total_edges": 0,
            "nodes_by_type": {},
            "conclusions": [],
            "top_dispute_nodes": [],
            "top_consensus_nodes": [],
            "reasoning_chains": []
        }
    
    conclusions = [n for n in graph.nodes.values() if n.node_type == NodeType.CONCLUSION]
    
    attack_counts = {}
    for edge in graph.edges:
        if edge.relation == RelationType.ATTACKS:
            attack_counts[edge.target_id] = attack_counts.get(edge.target_id, 0) + 1
    
    top_disputes = sorted(attack_counts.items(), key=lambda x: x[1], reverse=True)[:3]
    
    support_counts = {}
    for edge in graph.edges:
        if edge.relation == RelationType.SUPPORTS:
            support_counts[edge.target_id] = support_counts.get(edge.target_id, 0) + 1
    
    top_consensus = sorted(support_counts.items(), key=lambda x: x[1], reverse=True)[:3]
    
    nodes_by_type = {}
    for node_type in NodeType:
        count = sum(1 for n in graph.nodes.values() if n.node_type == node_type)
        if count > 0:
            nodes_by_type[node_type.value] = count
    
    summary = {
        "total_nodes": len(graph.nodes),
        "total_edges": len(graph.edges),
        "nodes_by_type": nodes_by_type,
        "conclusions": [
            {"node_id": n.node_id, "content": n.content[:200], "confidence": n.confidence} 
            for n in conclusions
        ],
        "top_dispute_nodes": [
            {
                "node_id": nid, 
                "content": graph.nodes[nid].content[:200], 
                "attack_count": count,
                "agent_id": graph.nodes[nid].agent_id,
                "round": graph.nodes[nid].round
            }
            for nid, count in top_disputes if nid in graph.nodes
        ],
        "top_consensus_nodes": [
            {
                "node_id": nid, 
                "content": graph.nodes[nid].content[:200], 
                "support_count": count,
                "agent_id": graph.nodes[nid].agent_id,
                "round": graph.nodes[nid].round
            }
            for nid, count in top_consensus if nid in graph.nodes
        ],
        "reasoning_chains": []
    }
    
    for conclusion in conclusions[:3]:
        try:
            chains = graph.get_reasoning_chain(conclusion.node_id)
            summary["reasoning_chains"].append({
                "conclusion": conclusion.content[:200],
                "conclusion_id": conclusion.node_id,
                "chains": [
                    [{"content": node.content[:100], "node_id": node.node_id, "agent_id": node.agent_id} 
                     for node in chain]
                    for chain in chains[:3]
                ]
            })
        except Exception as e:
            summary["reasoning_chains"].append({
                "conclusion": conclusion.content[:200],
                "error": str(e)
            })
    
    return summary


def save_audit_report(graph: ArgumentationGraph, filepath: str, include_full_graph: bool = False):
    
    report = {
        "audit_summary": generate_audit_summary(graph),
        "statistics": {
            "total_nodes": len(graph.nodes),
            "total_edges": len(graph.edges),
            "nodes_per_round": {},
            "edges_per_round": {},
            "agents_activity": {}
        }
    }
    
    nodes_per_round = {}
    for node in graph.nodes.values():
        nodes_per_round[node.round] = nodes_per_round.get(node.round, 0) + 1
    report["statistics"]["nodes_per_round"] = nodes_per_round
    
    edges_per_round = {}
    for edge in graph.edges:
        if edge.source_id in graph.nodes:
            source_round = graph.nodes[edge.source_id].round
            edges_per_round[source_round] = edges_per_round.get(source_round, 0) + 1
    report["statistics"]["edges_per_round"] = edges_per_round
    
    agents_activity = {}
    for node in graph.nodes.values():
        if node.agent_id not in agents_activity:
            agents_activity[node.agent_id] = {
                "total_claims": 0,
                "claims_by_type": {},
                "avg_confidence": 0
            }
        agents_activity[node.agent_id]["total_claims"] += 1
        node_type = node.node_type.value
        agents_activity[node.agent_id]["claims_by_type"][node_type] = \
            agents_activity[node.agent_id]["claims_by_type"].get(node_type, 0) + 1
    
    for agent_id in agents_activity:
        agent_nodes = [n for n in graph.nodes.values() if n.agent_id == agent_id]
        if agent_nodes:
            avg_conf = sum(n.confidence for n in agent_nodes) / len(agent_nodes)
            agents_activity[agent_id]["avg_confidence"] = round(avg_conf, 3)
    
    report["statistics"]["agents_activity"] = agents_activity
    
    if include_full_graph:
        report["full_graph"] = graph.to_dict()
    
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    
    return report


def generate_html_report(graph: ArgumentationGraph, output_path: str):
    
    audit_summary = generate_audit_summary(graph)
    
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <title>Debate Audit Report</title>
        <style>
            body {{
                font-family: Arial, sans-serif;
                margin: 20px;
                background-color: #f5f5f5;
            }}
            .container {{
                max-width: 1200px;
                margin: 0 auto;
                background-color: white;
                padding: 20px;
                border-radius: 8px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }}
            h1, h2, h3 {{
                color: #333;
            }}
            .stat-card {{
                background-color: #e3f2fd;
                padding: 15px;
                margin: 10px 0;
                border-radius: 5px;
                display: inline-block;
                width: 200px;
                margin-right: 15px;
            }}
            .stat-number {{
                font-size: 24px;
                font-weight: bold;
                color: #1976d2;
            }}
            .dispute-item, .consensus-item, .conclusion-item {{
                background-color: #f9f9f9;
                padding: 10px;
                margin: 10px 0;
                border-left: 4px solid;
                border-radius: 3px;
            }}
            .dispute-item {{
                border-left-color: #f44336;
            }}
            .consensus-item {{
                border-left-color: #4caf50;
            }}
            .conclusion-item {{
                border-left-color: #ff9800;
            }}
            .chain {{
                background-color: #fff3e0;
                padding: 10px;
                margin: 10px 0;
                border-radius: 3px;
            }}
            .node {{
                display: inline-block;
                background-color: #e0e0e0;
                padding: 5px 10px;
                margin: 2px;
                border-radius: 3px;
                font-size: 12px;
            }}
            .arrow {{
                display: inline-block;
                margin: 0 5px;
                color: #666;
            }}
            table {{
                width: 100%;
                border-collapse: collapse;
                margin: 10px 0;
            }}
            th, td {{
                border: 1px solid #ddd;
                padding: 8px;
                text-align: left;
            }}
            th {{
                background-color: #4caf50;
                color: white;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Debate Audit Report</h1>
            
            <h2>Statistics Overview</h2>
            <div class="stat-card">
                <div class="stat-number">{audit_summary['total_nodes']}</div>
                <div>Total Arguments</div>
            </div>
            <div class="stat-card">
                <div class="stat-number">{audit_summary['total_edges']}</div>
                <div>Total Relations</div>
            </div>
            
            <h2>Argument Type Distribution</h2>
            <table>
                <tr><th>Type</th><th>Count</th></tr>
    """
    
    for node_type, count in audit_summary.get('nodes_by_type', {}).items():
        html_content += f"<tr><td>{node_type}</td><td>{count}</td></tr>"
    
    html_content += """
            </table>
            
            <h2>Main Conclusions</h2>
    """
    
    for conclusion in audit_summary.get('conclusions', [])[:5]:
        html_content += f"""
            <div class="conclusion-item">
                <strong>{conclusion['content']}</strong><br>
                Confidence: {conclusion.get('confidence', 'N/A')}
            </div>
        """
    
    html_content += """
            <h2>Major Disputes (Most Attacked Arguments)</h2>
    """
    
    for dispute in audit_summary.get('top_dispute_nodes', []):
        html_content += f"""
            <div class="dispute-item">
                <strong>{dispute['content']}</strong><br>
                Attack Count: {dispute['attack_count']} | 
                Agent: {dispute.get('agent_id', 'N/A')} | 
                Round: {dispute.get('round', 'N/A')}
            </div>
        """
    
    html_content += """
            <h2>Major Consensus Points (Most Supported Arguments)</h2>
    """
    
    for consensus in audit_summary.get('top_consensus_nodes', []):
        html_content += f"""
            <div class="consensus-item">
                <strong>{consensus['content']}</strong><br>
                Support Count: {consensus['support_count']} | 
                Agent: {consensus.get('agent_id', 'N/A')} | 
                Round: {consensus.get('round', 'N/A')}
            </div>
        """
    
    html_content += """
            <h2>Reasoning Chain Examples</h2>
    """
    
    for chain_info in audit_summary.get('reasoning_chains', [])[:3]:
        html_content += f"""
            <div class="chain">
                <strong>Conclusion:</strong> {chain_info['conclusion']}<br>
                <strong>Reasoning Path:</strong><br>
        """
        for chain in chain_info.get('chains', []):
            html_content += "<div style='margin-left: 20px;'>"
            for i, node in enumerate(chain):
                html_content += f"<span class='node'>{node['content']}</span>"
                if i < len(chain) - 1:
                    html_content += "<span class='arrow'>→</span>"
            html_content += "</div>"
        html_content += "</div>"
    
    html_content += """
        </div>
    </body>
    </html>
    """
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    return output_path


def print_audit_summary(graph: ArgumentationGraph):
    summary = generate_audit_summary(graph)
    
    print("\n" + "=" * 80)
    print("Debate Audit Summary")
    print("=" * 80)
    print(f"Total Arguments: {summary['total_nodes']}")
    print(f"Total Relations: {summary['total_edges']}")
    print(f"Argument Type Distribution: {summary.get('nodes_by_type', {})}")
    
    if summary.get('conclusions'):
        print(f"\nMain Conclusions ({len(summary['conclusions'])}):")
        for conclusion in summary['conclusions'][:3]:
            print(f"  - {conclusion['content'][:100]}...")
    
    if summary.get('top_dispute_nodes'):
        print(f"\nMajor Disputes ({len(summary['top_dispute_nodes'])}):")
        for dispute in summary['top_dispute_nodes'][:3]:
            print(f"  - {dispute['content'][:100]}... (Attacks: {dispute['attack_count']})")
    
    if summary.get('top_consensus_nodes'):
        print(f"\nMajor Consensus ({len(summary['top_consensus_nodes'])}):")
        for consensus in summary['top_consensus_nodes'][:3]:
            print(f"  - {consensus['content'][:100]}... (Supports: {consensus['support_count']})")
    
    print("=" * 80)