import os
import re
import json
import time
import random
import argparse
from collections import Counter
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from tqdm import tqdm
from openai import OpenAI
from answer_parser import extract_choice_answer
from commons import parse_question_answer, query_model
from dataloader import get_dataset
from prompt import agent_prompt

from argumentation_graph import ArgumentationGraph, NodeType, RelationType
from conflict_centric import (
    compute_graph_inconsistency,
    detect_unresolved_conflicts,
    extract_arguments_with_llm,
    graph_conditioned_refinement,
    integrate_extraction,
)
from audit_visualizer import generate_audit_summary, save_audit_report


def parse_math(text):
    pattern = r'\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}'
    matches = re.findall(pattern, text)
    return matches[-1] if matches else ""


def construct_message(dataset_name, agents, question, idx, refinement_feedback=None):
    prefix_string = agent_prompt[dataset_name]['debate'][0]

    for agent in agents:
        if agent[idx]["role"] == "user":
            assert agent[idx + 1]["role"] == "assistant"
            agent_response = agent[idx + 1]["content"]
        else:
            agent_response = agent[idx]["content"]

        response = "\n\n One agent solution: ```{}```".format(agent_response)

        prefix_string = prefix_string + response

    if refinement_feedback:
        prefix_string += (
            "\n\nGraph-conditioned refinement feedback from the previous round:\n"
            + "\n\n".join(str(item) for item in refinement_feedback)
        )

    prefix_string = prefix_string + agent_prompt[dataset_name]['debate'][1]
    return {"role": "user", "content": prefix_string}


def construct_assistant_message(completion):
    return {"role": "assistant", "content": completion}


def generate_random_chess_move():
    possible_letter = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']
    possible_number = ['1', '2', '3', '4', '5', '6', '7', '8']
    return random.choice(possible_letter) + random.choice(possible_number)


def save_debate_graph(graph: ArgumentationGraph, sample_id: int, rep: int, out_dir: Path):
    graph_path = out_dir / f"graph_{sample_id}_{rep}.json"
    graph.save(str(graph_path))
    return str(graph_path)


def generate_round_summary(graph: ArgumentationGraph, round_num: int) -> dict:
    round_claims = graph.get_claims_by_round(round_num)
    round_edges = [e for e in graph.edges if 
                   any(n.round == round_num for n in graph.nodes.values() if n.node_id == e.source_id)]
    
    return {
        "round": round_num,
        "total_claims": len(round_claims),
        "claims_by_type": {
            node_type.value: sum(1 for c in round_claims if c.node_type.value == node_type.value)
            for node_type in NodeType
        },
        "attack_relations": sum(1 for e in round_edges if e.relation == RelationType.ATTACKS),
        "support_relations": sum(1 for e in round_edges if e.relation == RelationType.SUPPORTS),
        "citations": sum(1 for e in round_edges if e.relation == RelationType.CITES)
    }


def majority_vote(responses):
    votes = []
    for response in responses:
        choice = extract_choice_answer(response)
        if choice is not None:
            votes.append(choice)
    counts = Counter(votes)
    prediction = counts.most_common(1)[0][0] if counts else None
    return {"prediction": prediction, "votes": dict(counts)}


def main(args):
    enable_conflict_refinement = getattr(args, "conflict_refinement", True)
    conflict_top_k = getattr(args, "conflict_top_k", 1)
    conflict_k_hop = getattr(args, "conflict_k_hop", 3)
    out_dir = Path(args.output_dir, args.dataset, f"adv_{args.n_samples}_{args.n_agents}_{args.n_rounds}")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    audit_dir = out_dir / "audit_reports"
    audit_dir.mkdir(parents=True, exist_ok=True)

    if args.input_file:
        with open(args.input_file, 'r') as f:
            dataset = [json.loads(line) for line in f]
    else:
        dataset = get_dataset(dataset_name=args.dataset, n_samples=args.n_samples)

    client = OpenAI(
        api_key=os.environ.get("OPENAI_API_KEY", ""),
        base_url=os.environ.get("OPENAI_BASE_URL", "")
    )
    
    for current_rep in range(args.n_reps):
        print(f"Rep {current_rep}/{args.n_reps}")
        fname = f"adv_{args.dataset}_{args.n_samples}_{args.n_agents}_{args.n_rounds}_{current_rep}.jsonl"
        
        with open(out_dir / fname, 'w') as f:
            
            for i, sample in tqdm(enumerate(dataset), total=len(dataset)):
                try:
                    if args.input_file:
                        sample = sample['raw_task']

                    question, answer, raw_task = parse_question_answer(args.dataset, sample)

                    argumentation_graph = ArgumentationGraph()
                    
                    all_claims_by_round = {}
                    debate_history = []
                    refinement_feedback = []
                    final_agent_responses = {}
                    last_refinement_responses = []
                    
                    agent_contexts = []
                    for agent_id in range(args.n_agents):
                        agent_contexts.append([
                            {"role": "user", "content": question}
                        ])

                    debate_summary = {
                        "total_rounds": args.n_rounds,
                        "rounds": []
                    }
                    
                    for round_num in range(args.n_rounds):
                        print(f"\n{'=' * 80}")
                        print(f"Round {round_num + 1}")
                        print(f"{'=' * 80}")
                        
                        round_claims = []
                        
                        for agent_id, agent_context in enumerate(agent_contexts):
                            if round_num > 0:
                                other_agents = agent_contexts[:agent_id] + agent_contexts[agent_id + 1:]
                                message = construct_message(
                                    args.dataset,
                                    other_agents,
                                    question,
                                    2 * round_num - 1,
                                    refinement_feedback,
                                )
                                agent_context.append(message)

                            completion = query_model(client, agent_context, model_name=args.model_name)
                            assistant_message = construct_assistant_message(completion)
                            agent_context.append(assistant_message)
                            final_agent_responses[agent_id] = completion
                            debate_history.append(completion)
                            extraction = extract_arguments_with_llm(
                                client,
                                completion,
                                model=args.model_name,
                            )
                            mapping = integrate_extraction(
                                argumentation_graph,
                                extraction,
                                agent_id,
                                round_num,
                                completion,
                            )
                            for item in extraction.get("nodes", []):
                                node_id = mapping.get(str(item.get("id", "")))
                                if node_id is None:
                                    continue
                                node = argumentation_graph.nodes[node_id]
                                round_claims.append((node.content, node_id, agent_id))

                        all_claims_by_round[round_num] = round_claims

                        refinement = graph_conditioned_refinement(
                            argumentation_graph,
                            question=question,
                            history=debate_history,
                            client=client if enable_conflict_refinement else None,
                            agent_ids=(
                                tuple(range(min(args.n_agents, 3)))
                                if enable_conflict_refinement
                                else ()
                            ),
                            top_k=conflict_top_k,
                            k_hop=conflict_k_hop,
                            max_rounds=1,
                        )
                        debate_history.extend(refinement.responses)
                        refinement_feedback = list(refinement.responses)
                        last_refinement_responses = list(refinement.responses)
                        
                        round_summary = generate_round_summary(argumentation_graph, round_num)
                        round_summary.update({
                            "unresolved_conflicts": len(detect_unresolved_conflicts(argumentation_graph)),
                            "graph_inconsistency": compute_graph_inconsistency(argumentation_graph),
                            "refined_conflicts": len(refinement.selected_conflicts),
                            "refinement_responses": len(refinement.responses),
                        })
                        debate_summary["rounds"].append(round_summary)
                        
                        print(f"\nRound {round_num + 1} Statistics:")
                        print(f"  New Arguments: {len(round_claims)}")
                        print(f"  Attack Relations: {round_summary['attack_relations']}")
                        print(f"  Support Relations: {round_summary['support_relations']}")
                    
                    audit_summary = generate_audit_summary(argumentation_graph)
                    vote_responses = last_refinement_responses or list(final_agent_responses.values())
                    vote = majority_vote(vote_responses)
                    
                    graph_path = save_debate_graph(argumentation_graph, i, current_rep, out_dir)
                    
                    audit_report_path = audit_dir / f"audit_{i}_{current_rep}.json"
                    save_audit_report(argumentation_graph, str(audit_report_path))
                    
                    result = {
                        "id": i,
                        "dataset": args.dataset,
                        "question": question,
                        "correct_answer": answer,
                        "predicted_answer": vote["prediction"],
                        "majority_votes": vote["votes"],
                        "vote_responses": len(vote_responses),
                        "raw_task": raw_task,
                        "agent_responses": agent_contexts,
                        "argumentation_graph": argumentation_graph.to_dict(),
                        "audit_summary": audit_summary,
                        "debate_summary": debate_summary,
                        "graph_file": graph_path,
                        "audit_report": str(audit_report_path),
                        "metadata": {
                            "n_agents": args.n_agents,
                            "n_rounds": args.n_rounds,
                            "model_name": args.model_name,
                            "timestamp": time.time()
                        }
                    }
                    
                    f.write(json.dumps(result, ensure_ascii=False) + '\n')
                    f.flush()

                    print(f"\n{'=' * 80}")
                    print(f"Sample {i} debate completed")
                    print(f"Graph saved to: {graph_path}")
                    print(f"Audit report saved to: {audit_report_path}")
                    print(f"Graph stats: {len(argumentation_graph.nodes)} nodes, {len(argumentation_graph.edges)} edges")
                    print(f"Final conclusions: {len(audit_summary.get('conclusions', []))}")
                    print(f"Majority-vote answer: {vote['prediction']}")
                    print(f"Major disputes: {len(audit_summary.get('top_dispute_nodes', []))}")
                    
                    if audit_summary.get('conclusions'):
                        print("\nMain conclusions:")
                        for conclusion in audit_summary['conclusions'][:2]:
                            print(f"  - {conclusion['content'][:100]}...")
                    
                    if audit_summary.get('top_dispute_nodes'):
                        print("\nMajor disputes:")
                        for dispute in audit_summary['top_dispute_nodes'][:2]:
                            print(f"  - {dispute['content'][:100]}... (attacked {dispute['attack_count']} times)")

                except Exception as e:
                    print(f"Error processing sample {i}: {str(e)}")
                    import traceback
                    traceback.print_exc()
                    continue


if __name__ == "__main__":

    argparser = argparse.ArgumentParser(description="Moderate debate simulation with argumentation graph and audit features")
    argparser.add_argument("--dataset", type=str, default='mmlu',
                        choices=['mmlu', 'truthfulqa', 'medmcqa', 'scalr'])
    argparser.add_argument("--input_file", type=str, default=None, help="Input file path")
    argparser.add_argument("--n_samples", type=int, default=500, help="Number of samples")
    argparser.add_argument("--n_agents", type=int, default=30, help="Number of agents")
    argparser.add_argument("--n_rounds", type=int, default=30, help="Number of debate rounds")
    argparser.add_argument("--n_reps", type=int, default=1, help="Number of repetitions")
    argparser.add_argument("--output_dir", type=str, default='results/gpt-4o', help="Output directory")
    argparser.add_argument("--model_name", type=str, default='gpt-4o',
                        help="Model name to use")
    argparser.add_argument("--conflict_refinement", "--conflict-refinement",
                        dest="conflict_refinement", action="store_true", default=True,
                        help="Run API-backed graph-conditioned refinement")
    argparser.add_argument("--no_conflict_refinement", "--no-conflict-refinement",
                        dest="conflict_refinement", action="store_false",
                        help="Disable graph-conditioned refinement")
    argparser.add_argument("--conflict_top_k", type=int, default=1,
                        help="Number of unresolved conflicts selected per round")
    argparser.add_argument("--conflict_k_hop", type=int, default=3,
                        help="Directed incident hops used for local refinement context")

    args = argparser.parse_args()

    main(args)
