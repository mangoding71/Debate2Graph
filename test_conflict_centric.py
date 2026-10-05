import argparse
import json
import unittest

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

from argumentation_graph import ArgumentationGraph, NodeType, RelationType
from conflict_centric import (
    REFINEMENT_SYSTEM_PROMPT,
    build_argument_extraction_prompt,
    compute_graph_inconsistency,
    detect_unresolved_conflicts,
    extract_arguments_with_llm,
    graph_conditioned_refinement,
    integrate_extraction,
    propagate_conflict_utility,
    retrieve_khop_subgraph,
    select_topk_conflicts,
)


def sample_graph():
    graph = ArgumentationGraph()
    nodes = {
        name: graph.add_node(text, kind, agent, 1)
        for name, text, kind, agent in [
            ("a", "Claim A", NodeType.CLAIM, 1),
            ("b", "Claim B", NodeType.CLAIM, 2),
            ("c", "Claim C", NodeType.CLAIM, 3),
            ("d", "Evidence D", NodeType.EVIDENCE, 4),
            ("e", "Claim E", NodeType.CLAIM, 5),
            ("f", "Evidence F", NodeType.EVIDENCE, 6),
        ]
    }
    graph.add_edge(nodes["a"], nodes["b"], RelationType.ATTACKS, 0.8)
    graph.add_edge(nodes["c"], nodes["d"], RelationType.ATTACKS, 0.8)
    graph.add_edge(nodes["d"], nodes["c"], RelationType.ATTACKS, 0.8)
    graph.add_edge(nodes["e"], nodes["f"], RelationType.ATTACKS, 0.8)
    graph.add_edge(nodes["f"], nodes["a"], RelationType.SUPPORTS, 0.7)
    graph.add_edge(nodes["f"], nodes["d"], RelationType.CITES, 0.6)
    return graph, nodes


def directional_graph():
    graph = ArgumentationGraph()
    nodes = {
        name: graph.add_node(name.upper(), NodeType.CLAIM, 1, 1)
        for name in ("u", "v", "x", "y", "z")
    }
    graph.add_edge(nodes["u"], nodes["v"], RelationType.ATTACKS)
    graph.add_edge(nodes["x"], nodes["u"], RelationType.CITES)
    graph.add_edge(nodes["y"], nodes["x"], RelationType.SUPPORTS)
    graph.add_edge(nodes["v"], nodes["z"], RelationType.SUPPORTS)
    return graph, nodes


class Completion:
    def __init__(self, content):
        self.choices = [type("Choice", (), {"message": type("Message", (), {"content": content})()})()]


class FakeClient:
    def __init__(self, content):
        self.content = content
        self.messages = []

        outer = self

        class Completions:
            def create(self, **kwargs):
                outer.messages.append(kwargs["messages"])
                return Completion(outer.content)

        class Chat:
            completions = Completions()

        self.chat = Chat()


def api_reply(prompt, model="gpt-4o"):
    if OpenAI is None:
        raise RuntimeError("Install openai to run the optional API check")
    client = OpenAI(
        api_key="",
        base_url="",
    )
    response = client.chat.completions.create(
        model=model,
        temperature=0,
        messages=[
            {
                "role": "system",
                "content": REFINEMENT_SYSTEM_PROMPT,
            },
            {"role": "user", "content": prompt},
        ],
    )
    return response.choices[0].message.content.strip()


class Checks(unittest.TestCase):
    def test_conflicts_and_inconsistency(self):
        graph, nodes = sample_graph()
        conflicts = detect_unresolved_conflicts(graph)
        pairs = {item.key for item in conflicts}
        self.assertEqual(pairs, {(nodes["a"], nodes["b"]), (nodes["e"], nodes["f"])})
        self.assertEqual(compute_graph_inconsistency(graph, conflicts), 2 / 4)

        graph.add_edge(nodes["b"], nodes["a"], RelationType.ATTACKS)
        graph.add_edge(nodes["f"], nodes["e"], RelationType.ATTACKS)
        self.assertEqual(detect_unresolved_conflicts(graph), [])
        self.assertEqual(compute_graph_inconsistency(graph), 0.0)

    def test_components_keep_direction(self):
        graph, nodes = directional_graph()
        conflicts = detect_unresolved_conflicts(graph)
        utilities = propagate_conflict_utility(
            graph,
            conflicts,
            propagation_steps=3,
        )
        picked = select_topk_conflicts(conflicts, utilities, k=1)
        self.assertEqual(len(picked), 1)

        local = retrieve_khop_subgraph(graph, picked[0], k=1)
        self.assertEqual(len(local.nodes), 4)
        self.assertIn("X", {node.content for node in local.nodes.values()})
        self.assertNotIn("Y", {node.content for node in local.nodes.values()})
        self.assertIn(
            (RelationType.CITES, nodes["x"], nodes["u"]),
            {(edge.relation, edge.source_id, edge.target_id) for edge in local.edges},
        )

        outgoing = retrieve_khop_subgraph(graph, picked[0], k=1, direction="out")
        self.assertNotIn("X", {node.content for node in outgoing.nodes.values()})
        wider = retrieve_khop_subgraph(graph, picked[0], k=2)
        self.assertGreaterEqual(len(wider.nodes), len(local.nodes))
        self.assertIn("Y", {node.content for node in wider.nodes.values()})

    def test_extraction_matches_appendix_schema(self):
        payload = {
            "nodes": [
                {"id": "n1", "type": "claim", "text": "A claim", "confidence": 0.9},
                {"id": "n2", "type": "evidence", "text": "Evidence", "confidence": 0.8},
            ],
            "relations": [
                {
                    "source": "n2",
                    "target": "n1",
                    "type": "supports",
                    "rationale": "The evidence supports the claim.",
                    "confidence": 0.8,
                }
            ],
        }
        client = FakeClient(json.dumps(payload))
        result = extract_arguments_with_llm(client, "The evidence supports the claim.")
        self.assertEqual(set(result), {"nodes", "relations"})
        self.assertIn("nodes and relations", client.messages[0][1]["content"])
        self.assertIn("argumentation analyst", client.messages[0][0]["content"])

        graph = ArgumentationGraph()
        mapping = integrate_extraction(graph, result, agent_id=2, round_num=1)
        self.assertEqual(set(mapping), {"n1", "n2"})
        self.assertEqual(len(graph.nodes), 2)
        self.assertEqual(len(graph.edges), 1)
        self.assertIn("confidence", build_argument_extraction_prompt("text"))

    def test_refinement_preserves_nodes(self):
        graph, nodes = sample_graph()
        before_nodes = set(graph.nodes)
        calls = []

        def refine(prompt, context):
            calls.append((prompt, context))
            return (
                "[New Claims]\nThe qualification changes the interpretation.\n"
                "[Supporting Evidence]\nEvidence F is relevant.\n"
                "[Resolution]\nThe original attack is resolved.\n"
                "[Refined Conclusion]\nClaim B should be retained with the qualification."
            )

        result = graph_conditioned_refinement(
            graph,
            question="Which claim should be retained?",
            history=["agent 1: Claim A", "agent 2: Claim B"],
            refine_fn=refine,
            agent_ids=(1,),
            top_k=2,
            k_hop=2,
            max_rounds=3,
        )
        self.assertEqual(len(calls), 2)
        self.assertEqual(set(graph.nodes), before_nodes)
        self.assertEqual(set(result.graph.nodes), before_nodes)
        self.assertIn("New Claims", graph.nodes[nodes["b"]].content)
        self.assertIn("Refined Conclusion", graph.nodes[nodes["b"]].content)
        self.assertEqual(result.inconsistency_after, 0.0)
        self.assertIn(
            (nodes["a"], nodes["b"], RelationType.ATTACKS),
            {(edge.source_id, edge.target_id, edge.relation) for edge in graph.edges},
        )
        self.assertIn(
            (nodes["f"], nodes["b"], RelationType.SUPPORTS),
            {(edge.source_id, edge.target_id, edge.relation) for edge in graph.edges},
        )
        self.assertTrue(all("node_updates" not in call[0] for call in calls))

    def test_empty_refinement_does_not_resolve(self):
        graph, _ = sample_graph()
        conflict = detect_unresolved_conflicts(graph)[0]
        before = compute_graph_inconsistency(graph)
        edges_before = len(graph.edges)
        result = graph_conditioned_refinement(
            graph,
            conflict=conflict,
            refine_fn=lambda prompt, context: "",
            agent_ids=(1,),
            max_rounds=1,
        )
        self.assertEqual(result.inconsistency_before, before)
        self.assertEqual(result.inconsistency_after, before)
        self.assertEqual(len(graph.edges), edges_before)

    def test_client_refinement_uses_system_prompt(self):
        graph, nodes = sample_graph()
        client = FakeClient(
            "[New Claims]\nA qualification.\n"
            "[Supporting Evidence]\nEvidence F.\n"
            "[Resolution]\nResolved.\n"
            "[Refined Conclusion]\nKeep the qualified claim."
        )
        result = graph_conditioned_refinement(
            graph,
            conflict=detect_unresolved_conflicts(graph)[0],
            question="Resolve the conflict.",
            client=client,
            agent_ids=(1,),
            max_rounds=1,
        )
        self.assertEqual(len(result.responses), 1)
        self.assertIn("expert reasoning agent", client.messages[0][0]["content"])
        self.assertEqual(set(graph.nodes), set(nodes.values()))

    def test_reference_pipeline_without_model(self):
        graph, _ = sample_graph()
        result = graph_conditioned_refinement(
            graph,
            top_k=2,
            k_hop=1,
        )
        self.assertEqual(len(result.selected_conflicts), 2)
        self.assertEqual(len(result.responses), 0)
        self.assertEqual(set(result.contexts), {item.key for item in result.selected_conflicts})


def run_api_case():
    graph, _ = sample_graph()
    result = graph_conditioned_refinement(
        graph,
        question="Resolve the selected debate conflict.",
        refine_fn=api_reply,
        agent_ids=(0,),
        top_k=1,
        max_rounds=1,
    )
    print("API case completed:", len(result.responses), "response(s)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--api", action="store_true")
    args = parser.parse_args()
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(Checks)
    status = unittest.TextTestRunner(verbosity=2).run(suite)
    if args.api and status.wasSuccessful():
        run_api_case()
