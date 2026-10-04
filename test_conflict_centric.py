import argparse
import json
import unittest
from openai import OpenAI
from argumentation_graph import ArgumentationGraph, NodeType, RelationType
from conflict_centric import (
    compute_graph_inconsistency,
    detect_unresolved_conflicts,
    graph_conditioned_refinement,
    propagate_conflict_utility,
    retrieve_khop_subgraph,
    select_topk_conflicts,
)


def graph_one():
    graph = ArgumentationGraph()
    nodes = {
        name: graph.add_node(text, kind, agent, 1)
        for name, text, kind, agent in [
            ("a", "Claim A", NodeType.CLAIM, 1),
            ("b", "Claim B", NodeType.CLAIM, 2),
            ("c", "Claim C", NodeType.CLAIM, 3),
            ("d", "Evidence D", NodeType.EVIDENCE, 4),
            ("e", "Evidence E", NodeType.EVIDENCE, 5),
            ("f", "Claim F", NodeType.CLAIM, 6),
            ("g", "Claim G", NodeType.CLAIM, 7),
            ("h", "Claim H", NodeType.CLAIM, 8),
        ]
    }
    graph.add_edge(nodes["a"], nodes["b"], RelationType.ATTACKS, 0.8)
    graph.add_edge(nodes["c"], nodes["d"], RelationType.ATTACKS, 0.8)
    graph.add_edge(nodes["e"], nodes["d"], RelationType.SUPPORTS, 0.7)
    graph.add_edge(nodes["f"], nodes["g"], RelationType.ATTACKS, 0.8)
    graph.add_edge(nodes["h"], nodes["f"], RelationType.ATTACKS, 0.8)
    return graph, nodes


def graph_two():
    graph = ArgumentationGraph()
    nodes = {
        name: graph.add_node(name.upper(), NodeType.CLAIM, 1, 1)
        for name in ("p", "q", "r", "s", "t")
    }
    graph.add_edge(nodes["p"], nodes["q"], RelationType.ATTACKS)
    graph.add_edge(nodes["p"], nodes["r"], RelationType.ATTACKS)
    graph.add_edge(nodes["s"], nodes["t"], RelationType.ATTACKS)
    return graph, nodes


def api_reply(prompt, model="gpt-4o"):
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
                "content": (
                    "Return JSON only."
                ),
            },
            {"role": "user", "content": prompt},
        ],
    )
    text = response.choices[0].message.content.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[-1]
        text = text[:-3].strip() if text.endswith("```") else text
    return json.loads(text)


class Checks(unittest.TestCase):
    def test_conflicts(self):
        graph, nodes = graph_one()
        found = detect_unresolved_conflicts(graph)
        pairs = {(item.source_id, item.target_id) for item in found}

        self.assertEqual(pairs, {(nodes["a"], nodes["b"]), (nodes["h"], nodes["f"])})
        self.assertEqual(compute_graph_inconsistency(graph, found), 2 / 4)

        graph.add_edge(nodes["b"], nodes["a"], RelationType.ATTACKS)
        graph.add_edge(nodes["g"], nodes["h"], RelationType.ATTACKS)
        found = detect_unresolved_conflicts(graph)
        self.assertEqual(found, [])
        self.assertEqual(compute_graph_inconsistency(graph, found), 0.0)

    def test_edges(self):
        graph = ArgumentationGraph()
        u = graph.add_node("U", NodeType.CLAIM, 1, 1)
        v = graph.add_node("V", NodeType.CLAIM, 2, 1)
        graph.add_edge(u, v, RelationType.ATTACKS)
        graph.add_edge(u, v, RelationType.ATTACKS)
        found = detect_unresolved_conflicts(graph)
        self.assertEqual(len(found), 2)
        self.assertEqual(compute_graph_inconsistency(graph, found), 1.0)

        raw = graph.to_dict()
        found = detect_unresolved_conflicts(raw)
        self.assertEqual(len(found), 2)
        self.assertEqual(compute_graph_inconsistency(raw, found), 1.0)

    def test_scores(self):
        graph, nodes = graph_two()
        conflicts = detect_unresolved_conflicts(graph)
        scores = propagate_conflict_utility(
            graph,
            conflicts,
            propagation_steps=3,
            retrieval_hops=1,
        )

        self.assertEqual(set(scores), {(nodes["p"], nodes["q"]), (nodes["p"], nodes["r"]), (nodes["s"], nodes["t"])})
        self.assertGreater(scores[(nodes["p"], nodes["q"])], scores[(nodes["s"], nodes["t"])])
        picked = select_topk_conflicts(conflicts, scores, k=2)
        self.assertEqual([item.key for item in picked], [(nodes["p"], nodes["q"]), (nodes["p"], nodes["r"])])

    def test_hops(self):
        graph = ArgumentationGraph()
        u = graph.add_node("U", NodeType.CLAIM, 1, 1)
        v = graph.add_node("V", NodeType.CLAIM, 2, 1)
        x = graph.add_node("X", NodeType.EVIDENCE, 3, 1)
        y = graph.add_node("Y", NodeType.EVIDENCE, 4, 1)
        graph.add_edge(u, v, RelationType.ATTACKS)
        graph.add_edge(x, u, RelationType.CITES)
        graph.add_edge(y, x, RelationType.SUPPORTS)
        conflict = detect_unresolved_conflicts(graph)[0]

        self.assertEqual(set(retrieve_khop_subgraph(graph, conflict, 0).nodes), {u, v})
        self.assertEqual(set(retrieve_khop_subgraph(graph, conflict, 1).nodes), {u, v, x})
        self.assertEqual(set(retrieve_khop_subgraph(graph, conflict, 2).nodes), {u, v, x, y})

    def test_refine(self):
        graph, nodes = graph_one()
        conflict = detect_unresolved_conflicts(graph)[0]
        local = retrieve_khop_subgraph(graph, conflict, 2)
        before = set(graph.nodes)
        calls = []

        def reply(prompt, context):
            calls.append((prompt, context))
            self.assertIn("Localized Graph Representation", prompt)
            return {
                "node_updates": [
                    {
                        "node_id": nodes["a"],
                        "content": "Claim A revised with the local evidence",
                        "node_type": "claim",
                    }
                ],
                "edge_updates": [
                    {
                        "source": nodes["b"],
                        "target": nodes["a"],
                        "relation": "supports",
                        "action": "add",
                    }
                ],
            }

        result = graph_conditioned_refinement(
            graph,
            conflict,
            local,
            question="Which claim is supported?",
            history=["agent 1: claim A", "agent 2: claim B"],
            refine_fn=reply,
        )
        self.assertEqual(len(calls), 1)
        self.assertEqual(set(result.graph.nodes), before)
        self.assertEqual(set(graph.nodes), before)
        self.assertIn("local evidence", result.graph.nodes[nodes["a"]].content)
        self.assertEqual(len(result.responses), 1)

        with self.assertRaises(ValueError):
            graph_conditioned_refinement(graph, conflict, None)
        with self.assertRaises(ValueError):
            graph_conditioned_refinement(
                graph,
                conflict,
                local,
                refine_fn=lambda prompt, context: {
                    "nodes": [{"id": "new-node", "text": "must not be added"}]
                },
            )

    def test_api_shape(self):
        graph, nodes = graph_one()
        conflict = detect_unresolved_conflicts(graph)[0]
        local = retrieve_khop_subgraph(graph, conflict, 1)
        before = set(graph.nodes)
        result = graph_conditioned_refinement(
            graph,
            conflict,
            local,
            refine_fn=lambda prompt, context: {
                "nodes": [
                    {
                        "id": nodes["a"],
                        "text": "A with an embedded counterpoint",
                        "type": "claim",
                    }
                ]
            },
        )
        self.assertEqual(set(result.graph.nodes), before)
        self.assertIn("embedded counterpoint", result.graph.nodes[nodes["a"]].content)

        class Message:
            content = json.dumps({
                "nodes": [
                    {
                        "id": nodes["a"],
                        "text": "A updated through the client path",
                        "type": "claim",
                    }
                ]
            })

        class Completion:
            choices = [type("Choice", (), {"message": Message()})()]

        class Client:
            class Chat:
                class Completions:
                    def create(self, **kwargs):
                        return Completion()

                completions = Completions()

            chat = Chat()

        result = graph_conditioned_refinement(
            graph,
            conflict,
            local,
            client=Client(),
            model="test-model",
        )
        self.assertIn("client path", result.graph.nodes[nodes["a"]].content)


def run_api_case():
    graph, nodes = graph_one()
    conflict = detect_unresolved_conflicts(graph)[0]
    local = retrieve_khop_subgraph(graph, conflict, 2)
    result = graph_conditioned_refinement(
        graph,
        conflict,
        local,
        question="Resolve the selected debate conflict.",
        refine_fn=api_reply,
    )
    assert set(result.graph.nodes) == set(graph.nodes)
    print("API case completed:", conflict.key, "updated", result.updated_node_ids)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--api", action="store_true", help="API")
    args = parser.parse_args()
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(Checks)
    status = unittest.TextTestRunner(verbosity=2).run(suite)
    if args.api and status.wasSuccessful():
        run_api_case()
