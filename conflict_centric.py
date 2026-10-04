from __future__ import annotations

import copy
from collections import deque
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple
from argumentation_graph import (
    ArgumentEdge,
    ArgumentNode,
    ArgumentationGraph,
    NodeType,
    RelationType,
)
PROPAGATION_STEPS = 3
TOP_K = 3
RETRIEVAL_HOPS = 3

REFINEMENT_PROMPT = (
    "You are an expert reasoning agent participating in a structured multi-agent "
    "debate framework. Your primary objective is to resolve unresolved "
    "argumentative conflicts using evidence-aware reasoning. You must critically "
    "evaluate competing claims, identify logical fallacies or weak assumptions, "
    "and produce a justifiable conclusion."
)


@dataclass(frozen=True)
class Conflict:
    source_id: str
    target_id: str
    edge_index: Optional[int] = None
    utility: float = 0.0

    @property
    def key(self) -> Tuple[str, str]:
        return self.source_id, self.target_id


@dataclass
class RefinementResult:
    graph: Any
    selected_conflicts: List[Conflict]
    contexts: List[Dict[str, Any]]
    responses: List[Any]
    updated_node_ids: List[str]


def _field(value: Any, *names: str, default: Any = None) -> Any:
    for name in names:
        if isinstance(value, Mapping) and name in value:
            return value[name]
        if hasattr(value, name):
            return getattr(value, name)
    return default


def _node_items(graph: Any) -> List[Tuple[str, Any]]:
    nodes = _field(graph, "nodes", default={})
    if isinstance(nodes, Mapping):
        return [(str(node_id), node) for node_id, node in nodes.items()]

    result: List[Tuple[str, Any]] = []
    for node in nodes or []:
        node_id = _field(node, "node_id", "id")
        if node_id is not None:
            result.append((str(node_id), node))
    return result


def _edge_items(graph: Any) -> List[Tuple[int, Any]]:
    edges = _field(graph, "edges", default=[])
    if callable(edges):
        edges = list(edges(data=True))
    return list(enumerate(edges or []))


def _node_id_set(graph: Any) -> Set[str]:
    return {node_id for node_id, _ in _node_items(graph)}


def _canonical_relation(value: Any) -> str:
    value = getattr(value, "value", value)
    value = str(value or "").strip().lower().replace("-", "_")
    aliases = {
        "support": "supports",
        "supports": "supports",
        "attack": "attacks",
        "attacks": "attacks",
        "counter": "attacks",
        "counterargument": "attacks",
        "cite": "cites",
        "cites": "cites",
        "citation": "cites",
        "derives": "derives_from",
        "derives_from": "derives_from",
    }
    return aliases.get(value, value)


def _edge_record(edge: Any) -> Tuple[Optional[str], Optional[str], str, float]:
    source = _field(edge, "source_id", "source", "u")
    target = _field(edge, "target_id", "target", "v")
    relation = _canonical_relation(_field(edge, "relation", "type", default=""))
    strength = _field(edge, "strength", "weight", default=1.0)
    try:
        strength = max(0.0, float(strength))
    except (TypeError, ValueError):
        strength = 1.0
    return (
        None if source is None else str(source),
        None if target is None else str(target),
        relation,
        strength,
    )


def _node_record(node_id: str, node: Any) -> Dict[str, Any]:
    node_type = _field(node, "node_type", "type", default="")
    node_type = getattr(node_type, "value", node_type)
    return {
        "node_id": node_id,
        "content": str(_field(node, "content", "text", default="")),
        "node_type": str(node_type),
        "agent_id": _field(node, "agent_id", "agent", default=None),
        "round": _field(node, "round", "round_num", default=None),
        "confidence": _field(node, "confidence", default=None),
        "metadata": _field(node, "metadata", default={}) or {},
    }


def _coerce_conflict(value: Any, edge_index: Optional[int] = None) -> Conflict:
    if isinstance(value, Conflict):
        return value
    if isinstance(value, Mapping):
        source = _field(value, "source_id", "source", "u")
        target = _field(value, "target_id", "target", "v")
        utility = _field(value, "utility", "score", default=0.0)
        index = _field(value, "edge_index", default=edge_index)
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)) and len(value) >= 2:
        source, target = value[0], value[1]
        utility = value[2] if len(value) > 2 else 0.0
        index = edge_index
    else:
        raise TypeError("A conflict must be Conflict, mapping, or a pair of node IDs")

    if source is None or target is None:
        raise ValueError("A conflict must contain source and target node IDs")
    return Conflict(str(source), str(target), index, float(utility or 0.0))


def _incoming_relation_index(graph: Any) -> Dict[str, List[Tuple[str, str]]]:
    incoming: Dict[str, List[Tuple[str, str]]] = {}
    for _, edge in _edge_items(graph):
        source, target, relation, _ = _edge_record(edge)
        if source is None or target is None:
            continue
        incoming.setdefault(target, []).append((source, relation))
    return incoming


def detect_unresolved_conflicts(graph: Any) -> List[Conflict]:
    incoming = _incoming_relation_index(graph)
    conflicts: List[Conflict] = []
    for edge_index, edge in _edge_items(graph):
        source, target, relation, _ = _edge_record(edge)
        if source is None or target is None or relation != "attacks":
            continue

        refutes_source = any(edge_relation == "attacks" for _, edge_relation in incoming.get(source, []))
        supports_target = any(edge_relation == "supports" for _, edge_relation in incoming.get(target, []))
        if not refutes_source and not supports_target:
            conflicts.append(Conflict(source, target, edge_index))
    return conflicts


def compute_graph_inconsistency(
    graph: Any,
    unresolved_conflicts: Sequence[Any],
) -> float:
    attack_count = 0
    for _, edge in _edge_items(graph):
        if _edge_record(edge)[2] == "attacks":
            attack_count += 1
    if attack_count == 0:
        return 0.0

    unresolved_count = len(unresolved_conflicts)
    return unresolved_count / attack_count


def _undirected_weighted_adjacency(graph: Any) -> Tuple[List[str], List[List[float]]]:
    node_ids = [node_id for node_id, _ in _node_items(graph)]
    positions = {node_id: index for index, node_id in enumerate(node_ids)}
    matrix = [[0.0 for _ in node_ids] for _ in node_ids]

    for _, edge in _edge_items(graph):
        source, target, _, strength = _edge_record(edge)
        if source not in positions or target not in positions:
            continue
        weight = strength if strength > 0.0 else 1.0
        source_index = positions[source]
        target_index = positions[target]
        matrix[source_index][target_index] += weight
        if source_index != target_index:
            matrix[target_index][source_index] += weight
    for index, row in enumerate(matrix):
        if not any(row):
            row[index] = 1.0
    return node_ids, matrix


def _propagate(
    matrix: List[List[float]], initial: List[float], steps: int
) -> List[float]:
    scores = list(initial)
    for _ in range(steps):
        next_scores = [0.0 for _ in scores]
        for row_index, row in enumerate(matrix):
            degree = sum(row)
            if degree <= 0.0:
                next_scores[row_index] = scores[row_index]
                continue
            next_scores[row_index] = sum(
                (weight / degree) * scores[column_index]
                for column_index, weight in enumerate(row)
                if weight > 0.0
            )
        scores = next_scores
    return scores


def _conflict_region(graph: Any, conflict: Conflict, hops: int) -> Set[str]:
    node_ids, matrix = _undirected_weighted_adjacency(graph)
    positions = {node_id: index for index, node_id in enumerate(node_ids)}
    seeds = [
        node_id
        for node_id in (conflict.source_id, conflict.target_id)
        if node_id in positions
    ]
    if not seeds:
        return set()

    hops = max(0, int(hops))
    distances = {node_id: 0 for node_id in seeds}
    queue = deque(seeds)
    while queue:
        current = queue.popleft()
        if distances[current] >= hops:
            continue
        current_index = positions[current]
        for neighbor_index, weight in enumerate(matrix[current_index]):
            if weight <= 0.0:
                continue
            neighbor = node_ids[neighbor_index]
            if neighbor not in distances:
                distances[neighbor] = distances[current] + 1
                queue.append(neighbor)
    return set(distances)


def propagate_conflict_utility(
    graph: Any,
    conflicts: Sequence[Any],
    propagation_steps: int = PROPAGATION_STEPS,
    retrieval_hops: int = RETRIEVAL_HOPS,
) -> Dict[Tuple[str, str], float]:

    if propagation_steps < 0:
        raise ValueError("propagation_steps must be non-negative")
    if retrieval_hops < 0:
        raise ValueError("retrieval_hops must be non-negative")

    normalized_conflicts = [_coerce_conflict(item) for item in conflicts]
    node_ids, matrix = _undirected_weighted_adjacency(graph)
    positions = {node_id: index for index, node_id in enumerate(node_ids)}
    initial = [0.0 for _ in node_ids]
    for conflict in normalized_conflicts:
        for node_id in (conflict.source_id, conflict.target_id):
            if node_id in positions:
                initial[positions[node_id]] += 1.0

    propagated = _propagate(matrix, initial, int(propagation_steps))
    scores: Dict[Tuple[str, str], float] = {}
    for conflict in normalized_conflicts:
        region = _conflict_region(graph, conflict, retrieval_hops)
        utility = sum(propagated[positions[node_id]] for node_id in region)
        scores[conflict.key] = utility
    return scores


def _lookup_utility(
    conflict: Conflict, utilities: Optional[Mapping[Any, float]]
) -> float:
    if utilities is None:
        return conflict.utility
    for key in (conflict.key, conflict, f"{conflict.source_id}->{conflict.target_id}"):
        try:
            if key in utilities:
                return float(utilities[key])
        except TypeError:
            continue
    return conflict.utility


def select_topk_conflicts(
    conflicts: Sequence[Any],
    utilities: Optional[Mapping[Any, float]] = None,
    k: int = TOP_K,
) -> List[Conflict]:
    if k < 0:
        raise ValueError("k must be non-negative")
    normalized = [_coerce_conflict(item, index) for index, item in enumerate(conflicts)]
    scored = [
        (index, conflict, _lookup_utility(conflict, utilities))
        for index, conflict in enumerate(normalized)
    ]
    scored.sort(key=lambda item: (-item[2], item[1].source_id, item[1].target_id, item[0]))
    return [
        Conflict(conflict.source_id, conflict.target_id, conflict.edge_index, score)
        for _, conflict, score in scored[:k]
    ]


def _induced_subgraph(graph: Any, selected_ids: Set[str]) -> Any:
    if isinstance(graph, ArgumentationGraph):
        result = ArgumentationGraph()
        result.nodes = {
            node_id: copy.deepcopy(node)
            for node_id, node in _node_items(graph)
            if node_id in selected_ids
        }
        result.edges = [
            copy.deepcopy(edge)
            for _, edge in _edge_items(graph)
            if _edge_record(edge)[0] in selected_ids
            and _edge_record(edge)[1] in selected_ids
        ]
        result.node_counter = getattr(graph, "node_counter", len(result.nodes))
        return result

    result = copy.deepcopy(graph)
    nodes = _field(result, "nodes", default={})
    if isinstance(nodes, Mapping):
        result["nodes"] = {
            node_id: copy.deepcopy(node)
            for node_id, node in nodes.items()
            if str(node_id) in selected_ids
        }
    elif nodes is not None:
        result["nodes"] = [
            copy.deepcopy(node)
            for node in nodes
            if str(_field(node, "node_id", "id")) in selected_ids
        ]

    edges = _field(result, "edges", default=[])
    result["edges"] = [
        copy.deepcopy(edge)
        for edge in edges or []
        if _edge_record(edge)[0] in selected_ids
        and _edge_record(edge)[1] in selected_ids
    ]
    return result


def retrieve_khop_subgraph(graph: Any, conflict: Any, k: int = RETRIEVAL_HOPS) -> Any:
    if k < 0:
        raise ValueError("k must be non-negative")
    normalized = _coerce_conflict(conflict)
    return _induced_subgraph(graph, _conflict_region(graph, normalized, int(k)))


def _format_node(node_id: str, node: Any) -> str:
    record = _node_record(node_id, node)
    return (
        f"- {node_id} [{record['node_type']}]: {record['content']} "
        f"(confidence={record['confidence']})"
    )


def serialize_subgraph(graph: Any) -> str:
    lines = ["[Nodes]"]
    for node_id, node in _node_items(graph):
        lines.append(_format_node(node_id, node))
    lines.append("[Relations]")
    for _, edge in _edge_items(graph):
        source, target, relation, strength = _edge_record(edge)
        if source is not None and target is not None:
            lines.append(f"- {source} -[{relation}, strength={strength}]-> {target}")
    return "\n".join(lines)


def _node_type_is(node: Any, expected: str) -> bool:
    value = _field(node, "node_type", "type", default="")
    return str(getattr(value, "value", value)).lower() == expected


def _build_refinement_prompt(
    question: str,
    conflict: Conflict,
    subgraph: Any,
    history: Sequence[str],
) -> str:
    node_map = dict(_node_items(subgraph))
    source = node_map.get(conflict.source_id)
    target = node_map.get(conflict.target_id)
    evidence = [
        _format_node(node_id, node)
        for node_id, node in _node_items(subgraph)
        if _node_type_is(node, "evidence")
    ]
    assumptions = [
        _format_node(node_id, node)
        for node_id, node in _node_items(subgraph)
        if _node_type_is(node, "assumption")
    ]
    counter_arguments = []
    for _, edge in _edge_items(subgraph):
        source_id, target_id, relation, _ = _edge_record(edge)
        if relation == "attacks" and target_id in {conflict.source_id, conflict.target_id}:
            if source_id in node_map:
                counter_arguments.append(_format_node(source_id, node_map[source_id]))

    history_text = "\n".join(str(item) for item in history[-3:]) or "(none)"
    return "\n".join(
        [
            f"Original Question: {question or '(not provided)'}",
            "Unresolved Conflict: The debate has identified a disagreement between the following claims that remains unresolved after previous debate rounds.",
            f"- Claim A ({conflict.source_id}): {_node_record(conflict.source_id, source)['content'] if source else '(missing)'}",
            f"- Claim B ({conflict.target_id}): {_node_record(conflict.target_id, target)['content'] if target else '(missing)'}",
            "Supporting Evidence: The following evidence nodes from the local graph are relevant to this conflict.",
            "\n".join(evidence) or "- (none)",
            "Related Assumptions: The following assumptions underlie the conflicting claims.",
            "\n".join(assumptions) or "- (none)",
            "Neighboring Counter-Arguments: The following counter-arguments target one of the conflict claims.",
            "\n".join(counter_arguments) or "- (none)",
            "Localized Graph Representation:",
            serialize_subgraph(subgraph),
            "Recent Debate History:",
            history_text,
            "Task Description:",
            "1. Identify unsupported or weak assumptions that undermine either claim.",
            "2. Resolve semantic or logical inconsistencies when possible.",
            "3. Introduce additional supporting evidence if the existing evidence is insufficient.",
            "4. Preserve logically consistent conclusions that follow from the available evidence.",
            "5. Explicitly explain why one claim should be preferred over the other, or propose a synthesis if both have merit.",
            "Output Requirements:",
            "1. new_claims: state claims generated to address the unresolved conflict.",
            "2. supporting_evidence: state supporting evidence and provenance for each new claim where possible.",
            "3. resolution: analyze the structural conflict, identify the key assumption or evidence gap, and explain step by step how the conflict is resolved.",
            "4. refined_conclusion: provide the final answer or position.",
            "Return JSON only with the following fields:",
            '{"new_claims":[{"node_id":"existing-node-id","text":"..."}],"supporting_evidence":[{"node_id":"existing-node-id","text":"...","provenance":"..."}],"resolution":"...","refined_conclusion":"...","node_updates":[{"node_id":"existing-node-id","content":"...","node_type":"claim|evidence|assumption|conclusion"}],"edge_updates":[{"source":"existing-node-id","target":"existing-node-id","relation":"supports|attacks|cites","action":"add|update|remove"}],"remove_edges":[]}',
            "All node_id, source, and target values must refer to nodes that already exist in the graph.",
            "New arguments must be integrated into the text of existing nodes. Do not create graph nodes.",
            "The node_updates field is authoritative for graph text and type changes; edge_updates and remove_edges are authoritative for relation changes.",
        ]
    )


def _mapping_updates(response: Any) -> Tuple[List[Any], List[Any], List[Any]]:
    if not isinstance(response, Mapping):
        return [], [], []

    node_updates = response.get(
        "node_updates", response.get("nodes", response.get("updates", []))
    )
    edge_updates = response.get("edge_updates", response.get("relations", []))
    if not edge_updates and isinstance(response.get("edges"), (list, tuple, Mapping)):
        edge_updates = response.get("edges")
    removed_edges = response.get("remove_edges", response.get("removed_edges", []))

    if isinstance(node_updates, Mapping):
        node_updates = [dict(fields, node_id=node_id) for node_id, fields in node_updates.items()]
    if node_updates and not isinstance(node_updates, (str, bytes)):
        possible_node_updates = []
        possible_edge_updates = []
        for update in node_updates:
            if _field(update, "source_id", "source", "u") is not None or _field(update, "target_id", "target", "v") is not None:
                possible_edge_updates.append(update)
            else:
                possible_node_updates.append(update)
        if possible_edge_updates and not edge_updates:
            edge_updates = possible_edge_updates
            node_updates = possible_node_updates
    if isinstance(edge_updates, Mapping):
        edge_updates = [dict(fields, **{"source": source, "target": target}) for (source, target), fields in edge_updates.items()]
    if isinstance(removed_edges, Mapping):
        removed_edges = [removed_edges]
    return list(node_updates or []), list(edge_updates or []), list(removed_edges or [])


def _set_node_field(node: Any, name: str, value: Any) -> None:
    if isinstance(node, Mapping):
        node[name] = value
    else:
        setattr(node, name, value)


def _apply_node_updates(graph: Any, updates: Sequence[Any]) -> List[str]:
    node_map = dict(_node_items(graph))
    updated: List[str] = []
    for update in updates:
        node_id = _field(update, "node_id", "id")
        if node_id is None:
            raise ValueError("Every node update must specify an existing node_id")
        node_id = str(node_id)
        if node_id not in node_map:
            raise ValueError(f"Refinement attempted to create unknown node {node_id!r}")
        node = node_map[node_id]
        content = _field(update, "content", "text")
        node_type = _field(update, "node_type", "type")
        confidence = _field(update, "confidence")
        metadata = _field(update, "metadata")
        if content is not None:
            _set_node_field(node, "content" if not isinstance(node, Mapping) or "content" in node else "text", str(content))
        if node_type is not None:
            if isinstance(node, ArgumentNode):
                try:
                    node_type = NodeType(str(getattr(node_type, "value", node_type)).lower())
                except ValueError:
                    raise ValueError(f"Unknown node type {node_type!r}")
            _set_node_field(node, "node_type" if not isinstance(node, Mapping) or "node_type" in node else "type", node_type)
        if confidence is not None:
            _set_node_field(node, "confidence", max(0.0, min(1.0, float(confidence))))
        if metadata is not None:
            _set_node_field(node, "metadata", dict(metadata))
        updated.append(node_id)
    return updated


def _edge_matches(edge: Any, source: str, target: str, relation: Optional[str] = None) -> bool:
    edge_source, edge_target, edge_relation, _ = _edge_record(edge)
    return (
        edge_source == source
        and edge_target == target
        and (relation is None or edge_relation == relation)
    )


def _make_edge(graph: Any, source: str, target: str, relation: str, strength: float) -> Any:
    if isinstance(graph, ArgumentationGraph):
        return ArgumentEdge(source, target, RelationType(relation), strength)
    return {"source": source, "target": target, "relation": relation, "strength": strength}


def _apply_edge_updates(graph: Any, updates: Sequence[Any], removed: Sequence[Any]) -> None:
    node_ids = _node_id_set(graph)
    edges = _field(graph, "edges", default=[])
    if callable(edges):
        raise TypeError("graph_conditioned_refinement requires a mutable edge list")
    if edges is None:
        edges = []
        if isinstance(graph, Mapping):
            graph["edges"] = edges
        else:
            setattr(graph, "edges", edges)

    def remove_one(spec: Any) -> None:
        source = _field(spec, "source_id", "source", "u")
        target = _field(spec, "target_id", "target", "v")
        relation_value = _field(spec, "relation", "type")
        if source is None or target is None:
            raise ValueError("Every edge update must specify source and target")
        relation = _canonical_relation(relation_value) if relation_value is not None else None
        for index in range(len(edges) - 1, -1, -1):
            if _edge_matches(edges[index], str(source), str(target), relation):
                del edges[index]

    for spec in removed:
        remove_one(spec)

    for update in updates:
        source = _field(update, "source_id", "source", "u")
        target = _field(update, "target_id", "target", "v")
        if source is None or target is None:
            raise ValueError("Every edge update must specify source and target")
        source, target = str(source), str(target)
        if source not in node_ids or target not in node_ids:
            raise ValueError("Refinement edge updates may only reference existing nodes")
        relation = _canonical_relation(_field(update, "relation", "type"))
        if relation not in {"supports", "attacks", "cites", "derives_from"}:
            raise ValueError(f"Unknown relation type {relation!r}")
        strength = _field(update, "strength", "weight", default=0.5)
        strength = max(0.0, min(1.0, float(strength)))
        action = str(_field(update, "action", default="update")).lower()
        matching = [edge for edge in edges if _edge_matches(edge, source, target)]
        if action == "remove":
            for edge in matching:
                edges.remove(edge)
            continue
        if action == "add":
            edges.append(_make_edge(graph, source, target, relation, strength))
            continue
        if matching:
            for edge in matching:
                if isinstance(edge, Mapping):
                    edge["relation"] = relation
                    edge["strength"] = strength
                else:
                    edge.relation = RelationType(relation)
                    edge.strength = strength
        else:
            edges.append(_make_edge(graph, source, target, relation, strength))


def _client_response(client: Any, model: str, prompt: str) -> Any:
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": REFINEMENT_PROMPT},
            {"role": "user", "content": prompt},
        ],
    )
    content = response.choices[0].message.content
    if not isinstance(content, str):
        return content
    content = content.strip()
    if content.startswith("```"):
        content = content.split("\n", 1)[-1]
        if content.endswith("```"):
            content = content[:-3].rstrip()
    try:
        import json

        return json.loads(content)
    except (TypeError, ValueError):
        return content


def graph_conditioned_refinement(
    graph: Any,
    conflict: Any,
    subgraph: Any,
    *,
    question: str = "",
    history: Sequence[str] = (),
    refine_fn: Optional[Callable[[str, Dict[str, Any]], Any]] = None,
    client: Any = None,
    model: str = "gpt-4o-mini",
    in_place: bool = False,
) -> RefinementResult:
    if subgraph is None:
        raise ValueError("Pass a subgraph returned by retrieve_khop_subgraph")
    if refine_fn is not None and client is not None:
        raise ValueError("Use either refine_fn or client, not both")

    normalized = _coerce_conflict(conflict)
    prompt = _build_refinement_prompt(question, normalized, subgraph, history)
    context = {
        "conflict": normalized,
        "subgraph": subgraph,
        "serialized_subgraph": serialize_subgraph(subgraph),
        "prompt": prompt,
    }
    refined_graph = graph if in_place else copy.deepcopy(graph)
    original_node_ids = _node_id_set(refined_graph)
    responses: List[Any] = []

    if refine_fn is not None:
        responses.append(refine_fn(prompt, context))
    elif client is not None:
        responses.append(_client_response(client, model, prompt))

    updated_node_ids: List[str] = []
    for response in responses:
        node_updates, edge_updates, removed_edges = _mapping_updates(response)
        updated_node_ids.extend(_apply_node_updates(refined_graph, node_updates))
        _apply_edge_updates(refined_graph, edge_updates, removed_edges)

    if _node_id_set(refined_graph) != original_node_ids:
        raise AssertionError("Graph-conditioned refinement must preserve node IDs")
    return RefinementResult(
        refined_graph,
        [normalized],
        [context],
        responses,
        list(dict.fromkeys(updated_node_ids)),
    )


