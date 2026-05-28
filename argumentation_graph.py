# argumentation_graph.py
import json
from datetime import datetime
from typing import List, Dict, Any, Optional
from enum import Enum
from dataclasses import dataclass, field, asdict


class RelationType(Enum):
    SUPPORTS = "supports"
    ATTACKS = "attacks"
    CITES = "cites"
    DERIVES_FROM = "derives_from"


class NodeType(Enum):
    CLAIM = "claim"
    EVIDENCE = "evidence"
    ASSUMPTION = "assumption"
    QUESTION = "question"
    CONCLUSION = "conclusion"


@dataclass
class ArgumentNode:
    node_id: str
    content: str
    node_type: NodeType
    agent_id: int
    round: int
    confidence: float = 0.5
    metadata: Dict = field(default_factory=dict)
    
    def to_dict(self):
        return {
            "node_id": self.node_id,
            "content": self.content,
            "node_type": self.node_type.value,
            "agent_id": self.agent_id,
            "round": self.round,
            "confidence": self.confidence,
            "metadata": self.metadata,
            "timestamp": datetime.now().isoformat()
        }


@dataclass
class ArgumentEdge:
    source_id: str
    target_id: str
    relation: RelationType
    strength: float = 0.5
    metadata: Dict = field(default_factory=dict)
    
    def to_dict(self):
        return {
            "source": self.source_id,
            "target": self.target_id,
            "relation": self.relation.value,
            "strength": self.strength,
            "metadata": self.metadata
        }


class ArgumentationGraph:
    
    def __init__(self):
        self.nodes: Dict[str, ArgumentNode] = {}
        self.edges: List[ArgumentEdge] = []
        self.node_counter = 0
        
    def add_node(self, content: str, node_type: NodeType, agent_id: int, 
                 round: int, confidence: float = 0.5, metadata: Dict = None) -> str:
        self.node_counter += 1
        node_id = f"r{round}_a{agent_id}_n{self.node_counter}"
        node = ArgumentNode(
            node_id=node_id,
            content=content,
            node_type=node_type,
            agent_id=agent_id,
            round=round,
            confidence=confidence,
            metadata=metadata or {}
        )
        self.nodes[node_id] = node
        return node_id
    
    def add_edge(self, source_id: str, target_id: str, 
                 relation: RelationType, strength: float = 0.5):
        if source_id in self.nodes and target_id in self.nodes:
            self.edges.append(ArgumentEdge(source_id, target_id, relation, strength))
    
    def get_claims_by_agent(self, agent_id: int) -> List[ArgumentNode]:
        return [n for n in self.nodes.values() if n.agent_id == agent_id]
    
    def get_claims_by_round(self, round: int) -> List[ArgumentNode]:
        return [n for n in self.nodes.values() if n.round == round]
    
    def get_supporters(self, node_id: str) -> List[ArgumentNode]:
        supporters = []
        for edge in self.edges:
            if edge.target_id == node_id and edge.relation == RelationType.SUPPORTS:
                if edge.source_id in self.nodes:
                    supporters.append(self.nodes[edge.source_id])
        return supporters
    
    def get_attackers(self, node_id: str) -> List[ArgumentNode]:
        attackers = []
        for edge in self.edges:
            if edge.target_id == node_id and edge.relation == RelationType.ATTACKS:
                if edge.source_id in self.nodes:
                    attackers.append(self.nodes[edge.source_id])
        return attackers
    
    def get_reasoning_chain(self, conclusion_node_id: str) -> List[List[ArgumentNode]]:
        chains = []
        from collections import deque
        
        queue = deque([([conclusion_node_id], conclusion_node_id)])
        visited = set()
        
        while queue:
            chain, current = queue.popleft()
            
            predecessors = []
            for edge in self.edges:
                if edge.target_id == current and edge.relation == RelationType.SUPPORTS:
                    predecessors.append(edge.source_id)
            
            if not predecessors:
                chains.append([self.nodes[nid] for nid in reversed(chain)])
            else:
                for pred in predecessors:
                    if pred not in visited:
                        visited.add(pred)
                        queue.append((chain + [pred], pred))
        
        return chains
    
    def to_dict(self) -> Dict:
        return {
            "nodes": {nid: node.to_dict() for nid, node in self.nodes.items()},
            "edges": [edge.to_dict() for edge in self.edges]
        }
    
    def save(self, filepath: str):
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)