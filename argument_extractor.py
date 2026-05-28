# argument_extractor.py
import re
import json
from typing import List, Tuple
from openai import OpenAI

from argumentation_graph import NodeType, RelationType


class ArgumentExtractor:
    
    def __init__(self, client: OpenAI, model_name: str):
        self.client = client
        self.model_name = model_name
    
    def extract_claims(self, response_text: str, agent_id: int, round_num: int) -> List[Tuple[str, NodeType, float]]:
        claims = []
        
        claim_patterns = [
            (r"(?:I think|my point is|my conclusion is)[：:]\s*(.+?)[。\n]", NodeType.CLAIM),
            (r"(?:therefore|so|hence)[，,]?\s*(.+?)[。\n]", NodeType.CONCLUSION),
            (r"(?:assume|assuming that)\s*(.+?)[。\n]", NodeType.ASSUMPTION),
            (r"(?:evidence|according to)[^:]*[：:]\s*(.+?)[。\n]", NodeType.EVIDENCE),
            (r"(?:key point is)[：:]\s*(.+?)[。\n]", NodeType.QUESTION),
        ]
        
        for pattern, node_type in claim_patterns:
            matches = re.findall(pattern, response_text, re.IGNORECASE)
            for match in matches:
                claim_text = match.strip()[:200]
                if claim_text:
                    claims.append((claim_text, node_type, 0.7))
        
        if not claims:
            sentences = re.split(r'[。\n]', response_text)
            for sent in sentences[:3]:
                if len(sent) > 20 and len(sent) < 200:
                    claims.append((sent.strip(), NodeType.CLAIM, 0.5))
        
        return claims
    
    def identify_relations(self, new_claim: str, existing_claims: List[str], 
                          context: str = "") -> List[Tuple[int, RelationType, float]]:
        relations = []
        
        for idx, existing in enumerate(existing_claims):
            support_keywords = ["agree", "support", "correct", "yes", "also"]
            attack_keywords = ["disagree", "wrong", "but", "however", "although", 
                              "refute", "question", "the problem is"]
            
            new_lower = new_claim.lower()
            exist_lower = existing.lower()
            
            if any(kw in new_lower for kw in support_keywords) and exist_lower in new_lower:
                relations.append((idx, RelationType.SUPPORTS, 0.8))
            
            elif any(kw in new_lower for kw in attack_keywords) and (exist_lower in new_lower or 
                  any(word in new_lower for word in exist_lower.split()[:3])):
                relations.append((idx, RelationType.ATTACKS, 0.7))
        
        return relations
    
    async def extract_with_llm(self, response_text: str, agent_id: int, 
                               round_num: int, previous_claims: List[str]) -> dict:
        prompt = f"""
        Please analyze the following agent response, extract the core arguments, and determine relationships with previous arguments.
        
        Current agent response:
        {response_text}
        
        Previous arguments:
        {chr(10).join([f"{i+1}. {c}" for i, c in enumerate(previous_claims)])}
        
        Output in JSON format:
        {{
            "claims": [
                {{"content": "argument content", "type": "claim/evidence/assumption/conclusion", "confidence": 0.8}}
            ],
            "relations": [
                {{"target_index": 0, "relation": "supports/attacks/cites", "strength": 0.7}}
            ]
        }}
        """
        
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3
            )
            result = json.loads(response.choices[0].message.content)
            return result
        except:
            return {"claims": [], "relations": []}