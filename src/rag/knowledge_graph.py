"""Enhanced Knowledge Graph construction with entity linking and multi-hop reasoning.

Features:
- Extract triples from ALL documents (not just samples)
- Entity linking to dental ontologies
- Multi-hop path finding for complex queries
- Relationship inference
"""
from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import networkx as nx
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage
from pydantic import BaseModel, Field

from src.utils.llm import get_gemini_chat

logger = logging.getLogger(__name__)


class Entity(BaseModel):
    """Knowledge graph entity with ontology linking."""

    name: str = Field(description="Entity name (normalized)")
    entity_type: str = Field(description="Type: condition, treatment, symptom, anatomy")
    aliases: List[str] = Field(default_factory=list, description="Alternative names")
    ontology_ids: Dict[str, str] = Field(
        default_factory=dict,
        description="Ontology IDs (e.g., ICD-10, SNOMED CT)"
    )


class Triple(BaseModel):
    """Knowledge graph triple (subject, predicate, object)."""

    subject: str
    predicate: str
    object: str
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    source_doc_id: Optional[str] = None


class KnowledgeGraphBuilder:
    """Builds and maintains knowledge graph from documents."""

    TRIPLE_EXTRACTION_PROMPT = """Extract medical knowledge triples from dental/oral health text.

**Text:** {text}

**Task:**
Extract ALL relevant triples in format: (subject, predicate, object)

**Triple Types:**
1. **Causes**: (condition, causes, symptom) - e.g., (caries, causes, tooth pain)
2. **Treats**: (treatment, treats, condition) - e.g., (filling, treats, cavity)
3. **Prevents**: (action, prevents, condition) - e.g., (fluoride, prevents, caries)
4. **Is-A**: (specific, is_a, general) - e.g., (gingivitis, is_a, gum disease)
5. **Located-In**: (condition, located_in, anatomy) - e.g., (caries, located_in, tooth)
6. **Symptom-Of**: (symptom, symptom_of, condition) - e.g., (pain, symptom_of, pulpitis)

**Guidelines:**
- Use normalized medical terms (lowercase)
- Focus on causal, treatment, prevention relationships
- Include both Indonesian and English terms
- Be specific about anatomical locations

**Output JSON:**
{{
  "triples": [
    {{"subject": "caries", "predicate": "causes", "object": "tooth pain", "confidence": 1.0}},
    {{"subject": "fluoride", "predicate": "prevents", "object": "caries", "confidence": 0.9}}
  ]
}}

Respond with ONLY valid JSON."""

    # Simple dental ontology mapping (subset of ICD-10 / SNOMED CT)
    DENTAL_ONTOLOGY = {
        "caries": {"icd10": "K02", "snomed": "80967001"},
        "dental caries": {"icd10": "K02", "snomed": "80967001"},
        "gingivitis": {"icd10": "K05.0", "snomed": "66383009"},
        "periodontitis": {"icd10": "K05.3", "snomed": "41565005"},
        "pulpitis": {"icd10": "K04.0", "snomed": "32620007"},
        "tooth abscess": {"icd10": "K04.7", "snomed": "399939004"},
        "toothache": {"icd10": "K08.8", "snomed": "27355003"},
        "dental calculus": {"icd10": "K03.6", "snomed": "109564008"},
        "oral ulcer": {"icd10": "K12.0", "snomed": "397156002"},
    }

    def __init__(self):
        """Initialize KG builder."""
        self.llm = get_gemini_chat(model="gemini-2.5-flash", temperature=0.0)
        self.graph = nx.DiGraph()
        self.entities: Dict[str, Entity] = {}

        logger.info("KnowledgeGraphBuilder: Initialized")

    def build_from_documents(
        self,
        documents: List[Document],
        persist_path: Optional[str] = None,
    ) -> nx.DiGraph:
        """Build knowledge graph from ALL documents.

        Args:
            documents: List of documents to process
            persist_path: Optional path to save graph

        Returns:
            NetworkX DiGraph
        """
        logger.info(f"KnowledgeGraphBuilder: Building KG from {len(documents)} documents")

        all_triples: List[Triple] = []

        # Extract from each document
        for idx, doc in enumerate(documents):
            try:
                triples = self._extract_triples(doc.page_content, doc_id=f"doc_{idx}")
                all_triples.extend(triples)

                if (idx + 1) % 10 == 0:
                    logger.info(f"KnowledgeGraphBuilder: Processed {idx + 1}/{len(documents)} docs")

            except Exception as e:
                logger.error(f"KnowledgeGraphBuilder: Failed to extract from doc {idx} - {e}")

        logger.info(f"KnowledgeGraphBuilder: Extracted {len(all_triples)} total triples")

        # Build graph
        self._build_graph(all_triples)

        # Link entities to ontologies
        self._link_entities()

        logger.info(
            f"KnowledgeGraphBuilder: Graph built - "
            f"{self.graph.number_of_nodes()} nodes, "
            f"{self.graph.number_of_edges()} edges"
        )

        # Persist if requested
        if persist_path:
            self.save(persist_path)

        return self.graph

    def _extract_triples(self, text: str, doc_id: str) -> List[Triple]:
        """Extract triples from text chunk.

        Args:
            text: Text to process
            doc_id: Source document ID

        Returns:
            List of Triple objects
        """
        try:
            # Limit text length for LLM
            text_chunk = text[:1000]

            prompt = self.TRIPLE_EXTRACTION_PROMPT.format(text=text_chunk)

            response = self.llm.invoke(
                [HumanMessage(content=prompt)],
                config={"response_format": {"type": "json_object"}},
            )

            import json
            import re

            raw = response.content.strip()
            json_match = re.search(r"\{.*\}", raw, re.DOTALL)
            json_str = json_match.group(0) if json_match else raw

            result = json.loads(json_str)
            triples_data = result.get("triples", [])

            triples = [
                Triple(
                    subject=t["subject"].lower().strip(),
                    predicate=t["predicate"].lower().strip(),
                    object=t["object"].lower().strip(),
                    confidence=t.get("confidence", 1.0),
                    source_doc_id=doc_id,
                )
                for t in triples_data
            ]

            return triples

        except Exception as e:
            logger.warning(f"KnowledgeGraphBuilder: Triple extraction error - {e}")
            return []

    def _build_graph(self, triples: List[Triple]) -> None:
        """Build NetworkX graph from triples.

        Args:
            triples: List of Triple objects
        """
        for triple in triples:
            # Add nodes
            self.graph.add_node(triple.subject, type="entity")
            self.graph.add_node(triple.object, type="entity")

            # Add edge with attributes
            self.graph.add_edge(
                triple.subject,
                triple.object,
                relation=triple.predicate,
                confidence=triple.confidence,
                source=triple.source_doc_id,
            )

            # Track entities
            if triple.subject not in self.entities:
                self.entities[triple.subject] = Entity(
                    name=triple.subject,
                    entity_type="unknown"
                )
            if triple.object not in self.entities:
                self.entities[triple.object] = Entity(
                    name=triple.object,
                    entity_type="unknown"
                )

    def _link_entities(self) -> None:
        """Link entities to dental ontologies."""
        for entity_name, entity in self.entities.items():
            # Check if in ontology
            if entity_name in self.DENTAL_ONTOLOGY:
                entity.ontology_ids = self.DENTAL_ONTOLOGY[entity_name]
                logger.debug(f"KnowledgeGraphBuilder: Linked '{entity_name}' to {entity.ontology_ids}")

    def query(self, query_text: str, max_hops: int = 3) -> Dict[str, Any]:
        """Query knowledge graph with multi-hop reasoning.

        Args:
            query_text: Query text
            max_hops: Maximum path length to search

        Returns:
            Dict with paths, related entities, insights
        """
        logger.info(f"KnowledgeGraphBuilder: Querying '{query_text}'")

        # Extract key terms from query
        query_terms = self._extract_query_terms(query_text)

        # Find matching entities
        matching_nodes = self._find_matching_nodes(query_terms)

        if not matching_nodes:
            return {
                "paths": [],
                "related_entities": [],
                "insights": "No matching entities found in knowledge graph"
            }

        # Find paths between entities
        paths = self._find_multi_hop_paths(matching_nodes, max_hops)

        # Find related entities (neighbors)
        related = self._find_related_entities(matching_nodes)

        # Generate insights
        insights = self._generate_insights(matching_nodes, paths, related)

        logger.info(f"KnowledgeGraphBuilder: Found {len(paths)} paths, {len(related)} related entities")

        return {
            "paths": paths,
            "related_entities": related,
            "insights": insights
        }

    def _extract_query_terms(self, query_text: str) -> List[str]:
        """Extract key medical terms from query.

        Args:
            query_text: Query text

        Returns:
            List of normalized terms
        """
        # Simple approach: split and normalize
        import re
        words = re.findall(r'\w+', query_text.lower())

        # Filter common words
        stop_words = {"apa", "adalah", "yang", "dan", "atau", "untuk", "dari", "ke", "di", "the", "is", "a", "an", "of", "to", "in"}
        terms = [w for w in words if w not in stop_words and len(w) > 3]

        return terms

    def _find_matching_nodes(self, terms: List[str]) -> List[str]:
        """Find graph nodes matching query terms.

        Args:
            terms: Query terms

        Returns:
            List of matching node names
        """
        matching = []

        for node in self.graph.nodes():
            for term in terms:
                if term in node or node in term:
                    matching.append(node)
                    break

        return matching

    def _find_multi_hop_paths(
        self,
        nodes: List[str],
        max_hops: int
    ) -> List[Dict[str, Any]]:
        """Find paths between nodes (multi-hop reasoning).

        Args:
            nodes: Starting nodes
            max_hops: Maximum path length

        Returns:
            List of path dicts
        """
        paths = []

        for i, start in enumerate(nodes):
            for end in nodes[i+1:]:
                try:
                    # Find shortest path
                    if nx.has_path(self.graph, start, end):
                        path = nx.shortest_path(self.graph, start, end)

                        if len(path) <= max_hops + 1:
                            # Extract relations
                            relations = []
                            for j in range(len(path) - 1):
                                edge_data = self.graph.get_edge_data(path[j], path[j+1])
                                relations.append(edge_data.get("relation", "related_to"))

                            paths.append({
                                "path": path,
                                "relations": relations,
                                "length": len(path) - 1
                            })

                except nx.NetworkXNoPath:
                    continue

        return paths[:10]  # Limit to top 10

    def _find_related_entities(self, nodes: List[str]) -> List[Dict[str, str]]:
        """Find entities related to query nodes.

        Args:
            nodes: Query nodes

        Returns:
            List of related entity dicts
        """
        related = []
        seen = set(nodes)

        for node in nodes:
            # Get neighbors
            neighbors = list(self.graph.neighbors(node))

            for neighbor in neighbors:
                if neighbor not in seen:
                    edge_data = self.graph.get_edge_data(node, neighbor)
                    related.append({
                        "entity": neighbor,
                        "relation": edge_data.get("relation", "related_to"),
                        "from": node
                    })
                    seen.add(neighbor)

        return related[:20]  # Limit

    def _generate_insights(
        self,
        nodes: List[str],
        paths: List[Dict],
        related: List[Dict]
    ) -> str:
        """Generate textual insights from graph query.

        Args:
            nodes: Matched nodes
            paths: Found paths
            related: Related entities

        Returns:
            Insight text
        """
        insights = []

        if nodes:
            insights.append(f"Found entities: {', '.join(nodes[:5])}")

        if paths:
            for p in paths[:3]:
                path_str = " → ".join(
                    f"{p['path'][i]} [{p['relations'][i]}]"
                    for i in range(len(p['relations']))
                ) + f" → {p['path'][-1]}"
                insights.append(f"Relationship: {path_str}")

        if related:
            rel_summary = []
            for r in related[:5]:
                rel_summary.append(f"{r['from']} {r['relation']} {r['entity']}")
            insights.append("Related: " + "; ".join(rel_summary))

        return "\n".join(insights) if insights else "No significant insights found"

    def save(self, path: str) -> None:
        """Save graph to disk.

        Args:
            path: File path to save
        """
        try:
            Path(path).parent.mkdir(parents=True, exist_ok=True)

            with open(path, "wb") as f:
                pickle.dump({
                    "graph": self.graph,
                    "entities": self.entities
                }, f)

            logger.info(f"KnowledgeGraphBuilder: Saved to {path}")

        except Exception as e:
            logger.error(f"KnowledgeGraphBuilder: Save failed - {e}")

    @classmethod
    def load(cls, path: str) -> KnowledgeGraphBuilder:
        """Load graph from disk.

        Args:
            path: File path to load

        Returns:
            Loaded KnowledgeGraphBuilder
        """
        try:
            with open(path, "rb") as f:
                data = pickle.load(f)

            builder = cls()
            builder.graph = data["graph"]
            builder.entities = data.get("entities", {})

            logger.info(
                f"KnowledgeGraphBuilder: Loaded from {path} - "
                f"{builder.graph.number_of_nodes()} nodes, "
                f"{builder.graph.number_of_edges()} edges"
            )

            return builder

        except Exception as e:
            logger.error(f"KnowledgeGraphBuilder: Load failed - {e}")
            return cls()  # Return empty builder