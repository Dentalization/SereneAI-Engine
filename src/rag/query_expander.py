"""Query expansion for medical/dental domain using LLM.

Expands queries with:
- Medical synonyms (e.g., "gigi berlubang" → "caries", "dental cavity")
- Related terms (e.g., "sakit gigi" → "toothache", "dental pain", "odontogenic pain")
- Multilingual equivalents (Indonesian ↔ English medical terms)
"""
from __future__ import annotations

import json
import logging
import re
from typing import List

from langchain_core.messages import HumanMessage
from pydantic import BaseModel, Field

from src.utils.llm import get_gemini_chat

logger = logging.getLogger(__name__)


class ExpandedQuery(BaseModel):
    """Expanded query with synonyms and related terms."""

    original_query: str
    synonyms: List[str] = Field(default_factory=list, description="Direct synonyms")
    related_terms: List[str] = Field(default_factory=list, description="Related medical terms")
    multilingual: List[str] = Field(default_factory=list, description="Translations (ID↔EN)")
    expanded_query: str = Field(description="Combined query for retrieval")


class QueryExpander:
    """Expands queries with medical synonyms and related terms."""

    EXPANSION_PROMPT = """You are a dental/medical terminology expert. Expand the user's query with relevant medical synonyms and related terms.

**Original Query:** {query}

**Your Task:**
1. Identify key dental/medical concepts
2. Provide direct synonyms (same meaning)
3. Provide related terms (broader/narrower concepts)
4. Translate between Indonesian and English medical terms

**Guidelines:**
- Focus on dental and oral health terminology
- Include both lay terms and medical jargon
- Preserve original language + add translations
- For Indonesian queries, add English medical equivalents
- For English queries, add Indonesian equivalents

**Examples:**

Query: "gigi berlubang"
Synonyms: ["caries", "dental cavity", "tooth decay", "karies gigi"]
Related: ["bacterial infection", "enamel demineralization", "dental caries"]
Multilingual: ["dental caries", "tooth cavity", "cavity"]

Query: "sakit gigi"
Synonyms: ["toothache", "dental pain", "tooth pain", "odontogenic pain"]
Related: ["pulpitis", "abscess", "gingivitis", "periapical infection"]
Multilingual: ["toothache", "dental pain", "odontalgia"]

Query: "karang gigi"
Synonyms: ["dental calculus", "tartar", "calcified plaque"]
Related: ["plaque", "periodontal disease", "gingivitis", "dental hygiene"]
Multilingual: ["dental calculus", "tartar buildup"]

**Output JSON Schema:**
{{
  "original_query": "string",
  "synonyms": ["direct synonyms"],
  "related_terms": ["related medical concepts"],
  "multilingual": ["translations"],
  "expanded_query": "original + synonyms + related terms combined"
}}

**Important:**
- If query is general (e.g., "halo", "terima kasih"), return minimal expansion
- Focus on nouns/medical concepts, not common words
- Limit to 3-5 synonyms and 3-5 related terms

Respond with ONLY valid JSON."""

    def __init__(self, cache_size: int = 100):
        """Initialize query expander.

        Args:
            cache_size: Number of expanded queries to cache
        """
        self.llm = get_gemini_chat(
            model="gemini-2.5-flash",
            temperature=0.1,  # Low temp for consistent expansions
        )
        self.cache: dict[str, ExpandedQuery] = {}
        self.cache_size = cache_size

        logger.info(f"QueryExpander: Initialized with cache size {cache_size}")

    def expand(self, query: str) -> ExpandedQuery:
        """Expand query with medical synonyms and related terms.

        Args:
            query: Original user query

        Returns:
            ExpandedQuery with synonyms, related terms, and combined query
        """
        # Check cache
        if query in self.cache:
            logger.debug(f"QueryExpander: Cache hit for '{query}'")
            return self.cache[query]

        logger.info(f"QueryExpander: Expanding query '{query[:50]}...'")

        try:
            # Format prompt
            prompt = self.EXPANSION_PROMPT.format(query=query)

            # Invoke LLM
            response = self.llm.invoke(
                [HumanMessage(content=prompt)],
                config={"response_format": {"type": "json_object"}},
            )

            raw_response = response.content.strip()
            logger.debug(f"QueryExpander: Raw response: {raw_response[:200]}")

            # Parse JSON
            json_match = re.search(r"\{.*\}", raw_response, re.DOTALL)
            json_str = json_match.group(0) if json_match else raw_response

            expansion_dict = json.loads(json_str)
            expanded_query = ExpandedQuery(**expansion_dict)

            logger.info(
                f"QueryExpander: Expanded to {len(expanded_query.synonyms)} synonyms, "
                f"{len(expanded_query.related_terms)} related terms"
            )
            logger.debug(f"QueryExpander: Synonyms: {expanded_query.synonyms}")
            logger.debug(f"QueryExpander: Related: {expanded_query.related_terms}")

            # Cache result
            self._cache_expansion(query, expanded_query)

            return expanded_query

        except json.JSONDecodeError as e:
            logger.error(f"QueryExpander: JSON parse error - {e}")
            return self._fallback_expansion(query)

        except Exception as e:
            logger.error(f"QueryExpander: Expansion failed - {e}")
            return self._fallback_expansion(query)

    def _fallback_expansion(self, query: str) -> ExpandedQuery:
        """Fallback expansion using simple heuristics.

        Args:
            query: Original query

        Returns:
            Basic ExpandedQuery with minimal expansion
        """
        logger.warning(f"QueryExpander: Using fallback for '{query}'")

        # Simple keyword-based expansion
        synonyms = []
        related_terms = []
        multilingual = []

        query_lower = query.lower()

        # Common dental terms mapping
        term_map = {
            "gigi berlubang": {
                "synonyms": ["caries", "karies", "cavity"],
                "related": ["tooth decay", "enamel damage"],
                "multilingual": ["dental caries", "tooth cavity"],
            },
            "sakit gigi": {
                "synonyms": ["toothache", "tooth pain"],
                "related": ["pulpitis", "dental abscess"],
                "multilingual": ["dental pain", "odontalgia"],
            },
            "karang gigi": {
                "synonyms": ["calculus", "tartar"],
                "related": ["plaque", "periodontal disease"],
                "multilingual": ["dental calculus", "tartar buildup"],
            },
            "radang gusi": {
                "synonyms": ["gingivitis", "gum inflammation"],
                "related": ["periodontal disease", "gum disease"],
                "multilingual": ["gingivitis", "gum infection"],
            },
            "sariawan": {
                "synonyms": ["ulcer", "canker sore", "aphthous ulcer"],
                "related": ["oral lesion", "mouth sore"],
                "multilingual": ["oral ulcer", "mouth ulcer"],
            },
        }

        # Check for matches
        for term, expansions in term_map.items():
            if term in query_lower:
                synonyms.extend(expansions.get("synonyms", []))
                related_terms.extend(expansions.get("related", []))
                multilingual.extend(expansions.get("multilingual", []))

        # If no matches, minimal expansion
        if not synonyms:
            expanded_query_str = query
        else:
            # Combine all terms
            all_terms = [query] + synonyms + related_terms[:3]
            expanded_query_str = " OR ".join(all_terms)

        return ExpandedQuery(
            original_query=query,
            synonyms=synonyms[:5],
            related_terms=related_terms[:5],
            multilingual=multilingual[:5],
            expanded_query=expanded_query_str,
        )

    def _cache_expansion(self, query: str, expansion: ExpandedQuery) -> None:
        """Cache expansion result.

        Args:
            query: Original query
            expansion: Expanded query result
        """
        # Limit cache size (FIFO)
        if len(self.cache) >= self.cache_size:
            # Remove oldest entry
            oldest_key = next(iter(self.cache))
            del self.cache[oldest_key]
            logger.debug(f"QueryExpander: Cache evicted '{oldest_key}'")

        self.cache[query] = expansion

    def get_expanded_query_string(self, query: str) -> str:
        """Convenience method to get just the expanded query string.

        Args:
            query: Original query

        Returns:
            Expanded query string for retrieval
        """
        expansion = self.expand(query)
        return expansion.expanded_query

    def clear_cache(self) -> None:
        """Clear expansion cache."""
        self.cache.clear()
        logger.info("QueryExpander: Cache cleared")