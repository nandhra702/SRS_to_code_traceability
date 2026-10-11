"""
Loaders for DocAgent and SRS Agent serialised outputs.

GraphLoader  — reads a DocAgent dependency-graph JSON file.
               Handles both the legacy flat format (pre-V1, top-level keys are
               component IDs) and the V1 format ({"components":{…}, "relationships":[…]}).
               No DocAgent imports required.

SRSLoader    — reads an SRS Agent output JSON file (array of Requirement dicts).
               No SRS Agent imports required.

Neither loader touches the database or the embedding model.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple


ComponentRecord  = Dict[str, Any]   # matches CodeComponent.to_dict()
RelationshipRecord = Dict[str, str] # {"source":…, "target":…, "type":…}
RequirementRecord  = Dict[str, Any] # matches Requirement.model_dump(mode="json")


class GraphLoader:
    """
    Load a DocAgent dependency-graph JSON file.

    Supports two on-disk formats:

    Legacy (pre-V1):
        { "<component_id>": { "id":…, "component_type":…, … }, … }

    V1:
        { "components": { "<id>": {…}, … },
          "relationships": [ {"source":…,"target":…,"type":…}, … ] }

    In the legacy format, typed relationships are collected from the
    per-component "relationships" list that the V1 parser writes inline.
    Components that lack the "relationships" key are handled gracefully.
    """

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)
        self._components:    Dict[str, ComponentRecord]   = {}
        self._relationships: List[RelationshipRecord]     = []
        self._loaded = False

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def load(self) -> "GraphLoader":
        """Parse the file. Returns self for chaining."""
        with self._path.open("r", encoding="utf-8") as fh:
            raw = json.load(fh)
        self._parse(raw)
        self._loaded = True
        return self

    @property
    def components(self) -> Dict[str, ComponentRecord]:
        self._check()
        return self._components

    @property
    def relationships(self) -> List[RelationshipRecord]:
        """Flat, deduplicated list of typed relationships."""
        self._check()
        return self._relationships

    def as_embedding_texts(self) -> Dict[str, str]:
        """
        Return {component_id: text_for_embedding}.

        Priority: signature > docstring > source_code[:2000].
        Components with none of these get an empty string.
        """
        self._check()
        out: Dict[str, str] = {}
        for cid, comp in self._components.items():
            parts: List[str] = []
            if comp.get("signature"):
                parts.append(comp["signature"])
            if comp.get("docstring"):
                parts.append(comp["docstring"])
            sc = comp.get("source_code") or ""
            if sc:
                parts.append(sc[:2000])
            out[cid] = "\n\n".join(parts)
        return out

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _check(self) -> None:
        if not self._loaded:
            raise RuntimeError("Call .load() before accessing data.")

    def _parse(self, raw: Dict[str, Any]) -> None:
        if "components" in raw and isinstance(raw["components"], dict):
            # V1 format
            self._components = raw["components"]
            raw_rels: List[RelationshipRecord] = raw.get("relationships", [])
        else:
            # Legacy flat format — every top-level entry with "component_type" is a component
            self._components = {
                k: v for k, v in raw.items()
                if isinstance(v, dict) and "component_type" in v
            }
            raw_rels = []
            for comp in self._components.values():
                for rel in comp.get("relationships", []):
                    raw_rels.append(rel)

        # Deduplicate by (source, target, type) triple
        seen: set[Tuple[str, str, str]] = set()
        for rel in raw_rels:
            key = (rel["source"], rel["target"], rel["type"])
            if key not in seen:
                seen.add(key)
                self._relationships.append(rel)

    def __repr__(self) -> str:
        return (
            f"GraphLoader(path={self._path.name!r}, "
            f"{len(self._components)} components, "
            f"{len(self._relationships)} relationships)"
        )


class SRSLoader:
    """
    Load an SRS Agent output JSON file.

    The file contains a JSON array produced by:
        json.dumps([req.model_dump(mode="json") for req in requirements])

    The loader also extracts document-level metadata from the array so the
    caller can pass it to Store.upsert_document() without re-reading the file.
    """

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)
        self._requirements: List[RequirementRecord] = []
        self._loaded = False

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def load(self) -> "SRSLoader":
        """Parse the file. Returns self for chaining."""
        with self._path.open("r", encoding="utf-8") as fh:
            raw = json.load(fh)

        if not isinstance(raw, list):
            raise ValueError(
                f"SRS output must be a JSON array, got {type(raw).__name__}"
            )
        for i, rec in enumerate(raw):
            if not isinstance(rec, dict):
                raise ValueError(f"Record at index {i} is not a JSON object.")
            if "requirement_id" not in rec:
                raise ValueError(
                    f"Record at index {i} is missing 'requirement_id'."
                )

        self._requirements = raw
        self._loaded = True
        return self

    @property
    def requirements(self) -> List[RequirementRecord]:
        self._check()
        return self._requirements

    def document_record(self) -> Dict[str, Any]:
        """
        Return a dict suitable for Store.upsert_document().

        Infers document metadata from the first requirement that has a
        populated source.file_name, falling back to the file stem.
        """
        self._check()
        file_name = self._path.name
        source_type = "unknown"

        # Try to infer file_name and source_type from requirement source metadata
        for req in self._requirements:
            src = req.get("source") or {}
            if src.get("file_name"):
                file_name = src["file_name"]
                # Infer type from extension
                lower = file_name.lower()
                if lower.endswith(".pdf"):
                    source_type = "pdf"
                elif lower.endswith(".docx"):
                    source_type = "docx"
                else:
                    source_type = "text"
                break

        return {
            "document_id":         self._path.stem,
            "file_name":           file_name,
            "source_type":         source_type,
            "total_requirements":  len(self._requirements),
            "ingest_path":         str(self._path.resolve()),
        }

    def as_embedding_texts(self) -> Dict[str, str]:
        """
        Return {requirement_id: text_for_embedding}.

        Concatenates: retrieval_text · title · normalized_text ·
                      domain_terms · synonyms
        """
        self._check()
        out: Dict[str, str] = {}
        for req in self._requirements:
            rid = req["requirement_id"]
            parts: List[str] = []
            for field in ("retrieval_text", "title", "normalized_text"):
                val = req.get(field, "")
                if val:
                    parts.append(val)
            for field in ("domain_terms", "synonyms"):
                lst = req.get(field) or []
                if lst:
                    parts.append(" ".join(lst))
            out[rid] = " ".join(parts)
        return out

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _check(self) -> None:
        if not self._loaded:
            raise RuntimeError("Call .load() before accessing data.")

    def __repr__(self) -> str:
        return (
            f"SRSLoader(path={self._path.name!r}, "
            f"{len(self._requirements)} requirements)"
        )
