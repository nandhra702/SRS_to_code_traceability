"""
Loaders for DocAgent and SRS Agent serialised outputs.

GraphLoader  — reads a DocAgent dependency-graph JSON file.
               Handles both the legacy flat format (top-level keys are
               component IDs) and the V1 format ({"components":{…},
               "relationships":[…]}).  Also derives docstring records,
               optionally recovering DocAgent-*generated* docstrings from the
               post-generation source tree (see read_post_run_docstrings).

SRSLoader    — reads an SRS Agent output JSON file: either the bare array of
               Requirement dicts the CLI writes, or the SRSOutput wrapper.

No DocAgent / SRS Agent imports; no database or embedding-model access.
"""

from __future__ import annotations

import ast
import hashlib
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

ComponentRecord = Dict[str, Any]     # matches CodeComponent.to_dict()
RelationshipRecord = Dict[str, str]  # {"source":…, "target":…, "type":…}
RequirementRecord = Dict[str, Any]   # matches Requirement.model_dump(mode="json")
DocstringRecord = Dict[str, Any]     # {"component_id", "kind", "content", "provenance"}

_GRAPH_SUFFIX = "_dependency_graph"


def content_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class GraphLoader:
    """
    Load a DocAgent dependency-graph JSON file.

    Legacy format:  { "<component_id>": { "id":…, "component_type":…, … }, … }
    V1 format:      { "components": { "<id>": {…}, … },
                      "relationships": [ {"source":…,"target":…,"type":…}, … ] }

    For legacy files, relationships are collected from any per-component
    "relationships" list; pre-V1 files have none, which is reported, not an error.
    """

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)
        self._components: Dict[str, ComponentRecord] = {}
        self._relationships: List[RelationshipRecord] = []
        self.format: Optional[str] = None
        self.malformed_relationships = 0
        self._loaded = False

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def load(self) -> "GraphLoader":
        """Parse the file. Returns self for chaining."""
        with self._path.open("r", encoding="utf-8") as fh:
            raw = json.load(fh)
        if not isinstance(raw, dict):
            raise ValueError(f"Graph must be a JSON object, got {type(raw).__name__}")
        self._parse(raw)
        self._loaded = True
        return self

    @property
    def components(self) -> Dict[str, ComponentRecord]:
        self._check()
        return self._components

    @property
    def relationships(self) -> List[RelationshipRecord]:
        """Flat, deduplicated list of well-formed typed relationships."""
        self._check()
        return self._relationships

    def default_repo_id(self) -> str:
        """Graph file stem without DocAgent's '_dependency_graph' suffix."""
        stem = self._path.stem
        return stem[: -len(_GRAPH_SUFFIX)] if stem.endswith(_GRAPH_SUFFIX) else stem

    def docstring_records(
        self, post_run_docstrings: Optional[Dict[str, str]] = None
    ) -> List[DocstringRecord]:
        """
        Docstring records with honest provenance.

        - 'existing': the docstring DocAgent saw when it parsed the repo
          (has_docstring=True).  This may itself come from an earlier DocAgent
          run — the graph cannot tell.
        - 'generated': only when *post_run_docstrings* (from
          read_post_run_docstrings) shows a non-empty docstring that differs
          from the parsed one, i.e. DocAgent wrote it after saving the graph.

        Without post_run_docstrings no 'generated' records are produced, since
        the graph JSON does not contain them.
        """
        self._check()
        post = post_run_docstrings or {}
        out: List[DocstringRecord] = []
        for cid, comp in self._components.items():
            parsed = comp.get("docstring") or ""
            if comp.get("has_docstring") and parsed.strip():
                out.append({
                    "component_id": cid,
                    "kind": "existing",
                    "content": parsed,
                    "provenance": {"source": "docagent_graph",
                                   "graph": self._path.name},
                })
            after = post.get(cid)
            if after and after.strip() and after != parsed:
                out.append({
                    "component_id": cid,
                    "kind": "generated",
                    "content": after,
                    "provenance": {"source": "post_run_source",
                                   "replaced_existing": bool(parsed.strip())},
                })
        return out

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _check(self) -> None:
        if not self._loaded:
            raise RuntimeError("Call .load() before accessing data.")

    def _parse(self, raw: Dict[str, Any]) -> None:
        if "components" in raw and isinstance(raw["components"], dict):
            self.format = "v1"
            comps = raw["components"]
            raw_rels = raw.get("relationships") or []
        else:
            self.format = "legacy"
            comps = raw
            raw_rels = [
                rel for v in raw.values() if isinstance(v, dict)
                for rel in (v.get("relationships") or [])
            ]

        self._components = {
            k: v for k, v in comps.items()
            if isinstance(v, dict) and "component_type" in v and v.get("id")
        }
        dropped = len(comps) - len(self._components)
        if dropped:
            logger.warning("Ignored %d malformed component record(s).", dropped)

        seen: set[Tuple[str, str, str]] = set()
        for rel in raw_rels:
            if not (isinstance(rel, dict)
                    and all(isinstance(rel.get(f), str) and rel.get(f)
                            for f in ("source", "target", "type"))):
                self.malformed_relationships += 1
                continue
            key = (rel["source"], rel["target"], rel["type"])
            if key not in seen:
                seen.add(key)
                self._relationships.append(
                    {"source": key[0], "target": key[1], "type": key[2]}
                )
        if self.malformed_relationships:
            logger.warning(
                "Ignored %d malformed relationship record(s).",
                self.malformed_relationships,
            )

    def __repr__(self) -> str:
        return (
            f"GraphLoader(path={self._path.name!r}, format={self.format}, "
            f"{len(self._components)} components, "
            f"{len(self._relationships)} relationships)"
        )


def read_post_run_docstrings(
    components: Dict[str, ComponentRecord], source_root: str | Path
) -> Dict[str, str]:
    """
    Read each component's *current* docstring from the source tree.

    DocAgent saves the graph before generating docstrings and then writes them
    into the source files in place, so the post-run tree is the only place
    generated docstrings exist.  Components are located by qualified name
    (module path + class/def names), not by line number — inserting docstrings
    shifts every line below them.

    Returns {component_id: raw docstring ('' when the node has none)} for
    components that could be located; unlocatable components are omitted.
    """
    root = Path(source_root)
    by_file: Dict[str, List[str]] = {}
    for cid, comp in components.items():
        by_file.setdefault(comp["relative_path"].replace("\\", "/"), []).append(cid)

    out: Dict[str, str] = {}
    for rel_path, cids in by_file.items():
        path = root / rel_path
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except (OSError, SyntaxError, UnicodeDecodeError) as exc:
            logger.warning("Cannot read %s for post-run docstrings: %s", path, exc)
            continue
        module = rel_path[:-3].replace("/", ".") if rel_path.endswith(".py") else rel_path
        for cid in cids:
            if not cid.startswith(module + "."):
                continue
            node = _find_qualified(tree, cid[len(module) + 1:].split("."))
            if node is not None:
                # clean=False: DocAgent stores the raw literal value
                out[cid] = ast.get_docstring(node, clean=False) or ""
    return out


def _find_qualified(tree: ast.Module, names: List[str]) -> Optional[ast.AST]:
    body = tree.body
    node = None
    for name in names:
        node = next(
            (n for n in body
             if isinstance(n, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
             and n.name == name),
            None,
        )
        if node is None:
            return None
        body = getattr(node, "body", [])
    return node


class SRSLoader:
    """
    Load an SRS Agent output JSON file.

    Accepts the bare array the CLI writes
    (``[req.model_dump(mode="json") for req in requirements]``) or the
    ``SRSOutput`` wrapper (``{"analysis_id", "source", "requirements"}``).
    """

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)
        self._requirements: List[RequirementRecord] = []
        self._source_meta: Dict[str, Any] = {}
        self._loaded = False

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def load(self) -> "SRSLoader":
        """Parse and validate the file. Returns self for chaining."""
        with self._path.open("r", encoding="utf-8") as fh:
            raw = json.load(fh)

        if isinstance(raw, dict) and isinstance(raw.get("requirements"), list):
            self._source_meta = raw.get("source") or {}
            raw = raw["requirements"]
        if not isinstance(raw, list):
            raise ValueError(
                "SRS output must be a JSON array of requirements or an SRSOutput "
                f"object, got {type(raw).__name__}"
            )

        seen: Dict[str, int] = {}
        for i, rec in enumerate(raw):
            if not isinstance(rec, dict):
                raise ValueError(f"Record at index {i} is not a JSON object.")
            rid = rec.get("requirement_id")
            if not isinstance(rid, str) or not rid.strip():
                raise ValueError(f"Record at index {i} has no 'requirement_id'.")
            if not (rec.get("raw_text") or "").strip():
                raise ValueError(f"Requirement {rid!r} (index {i}) has empty 'raw_text'.")
            if rid in seen:
                # The DB key is (document, requirement_id); a duplicate would
                # silently overwrite the first record.
                raise ValueError(
                    f"Duplicate requirement_id {rid!r} at indexes {seen[rid]} and {i}."
                )
            seen[rid] = i

        self._requirements = raw
        self._loaded = True
        return self

    @property
    def requirements(self) -> List[RequirementRecord]:
        self._check()
        return self._requirements

    def document_record(self, document_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Return a dict suitable for Store.upsert_document().

        document_id defaults to the SRSOutput's source.document_id, then the
        output file stem.  file_name/source_type come from the SRSOutput
        metadata or the requirements' source.file_name.
        """
        self._check()
        meta = self._source_meta
        file_name = meta.get("file_name")
        if not file_name:
            file_name = next(
                ((r.get("source") or {}).get("file_name") for r in self._requirements
                 if (r.get("source") or {}).get("file_name")),
                None,
            )
        source_type = meta.get("source_type") or _source_type_from_name(file_name)
        return {
            "document_id":        document_id or meta.get("document_id") or self._path.stem,
            "file_name":          file_name or self._path.name,
            "source_type":        source_type,
            "total_requirements": len(self._requirements),
            "ingest_path":        str(self._path.resolve()),
        }

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


def _source_type_from_name(file_name: Optional[str]) -> str:
    if not file_name:
        return "unknown"
    lower = file_name.lower()
    if lower.endswith(".pdf"):
        return "pdf"
    if lower.endswith(".docx"):
        return "docx"
    return "text"
