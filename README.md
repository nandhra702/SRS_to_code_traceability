# traceability

This package is the integration layer between **SRS Agent** (requirements) and **DocAgent** (code components, docstrings and typed relationships). It loads both tools' JSON outputs into PostgreSQL with pgvector. Later phases use that database for retrieval, requirement↔docstring↔code mapping and verification.

It only consumes the outputs of those tools and never imports either one.

Status: **Phase 1 (schema and ingestion) and Phase 2 (query resolution and retrieval) are complete.** Mapping (Phase 3) and verification (Phase 4) are not built yet. The trace-link and verification tables already exist so those phases can write to them.

## Setup

```powershell
cd SRS_to_code_traceability
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip install -e .   # importable as `traceability`
```

You need **PostgreSQL with the pgvector extension**. This was tested on PostgreSQL 18.3 with pgvector 0.8.1. You also need two databases:
- a development database, `traceability_dev`
- a disposable test database, `traceability_test`

The embedding model, `sentence-transformers/all-MiniLM-L6-v2` (384 dimensions), loads from the **local Hugging Face cache only**. Set `EMBED_ALLOW_DOWNLOAD=1` once to fetch it on a machine that doesn't have it cached.

### Environment variables

| Variable | Default | Purpose |
|---|---|---|
| `TRACEABILITY_PG_DSN` | — | Full libpq DSN (overrides the variables below) |
| `TRACEABILITY_PG_HOST` / `_PORT` / `_DBNAME` / `_USER` | `localhost` / `5432` / `traceability_dev` / `postgres` | Connection |
| `TRACEABILITY_PG_PASSWORD` | empty | Password (environment only; never logged) |
| `EMBED_MODEL` / `EMBED_MODEL_VERSION` | MiniLM-L6-v2 / same as model | Model and the version tag stored with each vector |
| `EMBED_DIM` | from the model | Only needed for `--no-embed` runs on an empty database |
| `EMBED_CODE_MAX_CHARS` | `1500` | Source characters included in a code embedding |
| `EMBED_ALLOW_DOWNLOAD` | `0` | Allow the model to be downloaded |
| `TEST_PG_DBNAME` / `TEST_PG_USER` / `TEST_PG_PASSWORD` | `traceability_test` / `postgres` / empty | Test suite database |

## Commands

Run these from any directory once the package is installed. Each ingest command prints a JSON report to stdout. Errors exit with code 2 and a one-line explanation.

```powershell
$py = ".\SRS_to_code_traceability\.venv\Scripts\python.exe"

# Create the extension, tables and indexes (safe to repeat)
& $py -m traceability.cli init-schema

# SRS Agent output: a JSON array, or the SRSOutput wrapper
& $py -m traceability.cli ingest-requirements SRSAgent\output\Basic_SRS_Document_requirements.txt

# DocAgent graph. --source-root is the repository *after* DocAgent wrote its
# docstrings; it is how generated docstrings are recovered (see below).
& $py -m traceability.cli ingest-graph DocAgent\output\dependency_graphs\llama3_1_8b_dependency_graph.json `
    --repo-id "raw_test_repo@llama3.1-8b" --repo-snapshot 06f13f6 `
    --source-root DocAgent\benchmark\results\raw_generations\llama3.1-8b

# Re-embed rows that are missing a vector, came from another model, or whose text changed
& $py -m traceability.cli rebuild-embeddings

# Optional HNSW indexes (exact search is used until the data justifies them)
& $py -m traceability.cli build-vector-indexes

# Search: a requirement ID, full requirement text, or keywords
& $py -m traceability.cli search FR-018 --document-id library-srs-v1.1 --repo-id "library@yi-coder-9b"
& $py -m traceability.cli search "late return penalty" --repo-id "library@yi-coder-9b" --top-k 5
& $py -m traceability.cli search "overdue fine" --method lexical --entities component --json

# Score retrieval against a gold-standard mapping (format v2)
& $py -m traceability.cli evaluate --gold library_traceability_dataset\gold_standard\gold_standard_mapping.json `
    --repo-id "library@yi-coder-9b" --out eval_gold_text.json
& $py -m traceability.cli evaluate --gold library_traceability_dataset\gold_standard\gold_standard_mapping.json `
    --repo-id "library@yi-coder-9b" --document-id library-srs-v1.1 --query-source ingested
```

## Retrieval (Phase 2)

**Query resolution** (`query.py`). `search` decides what you typed:

| Input | Treated as |
|---|---|
| `FR-018` | A stored requirement, resolved within an SRS document. If the ID exists in several documents and no `--document-id` is given, the result is `ambiguous`; it never picks one. |
| A sentence of 6+ words | The stored requirement whose text matches exactly one. Otherwise an ad-hoc requirement-like query, with the exact text kept. |
| Shorter input | Keywords. Never creates or implies a formal requirement. |

You can force a mode with `--mode id|text|adhoc|keyword`.

**Channels** (`retrieval.py`). There are six, one per entity × method: {requirement, docstring, component} × {lexical, vector}.
- **Lexical** uses PostgreSQL full-text search. The query becomes an OR of its English-stemmed words, with identifiers split the same way as at index time (`calculate_fine` → *calcul fine*, `LendingService` → *lend servic*). Hits are ranked with `ts_rank`.
- **Vector** uses pgvector cosine similarity. It only compares rows embedded by the configured model version. A stored requirement reuses its own vector.

**Fusion.** Lexical and vector scores aren't comparable, so `hybrid` combines them with Reciprocal Rank Fusion (`Σ 1/(60 + rank)`). Every hit keeps each channel's rank and raw score.

**Implementation candidates.** The `components` list fuses code hits and docstring hits; a docstring hit counts for its owning component. This is the ranking Phase 3 mapping starts from. Retrieval never calls an LLM.

**Evaluation** (`evaluation.py`). This scores each channel against the gold's `primary` links:
- Hit@k
- Recall@k
- MRR
- graded nDCG@10, where primary = 2 and supporting = 1

There are two query sources:
- `gold_text` uses the gold requirement text. It measures search quality alone.
- `ingested` uses the SRS Agent's extracted requirement, matched to the gold by ID, else by text similarity. It measures the full pipeline.

**Code-only baseline** on the library dataset: gold-text queries, no docstrings, 33 implemented or partial requirements.

| channel | hit@1 | hit@5 | recall@10 | MRR | nDCG@10 |
|---|---|---|---|---|---|
| code_lexical | 0.879 | 0.970 | 0.955 | 0.916 | 0.854 |
| code_vector | 0.849 | 0.970 | 0.985 | 0.897 | 0.897 |
| code_hybrid | 0.818 | 0.970 | 0.985 | 0.884 | 0.872 |

Most top-1 misses happen because the *containing class* outranks the method that does the work. A class's source includes all of its methods, so it matches almost everything about that class.

## Design

### Tables

| Table | Natural key | Contents |
|---|---|---|
| `tr_schema_migrations` | version | Schema version **and** the vector dimension the tables were created with |
| `tr_srs_documents` | document_id | One row per SRS output |
| `tr_requirements` | (document, requirement_id) | Every SRS Agent field. `source_location` and `extraction_meta` hold the rest, so nothing is dropped. Also an embedding and a full-text vector. |
| `tr_repositories` | repo_id | Path, git snapshot, graph file and format |
| `tr_code_components` | (repo, component_id) | Signature, parameters, return type, containing class, source code and its hash. Also a code embedding and a full-text vector. |
| `tr_docstrings` | (component, kind) | Docstrings stored as their own entity, with their own embedding |
| `tr_relationships` | (source, target, type) | DocAgent's typed edges. The type is restricted to `calls`, `imports`, `inherits`, `instantiates` or `contains`. |
| `tr_trace_links` | (type, req, docstring, component, pipeline version) | Proposed links (Phase 3). Status defaults to `pending_verification`. |
| `tr_verifications` | — | Append-only verdict history (Phase 4) |

### Rules

- **Scoping.** The same requirement ID can appear in two SRS documents, and the same component ID can appear in two repositories. Neither ever collides.
- **Separate embeddings.**
  - The requirement text covers the title, retrieval text, normalized text (or the raw text when normalization is empty) and the semantic lists.
  - The code text is identity, signature, and source with the docstring stripped.
  - The docstring text is the owning symbol, its signature, and the docstring.
  - Empty texts are never embedded; they are reported instead.
  - Texts longer than the model's 256-token window are reported as truncated.
- **Docstring provenance.** DocAgent saves its graph *before* it generates docstrings, then writes them into the source files, so the graph JSON never contains generated docstrings.
  - `existing` is the docstring present when DocAgent parsed the repo. It may itself come from an earlier DocAgent run.
  - `generated` is a non-empty docstring found under `--source-root` that differs from the parsed one. Components are matched by qualified name, not by line, because inserted docstrings shift every line below them.
  - Line numbers and `source_code` always refer to the parsed snapshot.
- **Transactions.** Each ingest of one SRS file or one graph is a single transaction. A failure leaves the database unchanged.
- **Re-ingest.** Upserts are idempotent.
  - Rows that disappear from a source are flagged `is_current = FALSE` rather than deleted, so links and verification history survive.
  - A repository's relationships are replaced as a set.
  - A docstring kind that was not looked for is left alone. For example, `generated` is not touched when no `--source-root` is given.
- **Guards.** `init_schema` refuses a different schema version or vector dimension rather than silently keeping the old columns. Every vector's length is checked before it is written.
- **Safety.** All values go through parameterized SQL. There are no `DROP` or `TRUNCATE` statements. Credentials come from the environment only and are never logged.

## Tests

```powershell
cd SRS_to_code_traceability
.\.venv\Scripts\python.exe -m pytest
```

- **Fixtures.** `tests/fixtures/` contains *real* outputs:
  - a DocAgent V1 graph and a legacy graph of `raw_test_repo_simple`, plus its source tree
  - two SRS Agent outputs

  Absolute paths in these files are replaced with `<repo>/`.
- **Database tests.** They use only `test_`-prefixed tables in the test database. The reset helper refuses to drop anything else. If PostgreSQL is unreachable, they are skipped with the reason shown.
- **Model.** Most tests use a deterministic fake embedder. A few use the real cached model.

## Known limitations

- **Long docstrings are truncated.** DocAgent's docstrings are long: 25 of 55 in the llama3.1-8b graph exceed 256 tokens, so only their beginning is embedded. Truncation is reported. Chunking or a larger-context model is a Phase 2 decision.
- **Legacy graphs are thin.** Pre-V1 graphs have no source, signatures or typed edges. Their components are stored, but they get no code embedding and no relationships.
- **`depends_on` is stored verbatim** for reference. It is noisy: it includes local variables resolved as module members, such as `example.e`. Graph enrichment should use the typed relationships instead.
- **Migrations exist only from 2.0.0.** `init_schema` migrates 2.0.0 → 2.1.0 automatically; this rebuilds the code full-text column with stemming. Any other older or newer version is refused.
- **Classes compete with their own methods** in retrieval, because a class's source contains every method's source. Possible fixes are to embed classes by signature only, or to down-weight classes. Decide after the docstring channels are measured.
- **The library dataset is easy for keyword search.** Its SRS and code were generated together, so they share vocabulary. Expect lower lexical scores on real projects.
- **No real evaluation yet.** The sample SRS doesn't describe the sample repositories, so retrieval and mapping quality can't be judged until a paired SRS and codebase exists.
