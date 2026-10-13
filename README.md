# traceability

This package is the integration layer between **SRS Agent** (requirements) and **DocAgent** (code components, docstrings and typed relationships). It loads both tools' JSON outputs into PostgreSQL with pgvector. Later phases use that database for retrieval, requirement↔docstring↔code mapping and verification.

It only consumes the outputs of those tools and never imports either one.

Status: **Phase 1, schema and ingestion, is complete.** Retrieval (Phase 2), mapping (Phase 3) and verification (Phase 4) are not built yet. The trace-link and verification tables already exist so those phases can write to them.

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
```

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
- **No automatic migrations.** A database with a different schema version is refused, and a development database has to be reset by hand.
- **No real evaluation yet.** The sample SRS doesn't describe the sample repositories, so retrieval and mapping quality can't be judged until a paired SRS and codebase exists.
