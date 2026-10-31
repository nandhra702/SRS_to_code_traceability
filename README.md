# traceability

This package is the integration layer between **SRS Agent** (requirements) and **DocAgent** (code components, docstrings and typed relationships). It loads both tools' JSON outputs into PostgreSQL with pgvector. Later phases use that database for retrieval, requirement↔docstring↔code mapping and verification.

It only consumes the outputs of those tools and never imports either one.

Status: **Phases 1–4 are implemented:**
1. schema and ingestion;
2. query resolution and retrieval, including the candidate "search agent";
3. the deterministic mapping agent;
4. the LLM verifier agent (`verification/`).

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

# Search agent: store the top-N candidates for every requirement, then score them
& $py -m traceability.cli find-candidates --document-id library-srs-v1.1 --repo-id "library@yi-coder-9b" --top-n 10
& $py -m traceability.cli evaluate-candidates --gold library_traceability_dataset\gold_standard\gold_standard_mapping.json `
    --document-id library-srs-v1.1
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

**Implementation candidates.** The `components` list fuses code hits and docstring hits; a docstring hit counts for its owning component. The fused score is the RRF sum **divided by the number of channels the component is eligible for**. A component with no docstring can't appear in the docstring channels, and that absence must not count against it; DocAgent skips `__init__` methods, for example. A plain sum dropped `Member.__init__` and `Loan.__init__` out of the top 10 on the library dataset. This list is the ranking Phase 3 mapping starts from. Retrieval never calls an LLM.

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

**With DocAgent docstrings.** The docstrings came from yi-coder:9b: 36 of 41 components, all except the 5 `__init__` methods, averaging 290 characters, with none truncated. All rows below are fused rankings:

| query source | channel | hit@1 | hit@5 | recall@10 | MRR | nDCG@10 |
|---|---|---|---|---|---|---|
| gold text | code_hybrid | 0.818 | 0.970 | 0.985 | 0.884 | 0.872 |
| gold text | doc_hybrid | 0.727 | 0.939 | 0.909 | 0.811 | 0.820 |
| gold text | **all_hybrid** | **0.849** | **1.000** | **0.985** | **0.908** | **0.899** |
| ingested (SRS Agent) | code_hybrid | 0.939 | 0.970 | 1.000 | 0.951 | 0.893 |
| ingested (SRS Agent) | **all_hybrid** | 0.909 | **1.000** | **1.000** | 0.950 | **0.900** |

Two things to note:
- Docstrings alone are weaker than code, but fusing them adds value: gold-text MRR goes from 0.884 to 0.908.
- Queries built from the SRS Agent's structured requirements (title, retrieval text, actions, entities, …) beat raw requirement text.

### Candidate search stage (the "search agent")

`find-candidates` (`candidates.py`) works through a document's requirements one by one, in document order. For each requirement it:
- runs the fused code + docstring ranking;
- stores the top N as `requirement_to_component` links with status `candidate`;
- records the fused rank and every channel's rank and score with each link.

Re-runs are idempotent. Unreviewed candidates that drop out of the top N are removed; links that have moved past `candidate` are never touched.

The stage makes **no found / not-found decision**; that is the verifier's job. It reports signals instead: best code and docstring similarity, channel agreement, and the number of lexical hits.

On the library dataset (code only, gold-text requirements), the right component is among the top 10 candidates for **every** implemented or partial requirement: hit@10 = 1.0, MRR 0.89.

Best code similarity differs between groups on average, but the ranges overlap, so no threshold can decide "not found":

| requirement status | mean similarity | range |
|---|---|---|
| implemented | 0.64 | 0.45–0.78 |
| not implemented | 0.50 | 0.45–0.54 |
| cross-cutting NFRs | 0.23 | — |

## Mapping (Phase 3): the "mapping agent"

```powershell
& $py -m traceability.cli map-requirements --document-id library-srs-v1.1 --repo-id "library@yi-coder-9b"
& $py -m traceability.cli evaluate-mapping --gold library_traceability_dataset\gold_standard\gold_standard_mapping.json --document-id library-srs-v1.1
& $py -m traceability.cli show-mapping FR-016 --document-id library-srs-v1.1
```

`mapping.py` turns each requirement's stored candidates into a **requirement subgraph** of proposed links. It is deterministic: no LLM is used, and no ID or edge is invented.

**Direct links** are the code believed to implement the requirement:
- The rank-1 candidate is always direct.
- A rank 2–3 candidate is direct only if it's *strong*: in the top 3 of at least 75% of the channels it's eligible for.
- A class is a *container*: if one of its own methods is also a top candidate, the class becomes a supporting link and the method is used instead.

**Supporting links** are one hop from each direct component, capped at 6 per requirement. They come from two evidence sources that are always labelled separately:

| Source | Edges used |
|---|---|
| `docagent` | DocAgent's typed edges: the containing class, `calls`, `instantiates` (plus the instantiated class's `__init__`), and callers. The **constructor of a direct method's class** is always included, ahead of every other supporting link, because it defines the state the method relies on ("the catalog starts empty"). |
| `source` | Call sites in the direct component's own code, e.g. `member.can_borrow()` on line 24 → `Member.can_borrow`. A name is resolved only if it's unique in the repo, or unique among the requirement's candidates; ambiguous names are skipped. |

DocAgent can't resolve calls on variables, so the `source` evidence matters: it raises supporting recall from 0.41 to 0.76.

**Docstring links:**
- requirement → docstring, for direct components matched through a docstring channel;
- docstring → owning component, a database fact stored as `supported` with basis `ownership`.

All requirement links are stored as `pending_verification` with their evidence: the selection reason, every channel's rank, and the edges with their source and line numbers. Re-runs replace unverified links, and links with a verdict are never touched. `show-mapping` prints one requirement's subgraph: its components, their roles, and the edges between them.

**Results on the library dataset** (real pipeline output: yi-coder docstrings + SRS Agent requirements, 33 scored requirements):

| Links | Precision | Recall | F1 |
|---|---|---|---|
| direct vs gold primary | 0.714 | 0.921 | 0.805 |
| supporting vs gold supporting | 0.206 | 0.759 | 0.324 |
| all links vs all gold links | 0.391 | **0.910** | 0.547 |

What these numbers mean:
- **91% of the gold's code ends up inside the requirement's subgraph.**
- **Supporting precision is low by design.** Supporting links are context for the verifier, not claims, and many "extra" ones are real callees, e.g. `Book.checkout_copy` for checkout. The gold's supporting set is narrower.
- **The 3 not-implemented and 3 cross-cutting requirements still get 2–7 links each.** The verifier must reject these.

Rules were chosen by measurement:

| Rule set | Direct precision | Direct recall |
|---|---|---|
| Rank 1 only | 0.91 | 0.79 |
| Top 3 | 0.37 | 0.97 |
| **Chosen rules** | **0.71** | **0.92** |

Recall was favoured, because the verifier filters out false positives.

## Verification (Phase 4): the "verifier agent"

```powershell
ollama pull qwen2.5-coder:7b        # default model; override with --model or VERIFIER_MODEL
& $py -m traceability.cli verify --document-id library-srs-v1.1
& $py -m traceability.cli evaluate-verification --gold library_traceability_dataset\gold_standard\gold_standard_mapping.json --document-id library-srs-v1.1
& $py -m traceability.cli show-verification FR-016 --document-id library-srs-v1.1
```

The `verification/` subpackage is the only stage that calls an LLM, which runs locally through Ollama. For each requirement:

1. **Context.** The mapped components are shown as tagged blocks (`[C1]`, `[C2]`, …), with their **real file line numbers**.
   - Direct code is shown in full, up to 120 lines.
   - Supporting functions are shown up to 40 lines.
   - Supporting classes show only their header.
   - Docstrings appear explicitly labelled as *documentation, NOT evidence*.
   - Anything cut for length is listed as omitted and can't be cited.

   A requirement with no mapped code becomes `not_found` without calling the model. That means the pipeline found no candidate, not that the feature is definitely absent.
2. **Claims** (`claims.py`) start as the requirement's own sentences and list items.
   - **`--claims clauses` (default).** A compound sentence ("…fewer than 2 times **and** the member has no outstanding fines") is split into separate clauses, so partly-implemented requirements can be detected. The model only *proposes* the split; it's accepted only if mechanical checks pass:
     - every content word already occurs in the requirement;
     - no number is dropped;
     - alternatives such as "10 **or** 13 characters" stay in one part;
     - every part is a complete "shall/must" sentence.

     Otherwise the original claim is kept, and the rejected split is logged with its reason.
   - **`--claims rules`:** sentences and list items only, with no model call.
   - **`--claims llm`:** the model splits freely. On a 7B model this invented claims such as "applies the correct interest rate", so it isn't recommended.
   - **`--claims gold`:** verifies the gold standard's own claims, to evaluate the verifier in isolation.

   If the text introduces a list that is missing ("…shall reject the book when:"), the requirement is flagged **incomplete**. The SRS Agent drops bullets and tables, and that fragment isn't judged.
3. **Judgement.** One model call per requirement, with a JSON schema enforced through Ollama's structured output, temperature 0 and a fixed seed. Each claim gets one of:
   - `supported`, which must cite line ranges **and quote the code**;
   - `not_supported`;
   - `insufficient_evidence`.
4. **Citation validation** (`context.py`). Every citation must name a component that was shown, and its quote must match code inside that component's shown lines.

   Citations are **anchored by their quote**, because models miscount lines or number them from the start of the method:
   - if the cited lines don't contain the quote, but the quote occurs verbatim exactly once in the shown code, the citation is moved there and the original lines are recorded as `realigned_from`;
   - if the quote occurs several times, the strictly nearest occurrence within 3 lines is used, otherwise the citation is rejected as ambiguous;
   - a fuzzy (non-verbatim) match never moves a citation more than 3 lines.

   A "supported" claim with no valid citation is downgraded to `insufficient_evidence`. Support can never come from a docstring, a similarity score, or an invented reference.
5. **Requirement verdict.**
   - all claims supported → `supported`;
   - some supported → `partially_supported`;
   - none supported, some contradicted → `not_supported_by_candidate`;
   - otherwise, or on model failure → `insufficient_evidence`.

   A timeout or malformed output is never turned into a positive verdict.
6. **Persistence.** Verdicts are stored in `tr_verifications`, which is append-only: one requirement-level row with every claim, verdict, citation and rejected citation, plus one row per judged link, along with the model and prompt version. Link statuses change as follows:
   - a link cited by a supported claim becomes `supported`;
   - an uncited direct link becomes `not_supported_by_candidate`;
   - uncited supporting links stay `pending_verification`, because they were context.

**Model:** `qwen2.5-coder:7b`.
- It fits a 6 GB GPU with an 8k context.
- It's strong at reading code and following schemas.
- It's a different model from DocAgent's docstring writer, so the verifier doesn't share its blind spots.

**Results on the library dataset** (prompt v2.1, 36 requirements with an expected verdict):

| Mode | False-support rate | Exact accuracy | Implemented recall | Claim accuracy | Citation validity |
|---|---|---|---|---|---|
| claims from the requirement text (end to end) | **0.000** | 0.861 | 0.933 | — | 0.984 |
| gold claims (verifier in isolation) | **0.000** | 0.917 | 0.933 | 0.913 | 0.978 |

- **No unmet requirement or claim was ever declared supported**, and every claim-level error was a conservative one.
- **Re-anchoring off-by-one citations to their quoted code** raised gold-claims exact accuracy from 0.722 to 0.917.
- **The remaining errors come mostly from upstream:** two requirements whose code was never mapped (FR-001, FR-034), and the SRS Agent dropping lists and tables. See `TEST_RUN1_LIBRARY_RESULTS.md`, Part C, in the parent folder.

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
