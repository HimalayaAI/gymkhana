# Embedding retrieval data

`embedding-data` is a separate, seed-agnostic environment for preparing
query/positive/negative examples for embedding-model fine-tuning. It reuses
Gymkhana's shared environment runner and inference services, but writes
retrieval JSONL as its primary artifact because ShareGPT does not retain
contrastive labels.

## Swap datasets and domains with YAML

Copy `configs/embedding_data/domain_template.yaml`, set the dataset source and
map its columns to `id`, `query`, `positive`, `negatives`, `source`, `document`,
`license`, and `script`. Change `embedding.domain` and
`embedding.domain_instructions` to describe the new domain. The environment
code does not depend on legal-specific fields or prompt rules.

The checked-in `nepali_legal_seed.yaml` maps the seed dataset's `query`,
`positive`, and `negatives`. A run emits:

- `<output_basename>.jsonl`: seed pairs plus semantically accepted generated
  query paraphrases, each retaining positives, negative types, and provenance.
- `<output_basename>_audit.jsonl`: paraphrase candidates, judge decisions, and
  source provenance and row indices.
- Each row has a deterministic `split`. Rows are grouped by `doc_name`, and
  duplicate queries connect document groups so neither documents nor repeated
  queries cross splits. Without a document field, duplicate queries stay
  together and other seed rows get their own group. Defaults reserve 5% for
  validation and 5% for test.

Generated paraphrases are attached to the seed positive and negatives. Seed
rows themselves are preserved as the baseline labels. The judge is a filter,
not legal validation; generated examples should be sampled for human review.

## Negative verification stage (optional)

Set `embedding.verify_negatives: true` to verify and fill negatives inside the run:

- `load_tasks` retrieves near-neighbour passages for each row from a pool (`embedding.negative_pool_path`, a JSONL with a `text` field, plus the positives of the loaded rows). The retriever is `embedding.negative_retriever_model` (default `intfloat/multilingual-e5-small`, runs on CPU; `negative_retriever_query_prefix` / `negative_retriever_doc_prefix` default to e5's `query: ` / `passage: `). The default retriever needs the optional extra: `uv sync --extra embedding` (or `pip install 'gymkhana[embedding]'`); without it the run stops with an install hint. The row's own positive and the positives of rows with the same query are never candidates.
- The seed negatives are verified first, then the mined candidates. One LLM call scores the positive and every candidate from 0 to 10 for "contains the answer". A candidate is rejected as a likely false negative at a score of 0.5 or more (`negative_false_threshold`) or within 0.1 of the positive's score (`negative_false_margin`).
- The first `embedding.negative_target` survivors (default 5) are exported with `negative_types` `seed_verified` or `mined_verified`. Rows with fewer survivors are kept and flagged `below_target: true`; nothing is padded.
- If the judge call fails, the row is exported with no negatives, flagged `below_target`, and the error is recorded. Unverified negatives are never exported while this stage is on.
- The audit file gets a `negative_audit` entry per row: the positive's score, the rejected candidates with scores, and the below-target flag.

The judge uses the environment's configured LLM, so it runs on API credits and needs no GPU. `gymkhana.envs.embedding_data.negatives` also offers a local cross-encoder scorer for offline use (`postprocess` CLI).

## Suggested augmentation approaches

The current method is **label-preserving query augmentation**: change the
query while keeping its seed positives and negatives fixed. Other approaches
can teach additional retrieval behavior:

| Approach | What changes | What to verify |
| --- | --- | --- |
| Generate queries from positives | Write realistic questions that a passage answers, including facts not covered by the seed query. | The positive directly supports the query; reject questions that require outside facts. |
| Generate positives and negatives from queries | Start with queries from a non-embedding seed dataset, retrieve source passages as positives, and mine five negatives per positive from a domain corpus. A seed answer can guide retrieval, but should only be the positive when FAQ answers are the intended retrieval documents. | Verify that each positive supports the query and every negative fails to answer it. Require corpus provenance, deduplicate candidates, and flag queries without five valid negatives instead of padding. |
| Vary search style | Create keyword searches, conversational requests, common Nepali-English phrasing, and controlled noisy queries. | Each version keeps the same information need. Keep noisy queries labeled separately for sampling. |
| Cross-script and code-switched queries | Create Romanized Nepali, Devanagari, and mixed-language forms. | Check script and semantic equivalence; sample ambiguous transliterations for human review. |
| Add multiple positives | Attach multiple relevant passages or title/section/passage views to one query. | Verify every positive independently answers the query. This needs multi-positive schema support. |
| Mine hard negatives | Retrieve nearby passages that look relevant but do not answer the query. | Verify they are genuinely non-relevant; a negative containing the answer creates a false label. |
| Build multi-passage examples | Pair a query with multiple passages needed to answer it. | Verify the set supports the query and record whether one or all passages are required. This needs multi-passage labels and verification. |
| Use real search behavior | Turn privacy-reviewed user queries, reformulations, and clicks into retrieval examples. | Treat clicks as candidate relevance signals, not ground truth; verify the passage label. |

Recommended order: generate queries from source passages first, then add
cross-script variants. Once a domain corpus is indexed, use it both to mine
verified hard negatives for existing labels and to generate positives and
five negatives from query-only or QA seed rows. Multiple positives and
multi-passage examples require a schema extension; real search behavior
requires privacy-reviewed logs.

## Proposed corpus-mining environment and benchmark dataset

The current `embedding-data` environment only augments queries while keeping
seed positives and negatives fixed. A separate corpus-mining environment could
generate and verify positive and negative passage labels for a query; this is a
proposal, not part of the current implementation.

Use the [Nepali Embedding Bench](https://huggingface.co/datasets/W4ashabii/Embedding_Bench)
as the output-schema reference. Keep a corpus table keyed by `pid` with passage
text and provenance, and a query table keyed by `qid` with query metadata and
relevant/irrelevant passage IDs:

```json
{
  "qid": "legal_000001",
  "query": "कानुनी प्रश्न यहाँ",
  "domain": "legal",
  "subdomain": "family/marriage",
  "positives": [{"pid": "legal_p000144"}],
  "negatives": [{"pid": "legal_p000994"}],
  "source": "Somtharu181coder/jiban4_law_qa_clean",
  "source_id": "seed-row-id",
  "language": "ne",
  "script": "Deva"
}
```

Each corpus passage should carry `pid`, `text`, `domain`, `subdomain`,
`source`, `source_id` or document/chunk ID, source URL when available, and
license. Keep the benchmark-style domain partitions named `news`, `legal`,
`health`, and `textbook`; keep any train/validation/test assignment as a
separate field and group it by source document to prevent passage leakage.

### Suggested initial domain partitions

| Partition | Seed/source | Positive passage and negative strategy | Initial target |
| --- | --- | --- | ---: |
| `news` | [HimalayaAI Nepali news corpus](https://huggingface.co/datasets/himalaya-ai/nepali-news-corpus) | Chunk articles as positives; generate information-seeking queries; mine negatives from nearby stories that do not answer the query. | 4–5k triplets |
| `legal` | [Nepali legal QA seed](https://huggingface.co/datasets/Somtharu181coder/jiban4_law_qa_clean) and [bilingual law RAG QA](https://huggingface.co/datasets/chhatramani/nepal_5_law_RAG_QA) | Prefer the cited law section or retrieved context as the positive passage; use other sections from the same law as hard negatives only after checking they do not answer the query. | 4–5k triplets |
| `health` | [Nepali Health QA](https://huggingface.co/datasets/NepaliAI/Nepali-Health-QA) plus vetted health guidance | Use an authoritative, reviewed guidance passage as the positive; retrieve topic-similar passages for candidate negatives and review health labels before export. | 3–5k triplets |
| `textbook` | [Nepali textbook QA](https://huggingface.co/datasets/dineshkarki/textbook-qa-nepali-multiturn) | Use the matching textbook context/chunk as the positive; use nearby chapter passages as candidate negatives after answerability checks. | 4–5k triplets |

The target is **15–20k expanded `(query, positive, negative)` triplets** across
the four partitions. Report both the number of unique `qid`s and the expanded
triplet count, since each query can have multiple positives and negatives.
Preserve original source text as corpus passages; generated content should
create queries or propose candidate labels, with verification before a label
enters the training set.

### Negative count for training exports

The training target is **five distinct negatives per positive passage for a
query**. Count an expanded triplet for each `(query, positive, negative)`
combination. At a 5:1 ratio, 15–20k triplets therefore requires 3–4k
query-positive pairs before deduplication. Report the unique query count,
query-positive pair count, verified negative count, and final triplet count
separately. When a seed row has fewer than five negatives, a corpus-backed
export must retrieve and verify additional candidates; do not pad the count
with duplicates or unverified passages. If five valid negatives cannot be
found, keep the row flagged below target rather than silently treating it as
complete.

The current seed-preserving query-augmentation mode retains the seed's
negative list and does not guarantee five negatives per positive. Meeting the
5:1 target requires a negative-mining/export step; keep the original seed
labels identifiable in provenance.

Run a small generation pass:

```bash
python -m gymkhana.run \
  --config configs/embedding_data/nepali_legal_seed.yaml \
  --limit 2 \
  --no-database
```

`embedding-data` does not train the embedding model. Feed the retrieval JSONL to
the training job; use the separate multi-turn QA environment when you need
ShareGPT answer-model examples.

## Future corpus-backed curriculum

For domains with a source corpus, a later corpus adapter can build a local
vector index over versioned documents. The environment can then generate
query/positive examples from retrieved evidence, mine semantically close
negative passages, and verify document/page citations. Curriculum stages can
progress from direct paraphrases to close negatives, multi-document retrieval,
and Devanagari/Romanized cross-script matching. The seed-only path remains
available as a stable baseline; the vector index is an optional source/retriever
backend, not a requirement for every domain.
