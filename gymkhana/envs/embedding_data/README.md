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
