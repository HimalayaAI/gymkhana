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
