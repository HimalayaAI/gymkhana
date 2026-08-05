# Built-in RAG verifiers

This package contains reusable, external verification methods for retrieval
tasks. They are verifier components rather than standalone dataset
environments: an answer-producing environment passes its question, retrieved
contexts, and candidate answer to a verifier from `compute_reward()`.

The design follows two rules:

1. The answer policy never sees verifier-only prompts or outputs.
2. The judge emits typed atomic verdicts, while Gymkhana computes the score.

This prevents a rollout from selecting its own reward and makes every metric
auditable at the claim or retrieved-context level.

## Metrics

### Faithfulness

The judge decomposes the candidate answer into atomic factual claims and marks
each claim supported or unsupported by the retrieved contexts alone.

```text
faithfulness = supported answer claims / total answer claims
```

No reference answer is required. Empty answers, empty claim sets, malformed
structured output, and judge failures score `0.0` and include an `error`.

### Groundedness

Groundedness is the token-efficient holistic counterpart to faithfulness. The
judge returns one of `ungrounded`, `partially_grounded`, or `fully_grounded`,
and Gymkhana maps those labels to `0.0`, `0.5`, and `1.0`. The model does not
provide a free-form numeric reward.

Use groundedness for lower-cost filtering and faithfulness when claim-level
auditability is required.

### Response relevance

The judge decides whether the candidate answer addresses the user's question.
The labels `irrelevant`, `partially_relevant`, and `fully_relevant` map to
`0.0`, `0.5`, and `1.0`. This metric evaluates topical/task relevance, not
factual support.

### Context relevance

The judge decides whether each retrieved context contains information useful
for answering the question.

```text
context relevance = relevant contexts / retrieved contexts
```

The verifier requires exactly one verdict for every zero-based context index.
Missing, duplicate, or unexpected indices fail closed with score `0.0`.

### Context precision

Context precision uses the same per-context relevance contract but preserves
retrieval order and computes average precision:

```text
context precision = sum(precision@k for each relevant context at k)
                    / number of relevant contexts
```

Relevant contexts ranked earlier receive a higher score. If none are relevant,
the score is `0.0`.

## Usage inside an environment

```python
from gymkhana.envs.config import LLMJudgeSettings
from gymkhana.verifiers.rag import FaithfulnessVerifier, RAGSample

async def compute_reward(self, result, answer_correct=None, task=None):
    verifier = FaithfulnessVerifier(
        settings=LLMJudgeSettings(model="openai:gpt-4.1-mini"),
        inference_service=self._inference_service,
        threshold=1.0,
    )
    metric = await verifier.verify(
        RAGSample(
            question=task.prompt,
            answer=result.final_answer,
            contexts=task.metadata["retrieved_contexts"],
        )
    )
    result.total_reward = metric.score
    result.answer_correct = metric.passed
    result.reward_function = metric.metric
    result.metadata["rag_verification"] = metric.model_dump(mode="json")
    return metric.score
```

Use a fixed judge model that is separate from the answer policy during RL.
Record its model/configuration with trajectory provenance. These metrics are
LLM-assisted and should be calibrated against human-labeled examples before
being treated as production reward signals.

## Scope

The implementations are RAGAS-inspired rather than API-compatible copies.
Reference-dependent context recall and embedding-similarity variants of answer
relevance are intentionally left for follow-up work so their dataset and
embedding contracts can be introduced explicitly.
