# English ShareGPT to Nepali

`english-sharegpt-to-nepali` translates a complete English conversation into
Nepali Devanagari while retaining canonical ShareGPT roles. It uses Gymkhana's
plain-text inference service; it does not create a provider client or submit an
external batch from inside the environment.

## Accepted input

The loader accepts local JSON/JSONL files, Hugging Face datasets, injected test
records, and these common row shapes:

- ShareGPT: `{"conversations":[{"from":"human","value":"..."}, ...]}`
- OpenAI messages: `{"messages":[{"role":"user","content":"..."}, ...]}`
- Flattened Hermes rows: `{"instruction":"...", "response":"..."}`

Original row fields such as `id`, `source`, category, and license metadata are
kept in `Task.metadata.source_provenance`. Optional reviewed translations may
be supplied in `nepali_conversations`, `reference_conversations`, or
`translation_reference`; references are verifier-only and are never put in the
policy prompt.

## Reward contract

The reward is in `[0, 1]`. Without a reference, it combines strict JSON/schema
validity, exact message-role structure, non-empty messages, Devanagari use, and
exact preservation of code/math/URL/tag/number spans. When a reference is
present, normalized per-message edit similarity contributes 35% of the reward.
Role/count changes and empty messages are hard failures. Missing any protected
span caps the score below the `0.80` export threshold.

This deterministic reward catches common data-corruption failures but cannot
prove semantic translation accuracy. Review a statistically meaningful sample
with fluent Nepali speakers before training or publishing a generated dataset.

## Dataset and license policy

No OpenHermes or other third-party dataset rows are bundled. OpenHermes combines
multiple sources with different licensing and attribution requirements. Keep
the upstream provenance fields, select only sources you are entitled to use,
and retain all required notices when distributing translations.
