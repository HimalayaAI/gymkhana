# Deterministic Nepali Romanized Bidirectional Translator

Pure Python transliteration for Nepali text:

```text
Devanagari Nepali <-> romanized Nepali
```

The package is deterministic, local, and self-contained. It does not call an
external service or JavaScript runtime.

This is the deterministic reference utility bundled with Gymkhana's
`romanized-nepali` RLVR environment. LLM candidates are generated separately
through Gymkhana's Pydantic AI inference service; the translator is not exposed
to the LLM as a tool.

## Install Gymkhana

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
```

## CLI

```bash
python -m gymkhana.envs.romanized_nepali.translator.transliterate dev-to-roman \
  "नेपालमा बैंकहरुले एआई कसरी प्रयोग गर्न सक्छन्?"

python -m gymkhana.envs.romanized_nepali.translator.transliterate roman-to-dev \
  "bank le AI prayog garna sakcha"

python -m gymkhana.envs.romanized_nepali.translator.transliterate round-trip \
  "ग्राहक सेवा सुधार गर्न सक्छ"
```

## Python API

```python
from gymkhana.envs.romanized_nepali.translator import create_translator

translator = create_translator()

romanized = translator.devanagari_to_romanized("एआई प्रयोग")
devanagari = translator.romanized_to_devanagari("bank le AI prayog garna sakcha")
round_trip = translator.round_trip("ग्राहक सेवा सुधार गर्न सक्छ")
```

## Design

The converter uses:

- high-confidence phrase and word overrides
- protected Latin technical terms such as `AI`, `API`, `GPU`, `CPU`, `QA`, and `RAG`
- suffix handling for common Nepali forms
- deterministic phonetic fallback rules for the long tail
- fixture-driven regression tests

Romanized Nepali is ambiguous, so this project prioritizes predictable output
over pretending there is a single perfect transliteration for every word.

## Test

```bash
python -m pytest -q tests/envs/test_romanized_translator.py
python -m gymkhana.envs.romanized_nepali.translator.evals
```

## Attribution

This package was informed by two Nepali transliteration projects:

- [`isDipesh/nepali-romanization`](https://github.com/isDipesh/nepali-romanization)
- [`BipinBudhathoki01/Nepaile-Unicode`](https://github.com/BipinBudhathoki01/Nepaile-Unicode)

The default translator is self-contained Python and does not vendor either
project. See [`ACKNOWLEDGMENTS.md`](../ACKNOWLEDGMENTS.md) and
[`THIRD_PARTY_NOTICES.md`](../THIRD_PARTY_NOTICES.md).
