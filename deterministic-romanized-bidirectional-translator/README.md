# Deterministic Nepali Romanized Bidirectional Translator

Pure Python transliteration for Nepali text:

```text
Devanagari Nepali <-> romanized Nepali
```

The package is deterministic, local, and self-contained. It does not call an
external service or JavaScript runtime.

## Install

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[test]"
```

## CLI

```bash
PYTHONPATH=src python -m nepali_romanized_translator.transliterate dev-to-roman \
  "नेपालमा बैंकहरुले एआई कसरी प्रयोग गर्न सक्छन्?"

PYTHONPATH=src python -m nepali_romanized_translator.transliterate roman-to-dev \
  "bank le AI prayog garna sakcha"

PYTHONPATH=src python -m nepali_romanized_translator.transliterate round-trip \
  "ग्राहक सेवा सुधार गर्न सक्छ"
```

## Python API

```python
from nepali_romanized_translator import create_translator

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
PYTHONPATH=src:tests python tests/run_local_checks.py
PYTHONPATH=src:tests python tests/run_smoke_cases.py
PYTHONPATH=src python -m nepali_romanized_translator.evals
```

If `pytest` is installed:

```bash
pytest
```

## Attribution

This package was informed by two Nepali transliteration projects:

- [`isDipesh/nepali-romanization`](https://github.com/isDipesh/nepali-romanization)
- [`BipinBudhathoki01/Nepaile-Unicode`](https://github.com/BipinBudhathoki01/Nepaile-Unicode)

The default translator is self-contained Python and does not vendor either
project. See [`ACKNOWLEDGMENTS.md`](ACKNOWLEDGMENTS.md) and
[`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md).
