# Preeti → Unicode fixer

Deterministic detection and conversion of **Preeti** (legacy Nepali ASCII font)
text, built for PDF extractions where most of a page is Unicode Devanagari
but some runs were typed in Preeti and came out as Latin/symbol text:

| extracted | meaning | fixed |
|---|---|---|
| `आर्थिक वर्ष %÷)*) को बजेट` | year 2080/81 | `आर्थिक वर्ष ५/०८० को बजेट` |
| `!@= कर निर्धारण` | item 12 | `१२. कर निर्धारण` |
| `-3_ सेवा शुल्क` | list item (घ) | `(घ) सेवा शुल्क` |
| `Ao'l6 पार्लर` | beauty parlour | `ब्युटि पार्लर` |
| `तिर्नुपर्ने 5 .` | sentence-final छ । | `तिर्नुपर्ने छ ।` |

Converting a whole chunk would corrupt its Unicode parts, so the fixer works
**per token, only inside Devanagari context**, and leaves real English,
acronyms, URLs, e-mails and Arabic numbers alone.

## Layout

| module | contents |
|---|---|
| `mapping.py` | Preeti glyph map, glyph combos (`km`→फ, `qm`→क्र, …) and `convert_preeti()` for text that is *entirely* Preeti (handles i-matra and reph re-ordering). |
| `detect.py` | `PreetiFixer`: splits text into Devanagari / non-Devanagari pieces, labels each piece, converts the Preeti ones. |
| `vocab.py` | Corpus vocabularies that sharpen Preeti-vs-English decisions; save/load as one-word-per-line files. |
| `frame.py` | `fix_frame()`: fix a pandas column, keep the original, return counts and seeded before/after review pairs. |
| `cli.py` | `python -m gymkhana.text.preeti {convert,build-vocab,fix}`. |

## How a token is decided

Each non-Devanagari piece (between spaces, or glued to Devanagari) gets one label:

* **strong**, converted whenever it is within two words of Devanagari (or glued to it):
  * Preeti numerals: shifted digit keys with `=` (dot) / `÷` (slash): `!@=`, `@)*)÷*!`
  * list markers: `-3_`, `-s_`
  * tokens whose conversion is a known Nepali word (needs a Nepali vocab)
  * tokens with an in-word Preeti signature (`'`, `"`, `[`, `]`, `{`, `}`, `|`, `\`, `;`, `/` inside the word), with no multi-digit numbers, dots or underscores
* **weak**, converted only when the chunk already has clear Preeti evidence (≥2 strong vocabulary hits or ≥5 strong tokens) and the neighbours fit:
  * a bare `.` after a Nepali word → `।`
  * a bare `5` after a Nepali word (not a counter word, not next to a number) → `छ`
* **keep**, everything else: URLs, e-mails, acronyms (`EOI`, `CCTV`, `PAMS`, any 2–8 capital letters), Arabic numbers (`082-560123`, `1,18,94,000/-`, `100%`), English words (`Pre-Bid`, `intervention/surgery`, any word in the English vocab), and anything whose conversion is not well-formed Devanagari.

`<Nepali word> 5 .` and `<Nepali word> 5 /` always become `छ ।` (and `5 <`
becomes `छ ?`) unless the word before is a counter such as दफा, धारा, वडा or नं.

A chunk with **no** Unicode Devanagari is converted as a whole only if at least
40% of its alphabetic tokens convert to known Nepali words (needs a Nepali vocab).

## Usage

```python
from gymkhana.text.preeti import PreetiFixer, build_english_vocab, build_nepali_vocab, convert_preeti

convert_preeti("g]kfn ;/sf/")                 # 'नेपाल सरकार'

fixer = PreetiFixer()                          # rules only
fixer.fix("Ao'l6 पार्लर 5 .").text              # 'ब्युटि पार्लर छ ।'

# with corpus vocabularies (recommended for real corpora)
fixer = PreetiFixer(
    nepali_vocab=build_nepali_vocab(clean_unicode_texts, min_freq=3),
    english_vocab=build_english_vocab(all_texts),
)
result = fixer.fix(text)
result.text, result.converted                  # fixed text, [(preeti, unicode), ...]
```

**Build the Nepali vocab from clean, hand-typed Unicode** (for the Nepal gov
corpus: `doc_type == "html"`). PDF extractions carry font damage and would teach
the vocabulary broken spellings.

### Command line

```bash
# one string
python -m gymkhana.text.preeti convert "Ao'l6 पार्लर सञ्चालन 5 ."
python -m gymkhana.text.preeti convert --whole "g]kfn ;/sf/"

# vocabularies once per corpus version
python -m gymkhana.text.preeti build-vocab \
    --hf voidash/previllage-nepal-gov-corpus --hf-config chunks --revision <sha> \
    --where document_is_current=1 --nepali-where doc_type=html --out vocab/

# fix a table (parquet, jsonl or HF dataset)
python -m gymkhana.text.preeti fix \
    --hf voidash/previllage-nepal-gov-corpus --hf-config chunks --revision <sha> \
    --where document_is_current=1 --vocab-dir vocab/ --output fixed.parquet --pairs 50
```

`fix` writes `fixed.parquet` (fixed `text`, original in `text_original`,
`preeti_touched`, `preeti_tokens`), `fixed.report.json` (row/token counts and
top conversions) and `fixed.pairs.jsonl` (seeded before/after samples to
eyeball).

## Limits

* Only Preeti. Other legacy fonts (Kantipur, PCS Nepali, …) map keys differently.
* Text that is broken Unicode rather than Preeti (e.g. `ववभागहरुसुँग` for
  विभागहरूसँग, lost conjuncts from bad PDF font tables) cannot be repaired here;
  filter it instead.
* Without an English vocabulary, unusual English words containing `'` (e.g.
  `agenda's`) can look like Preeti. Pass `english_vocab` for real corpora.
* A bare `/` is never converted: in government PDFs it is as often a danda or a
  real slash as Preeti र.
* Rare extended glyphs (`Ø`, `å`, `ß`, …) are mapped conservatively; unknown
  ones are left unchanged.
