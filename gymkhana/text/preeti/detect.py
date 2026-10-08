"""Span-level detection and conversion of Preeti runs inside Unicode text.

PDF extraction of Nepal government documents often yields Unicode Devanagari
with fragments typed in the Preeti font, which come out as Latin/symbol text
(``g]kfn`` for नेपाल, ``5 .`` for छ ।). Converting a whole chunk would corrupt
its Unicode parts, so :class:`PreetiFixer` splits text into Devanagari and
non-Devanagari pieces, labels each non-Devanagari piece, and converts only
pieces that are Preeti *and* sit in Devanagari context.

Token labels:

``strong``
    Converted whenever it is near Devanagari: Preeti numerals (``!@=``,
    ``%÷)*)``), list markers (``-3_``), tokens whose conversion is a known
    Nepali word, and tokens with an in-word Preeti signature (``Ao'l6``).
``weak``
    A bare ``.`` (danda) or ``5`` (छ). Converted only when the chunk already
    shows clear Preeti evidence and the neighbours fit.
``keep``
    Everything else: English words, acronyms, URLs, e-mails, Arabic numbers,
    and anything whose conversion is not well-formed Devanagari.

``<Nepali word> 5 .`` and ``<Nepali word> 5 /`` are always rewritten to
``छ ।`` (``5 <`` to ``छ ?``) unless the word before is a counter such as दफा
or वडा.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Iterable, Literal, Optional

from .mapping import PREETI_ALPHABET, convert_preeti

Label = Literal["strong", "weak", "keep"]

DEVANAGARI_RE = re.compile(r"[\u0900-\u097F]")
_DEVA_RUN_RE = re.compile(r"[\u0900-\u097F]+")
_WS_SPLIT_RE = re.compile(r"(\s+)")

URL_RE = re.compile(r"^(?:https?://|www\.)\S+$|^\S+\.(?:gov|org|com|edu|net)(?:\.np)?\S*$", re.I)
EMAIL_RE = re.compile(r"^\S+@\S+\.\w{2,}\S*$")
ACRONYM_RE = re.compile(r"^[A-Z]{2,8}s?[.,:;)]*$")
NUMBER_RE = re.compile(r"^[(\[]?[0-9][0-9.,/:%\-]*[)\]]?[.,;:]?$")
# Preeti numerals: shifted digit keys, optionally with ÷ (slash), = (dot)
PREETI_NUMERAL_RE = re.compile(r"^[-(]?[!@#$%^&*()]+(?:[÷=][!@#$%^&*()]+)*[=_]?[,]?$")
# Preeti list markers: -s_ = (क), -3_ = (घ)
PREETI_LIST_MARKER_RE = re.compile(r"^-[A-Za-z0-9'\"]{1,3}_[.,]?$")
# Characters that are punctuation in English but letters/matras in Preeti.
_PREETI_INTERNAL = set("'\"[]{}|\\;/")
_ENGLISH_SHAPE_RE = re.compile(r"^[A-Za-z]+(?:[-'/&][A-Za-z]+)*[.,;:)]*$")
_EDGE_PUNCT = ".,;:()\"'"
_MULTI_DIGIT_RE = re.compile(r"[0-9]{2,}")
# English words joined by slashes ("intervention/surgery"); Preeti uses "/"
# for र inside short glyph runs, never between two long letter-only words.
_SLASH_WORDS_RE = re.compile(r"^[A-Za-z]{4,}(?:/[A-Za-z]{4,})+$")
# Bare tokens that are Preeti only when the chunk already shows Preeti:
# "." is the danda and "5" is छ. A bare "/" is left alone: in this corpus it
# is as often a danda or a real slash as Preeti र.
WEAK_TOKENS = frozenset({".", "5"})

# Words that precede a real Arabic numeral, so "दफा 5 ." is not "दफा छ ।".
COUNTER_WORDS = frozenset(
    "दफा धारा उपदफा नियम उपनियम नं नं. नम्बर वडा कक्षा बुँदा खण्ड अनुसूची प्रदेश तह "
    "पृष्ठ पाना संख्या क्र.सं. क्रम".split()
)

DEFAULT_ACRONYMS = frozenset(
    """EOI CCTV PAMS PDF NPR VAT PAN ID ICT GIS NGO INGO UNDP ADB WHO COVID SMS IT
    RFP RFQ BOQ DPR EIA IEE GPS QR LMBIS SuTRA PPMO GoN NRB NEA CDO DAO PIS OCR
    UN UNICEF USAID DFID JICA KOICA GIZ FY BS AD Rs NRs Pvt Ltd No Mr Mrs Dr""".split()
)

# Small built-in English list so the fixer behaves sensibly without a corpus
# vocabulary; real runs pass a corpus-derived ``english_vocab``.
DEFAULT_ENGLISH = frozenset(
    """a an and are as at be bid by do for from has have in is it of on or pre
    the this that to was were will with home read more you here success
    submitted notice office municipality rural government nepal province
    district ward tender form application download""".split()
)


@dataclass
class FixResult:
    text: str
    converted: list[tuple[str, str]] = field(default_factory=list)

    @property
    def touched(self) -> bool:
        return bool(self.converted)


def _strip_edges(token: str) -> str:
    return token.strip(_EDGE_PUNCT)


def _is_wellformed(deva: str) -> bool:
    """Reject conversions that leave Latin letters or start with a dependent sign."""

    if re.search(r"[A-Za-z]", deva):
        return False
    core = deva.strip(" ।.,()?“”/")
    if not core:
        return True
    if re.match(r"^[\u093E-\u094D\u0901-\u0903]", core):
        return False
    if re.search(r"[\u093E-\u094C]{2,}|\u094D[^\u0915-\u0939\u0958-\u095F\u200D]", core + " "):
        return False
    return True


class PreetiFixer:
    """Detect and convert Preeti runs inside Unicode Devanagari text."""

    def __init__(
        self,
        nepali_vocab: Optional[Iterable[str]] = None,
        english_vocab: Optional[Iterable[str]] = None,
        acronyms: Optional[Iterable[str]] = None,
        whole_chunk_min_vocab_rate: float = 0.4,
    ) -> None:
        self.nepali_vocab = frozenset(nepali_vocab or ())
        self.english_vocab = frozenset(w.lower() for w in (english_vocab or ())) | DEFAULT_ENGLISH
        self.acronyms = frozenset(acronyms or ()) | DEFAULT_ACRONYMS
        self.whole_chunk_min_vocab_rate = whole_chunk_min_vocab_rate

    # -- token predicates -------------------------------------------------

    def _in_nepali_vocab(self, deva: str) -> bool:
        if not self.nepali_vocab:
            return False
        core = deva.strip(" ।.,()?“”/:")
        return len(core) >= 2 and core in self.nepali_vocab

    def is_whitelisted(self, token: str) -> bool:
        core = _strip_edges(token)
        if not core:
            return False
        if URL_RE.match(core) or EMAIL_RE.match(core):
            return True
        if core in self.acronyms or ACRONYM_RE.match(core):
            return True
        if NUMBER_RE.match(token):
            return True
        if _SLASH_WORDS_RE.match(core):
            return True
        if core.endswith(("'s", "'S")):
            core = core[:-2]
        if _ENGLISH_SHAPE_RE.match(core):
            parts = [p for p in re.split(r"[-'/&]", core.lower()) if p]
            if parts and all(p in self.english_vocab for p in parts):
                return True
        return False

    def classify(self, token: str) -> Label:
        """Return ``strong``, ``weak`` or ``keep`` for a non-Devanagari token.

        ``strong`` tokens are converted whenever they sit in Devanagari
        context. ``weak`` tokens (a bare ``.`` or ``5``) are converted
        only in chunks that already show clear Preeti evidence.
        """

        if not token or not set(token) <= PREETI_ALPHABET:
            return "keep"
        if token in WEAK_TOKENS:
            return "weak"
        if PREETI_NUMERAL_RE.match(token):
            numerals = sum(1 for ch in token if ch in "!@#$%^&*")
            if numerals >= 1 and len(token) >= 2:
                return "strong"
            return "keep"
        if PREETI_LIST_MARKER_RE.match(token):
            return "strong"
        if self.is_whitelisted(token):
            return "keep"
        core = _strip_edges(token)
        if len(core) < 2 or not re.search(r"[A-Za-z]", core):
            return "keep"
        deva = convert_preeti(token)
        if not _is_wellformed(deva):
            return "keep"
        if self._in_nepali_vocab(deva):
            return "strong"
        # Without a vocabulary hit, require an in-word Preeti signature and
        # nothing that looks like a number, file name or slash-joined words.
        if _MULTI_DIGIT_RE.search(core) or "." in core or "_" in core:
            return "keep"
        has_internal = any(ch in _PREETI_INTERNAL for ch in core[1:-1])
        if has_internal and len(core) >= 3:
            return "strong"
        return "keep"

    # -- span-level fixing -----------------------------------------------

    def _segments(self, text: str) -> list[tuple[str, str]]:
        """Split text into (kind, piece): ws, deva, or other."""

        segs: list[tuple[str, str]] = []
        for piece in _WS_SPLIT_RE.split(text):
            if not piece:
                continue
            if piece.isspace():
                segs.append(("ws", piece))
                continue
            pos = 0
            for m in _DEVA_RUN_RE.finditer(piece):
                if m.start() > pos:
                    segs.append(("other", piece[pos:m.start()]))
                segs.append(("deva", m.group()))
                pos = m.end()
            if pos < len(piece):
                segs.append(("other", piece[pos:]))
        return segs

    def fix(self, text: str) -> FixResult:
        segs = self._segments(text)
        n = len(segs)
        has_deva = any(kind == "deva" for kind, _ in segs)
        labels: list[str] = ["" for _ in segs]
        for i, (kind, piece) in enumerate(segs):
            if kind == "other":
                labels[i] = self.classify(piece)

        word_idx = [i for i, (kind, _) in enumerate(segs) if kind != "ws"]

        if not has_deva:
            # Whole-Preeti chunk: convert only if most alphabetic tokens
            # turn into known Nepali words.
            alpha = [i for i in word_idx if re.search(r"[A-Za-z]", segs[i][1])]
            if not alpha or not self.nepali_vocab:
                return FixResult(text)
            hits = sum(1 for i in alpha if self._in_nepali_vocab(convert_preeti(segs[i][1])))
            if hits / len(alpha) < self.whole_chunk_min_vocab_rate:
                return FixResult(text)
            convert = {i for i in word_idx if labels[i] in ("strong", "weak")}
            return self._apply(segs, convert)

        def is_deva_ctx(j: int, chosen: set[int]) -> bool:
            return segs[j][0] == "deva" or j in chosen

        # Pass 1: strong tokens next to Devanagari (within two word slots).
        pos_in_words = {seg_i: k for k, seg_i in enumerate(word_idx)}
        chosen: set[int] = set()
        for _ in range(3):
            changed = False
            for k, i in enumerate(word_idx):
                if labels[i] != "strong" or i in chosen:
                    continue
                window = word_idx[max(0, k - 2):k] + word_idx[k + 1:k + 3]
                if any(is_deva_ctx(j, chosen) for j in window):
                    chosen.add(i)
                    changed = True
            if not changed:
                break

        strong_vocab_hits = sum(
            1 for i in chosen if self._in_nepali_vocab(convert_preeti(segs[i][1]))
        )
        evidence = strong_vocab_hits >= 2 or len(chosen) >= 5

        # Pass 2: weak tokens need chunk-level evidence, must stand alone,
        # and must follow a Devanagari word ("/" also needs one after it).
        if evidence:
            for i in word_idx:
                if labels[i] != "weak" or i in chosen:
                    continue
                k = pos_in_words[i]
                prev_ok = k > 0 and segs[word_idx[k - 1]][0] == "deva"
                next_ok = k + 1 < len(word_idx) and segs[word_idx[k + 1]][0] == "deva"
                piece = segs[i][1]
                if not prev_ok:
                    continue
                if piece == "5":
                    # छ after a Nepali word, not a number in a table row.
                    prev_word = segs[word_idx[k - 1]][1]
                    nxt = segs[word_idx[k + 1]][1] if k + 1 < len(word_idx) else ""
                    if prev_word in COUNTER_WORDS or re.search(r"[0-9\u0966-\u096F]", nxt):
                        continue
                    if not (nxt[:1] in ("", ".", "।", ",", "/") or next_ok):
                        continue
                chosen.add(i)

        # Glued pieces (Preeti stuck to Devanagari without a space) count
        # as in-context by construction.
        for i in range(n):
            if segs[i][0] == "other" and labels[i] == "strong" and i not in chosen:
                if (i > 0 and segs[i - 1][0] == "deva") or (i + 1 < n and segs[i + 1][0] == "deva"):
                    chosen.add(i)

        result = self._apply(segs, chosen)
        return self._fix_trailing_chha(result)

    def _apply(self, segs: list[tuple[str, str]], convert: set[int]) -> FixResult:
        parts: list[str] = []
        converted: list[tuple[str, str]] = []
        for i, (_, piece) in enumerate(segs):
            if i in convert:
                new = convert_preeti(piece)
                if new != piece:
                    converted.append((piece, new))
                parts.append(new)
            else:
                parts.append(piece)
        return FixResult("".join(parts), converted)

    _TRAILING_CHHA_RE = re.compile(r"(?P<prev>[\u0900-\u097F.]+)(?P<ws>\s+)5(?P<ws2>\s*)(?P<end>[./<])(?=\s|$)")

    def _fix_trailing_chha(self, result: FixResult) -> FixResult:
        """``<word> 5 .`` / ``5 /`` is Preeti for ``छ ।`` (``5 <`` for ``छ ?``) unless 5 is a counted number."""

        converted = list(result.converted)

        def repl(m: re.Match[str]) -> str:
            if m.group("prev") in COUNTER_WORDS:
                return m.group(0)
            end = "?" if m.group("end") == "<" else "।"
            converted.append((m.group(0)[len(m.group("prev")) + len(m.group("ws")):], "छ" + m.group("ws2") + end))
            return f"{m.group('prev')}{m.group('ws')}छ{m.group('ws2')}{end}"

        text = self._TRAILING_CHHA_RE.sub(repl, result.text)
        return FixResult(text, converted)
