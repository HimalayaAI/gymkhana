import pytest

from gymkhana.text.preeti import PreetiFixer


@pytest.fixture
def fixer() -> PreetiFixer:
    return PreetiFixer()


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("आर्थिक वर्ष %÷)*) को बजेट", "आर्थिक वर्ष ५/०८० को बजेट"),
        ("!@= कर निर्धारण गर्ने", "१२. कर निर्धारण गर्ने"),
        ("-3_ सेवा शुल्क तोकिएको", "(घ) सेवा शुल्क तोकिएको"),
        ("Ao'l6 पार्लर सञ्चालन", "ब्युटि पार्लर सञ्चालन"),
        ("कर तिर्नुपर्ने 5 .", "कर तिर्नुपर्ने छ ।"),
        ("विभेद व्याप्त 5 / दलितलाई", "विभेद व्याप्त छ । दलितलाई"),
        ("के कस्तो अधिकार प्राप्त 5 <", "के कस्तो अधिकार प्राप्त छ ?"),
    ],
)
def test_converts_preeti_spans(fixer: PreetiFixer, text: str, expected: str) -> None:
    assert fixer.fix(text).text == expected


@pytest.mark.parametrize(
    "text",
    [
        "EOI पेश गर्ने सम्बन्धी सूचना",
        "CCTV जडान गर्ने कार्य",
        "PAMS मार्फत हाजिरी",
        "Pre-Bid बैठक बस्ने",
        "थप जानकारीको लागि https://moha.gov.np/notice/123 हेर्नुहोला",
        "इमेल info@gadhawamun.gov.np मा पठाउनुहोस्",
        "फोन नं. 082-560123 मा सम्पर्क",
        "दफा 5 . अनुसार",
        "जम्मा 100% अनुदान",
        "रकम रु 1,18,94,000/- भुक्तानी",
        "शल्यक्रिया intervention/surgery. गरियो",
        "उपदफा (1) बमोजिम",
    ],
)
def test_whitelist_untouched(fixer: PreetiFixer, text: str) -> None:
    result = fixer.fix(text)
    assert result.text == text
    assert not result.touched


def test_unicode_text_untouched(fixer: PreetiFixer) -> None:
    text = "नेपालमा जम्मा ७५३ स्थानीय तह छन् । यीमध्ये ६ महानगरपालिका छन् ।"
    assert fixer.fix(text).text == text


def test_pure_english_untouched(fixer: PreetiFixer) -> None:
    text = "The municipality invites sealed bids for road construction."
    assert fixer.fix(text).text == text


def test_whole_preeti_chunk_needs_vocab() -> None:
    text = "g]kfn ;/sf/ sf] lg0f{o"
    assert PreetiFixer().fix(text).text == text
    vocab = {"नेपाल", "सरकार", "को", "निर्णय"}
    assert PreetiFixer(nepali_vocab=vocab).fix(text).text == "नेपाल सरकार को निर्णय"


def test_glued_preeti_inside_unicode() -> None:
    fixer = PreetiFixer(nepali_vocab={"टोलमा"})
    assert fixer.fix("मुक्तकमैया 6f]ndf३००००० ११").text == "मुक्तकमैया टोलमा३००००० ११"


def test_weak_tokens_need_evidence() -> None:
    vocab = {"न्यूनतम", "क्षमता"}
    plain = "सन्दर्भ / अवस्था ."
    assert PreetiFixer(nepali_vocab=vocab).fix(plain).text == plain
    mixed = "Go\"gtd ज्याला र Ifdtf सन्दर्भ / अवस्था ."
    # "/" stays (often a danda or real slash); "." after a word becomes the danda
    assert PreetiFixer(nepali_vocab=vocab).fix(mixed).text == "न्यूनतम ज्याला र क्षमता सन्दर्भ / अवस्था ।"


def test_bare_five_is_chha_after_words_not_in_tables() -> None:
    fixer = PreetiFixer(nepali_vocab={"न्यूनतम", "क्षमता"})
    evidence = "Go\"gtd र Ifdtf "
    assert fixer.fix(evidence + "भइरहेको 5 भन्ने कुरा").text.endswith("भइरहेको छ भन्ने कुरा")
    assert fixer.fix(evidence + "मैनपोखरी 5 1 0").text.endswith("मैनपोखरी 5 1 0")
    assert fixer.fix(evidence + "वडा 5 भित्र").text.endswith("वडा 5 भित्र")


def test_converted_list_reports_changes(fixer: PreetiFixer) -> None:
    result = fixer.fix("Ao'l6 पार्लर")
    assert result.converted == [("Ao'l6", "ब्युटि")]


def test_english_vocab_protects_words(fixer: PreetiFixer) -> None:
    text = "बैठकको agenda's सूची"
    assert PreetiFixer(english_vocab={"agenda"}).fix(text).text == text
