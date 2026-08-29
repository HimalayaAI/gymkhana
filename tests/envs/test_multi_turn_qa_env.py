import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest
from pydantic import Field, ValidationError

from gymkhana.core.services import ServiceContainer
from gymkhana.core.services.inference import InferenceService
from gymkhana.envs import ENVIRONMENTS
from gymkhana.envs.config import EnvironmentType
from gymkhana.envs.multi_turn_qa import (
    AnswerType,
    ContextPolicy,
    MultiTurnQAEnv,
    PROFILES,
    QAGenerationSettings,
    QAVerifier,
    QuestionDraft,
    VerifierType,
)
from gymkhana.envs.multi_turn_qa.profiles import get_profile, select_qa_subcategory
from gymkhana.envs.multi_turn_qa.sources import SourceLoader
from gymkhana.run import load_environment_config


class ScriptedInference(InferenceService):
    responses: List[str]
    calls: List[Dict[str, Any]] = Field(default_factory=list)

    async def generate(
        self,
        *,
        messages: List[Dict[str, str]],
        system_prompt: Optional[str] = None,
        **kwargs: Any,
    ) -> str:
        self.calls.append(
            {
                "messages": [dict(message) for message in messages],
                "system_prompt": system_prompt,
                "kwargs": kwargs,
            }
        )
        return self.responses.pop(0)

    async def batch_generate(
        self,
        *,
        prompts: List[str],
        system_prompt: Optional[str] = None,
        **kwargs: Any,
    ) -> List[str]:
        return [self.responses.pop(0) for _ in prompts]


SOURCE = (
    "नियम १ अनुसार यस नियमावलीको नाम परीक्षण नियमावली हो। "
    "नियम २ अनुसार निवेदन सम्बन्धित कार्यालयमा पेस गर्नुपर्छ। "
    "कार्यालयले आवश्यक कागजात जाँच गरी निर्णय दिन्छ। "
) * 4


def question_draft(
    *,
    question: str,
    expected_answer: str,
    subcategory: str,
    visible_context: str = "",
    evidence: Optional[List[str]] = None,
    standalone: bool = True,
    answer_type: str = "source_grounded",
    verifier: str = "source_grounded",
) -> str:
    return json.dumps(
        {
            "question": question,
            "visible_context": visible_context,
            "expected_answer": expected_answer,
            "answer_type": answer_type,
            "verifier": verifier,
            "evidence": evidence or [],
            "learning_objective": "स्रोतबाट सही नियम पहिचान गर्नु",
            "subcategory": subcategory,
            "standalone": standalone,
        },
        ensure_ascii=False,
    )


JUDGE_PASS = json.dumps(
    {
        "accuracy": 4,
        "visible_context_sufficiency": 2,
        "grounding": 2,
        "clarity_and_instruction_following": 2,
        "total_score": 10,
        "reasoning": "The visible excerpt fully supports the answer.",
    }
)


def judge_result(
    *,
    total_score: int,
    grounding: int = 2,
    visible_context_sufficiency: int = 2,
) -> str:
    return json.dumps(
        {
            "accuracy": max(
                0,
                total_score - grounding - visible_context_sufficiency - 2,
            ),
            "visible_context_sufficiency": visible_context_sufficiency,
            "grounding": grounding,
            "clarity_and_instruction_following": 2,
            "total_score": total_score,
            "reasoning": "Scripted judge result.",
        }
    )


def base_config(tmp_path: Path, *, turns: int = 2):
    config = MultiTurnQAEnv.default_config.model_copy(deep=True)
    config.dataset.output_dir = str(tmp_path)
    config.dataset.output_basename = "qa"
    config.dataset.limit = 1
    config.dataset.field_mapping = {
        "id": "id",
        "text": "text",
        "source": "source",
        "subject": "subject",
        "grade": "grade",
        "title": "title",
        "language": "language",
        "license": "license",
        "jurisdiction": "jurisdiction",
        "document_date": "document_date",
        "pdf": "pdf",
    }
    config.generation.profile = "legal"
    config.generation.subcategory = "definitions"
    config.generation.turns = turns
    config.generation.target_language = "ne-Deva"
    config.generation.context_policy = ContextPolicy.INLINE_EXCERPT
    config.generation.min_source_chars = 20
    return config


def test_environment_and_profiles_are_registered() -> None:
    assert ENVIRONMENTS.get("multi-turn-qa") is MultiTurnQAEnv
    assert ENVIRONMENTS.get("multi_turn_qa") is MultiTurnQAEnv
    assert EnvironmentType.MULTI_TURN_QA.value == "multi-turn-qa"
    assert set(PROFILES) == {
        "agriculture",
        "banking",
        "ecommerce",
        "finance",
        "general",
        "health",
        "legal",
        "textbook",
    }
    for profile in PROFILES.values():
        assert profile.questioner_instructions
        assert profile.answerer_instructions
        assert profile.subcategories


def test_unknown_profile_fails_with_supported_names() -> None:
    with pytest.raises(ValueError, match="unknown QA profile"):
        get_profile("unknown")


def test_generation_config_validates_chunk_overlap() -> None:
    with pytest.raises(ValidationError, match="chunk_overlap_chars"):
        QAGenerationSettings(chunk_size_chars=500, chunk_overlap_chars=500)


def test_checked_in_dataset_configs_load_typed_settings() -> None:
    root = Path(__file__).parents[2]
    _, textbook = load_environment_config(
        root / "configs/multi_turn_qa/nepali_textbooks.yaml"
    )
    _, legal = load_environment_config(
        root / "configs/multi_turn_qa/nepali_legal_pdf.yaml"
    )

    assert textbook.dataset.dataset_name == "dineshkarki/nepali-textbooks-corpus"
    assert textbook.generation.profile == "textbook"
    assert textbook.generation.target_language == "ne-Deva"
    assert legal.dataset.dataset_name == "w4ashabii/nepali_legal_pdf"
    assert legal.dataset.dataset_split == "validation"
    assert legal.generation.source_kind == "pdf"
    assert legal.generation.document_date_strategy == "mapped_or_filename_or_text"


def test_all_legal_subcategories_are_reachable_under_auto() -> None:
    profile = get_profile("legal")
    signaled_cases = {
        "definitions": "यस परिच्छेदमा सेवाको परिभाषा दिइएको छ।",
        "rights_and_duties": "नागरिकको अधिकार र निकायको कर्तव्य उल्लेख छ।",
        "procedure": "निवेदन दर्ता गर्ने प्रक्रिया यहाँ उल्लेख छ।",
        "statutory_interpretation": "दफा ३ को उपदफा २ यस विषयमा लागू हुन्छ।",
        "scenario_application": "यदि यस्तो विवाद भएमा के व्यवस्था लागू हुन्छ?",
    }
    for expected, text in signaled_cases.items():
        assert select_qa_subcategory(profile, "कानून", "", text) == expected

    fallback_categories = {
        select_qa_subcategory(
            profile,
            "कानून",
            f"तटस्थ शीर्षक {index}",
            "सामग्री",
        )
        for index in range(100)
    }
    assert fallback_categories == set(profile.subcategories)


def test_textbook_loader_maps_schema_and_skips_frontmatter(tmp_path: Path) -> None:
    config = base_config(tmp_path, turns=1)
    config.generation.profile = "textbook"
    config.generation.subcategory = "conceptual"
    records = [
        {
            "id": "front",
            "text": SOURCE,
            "source": "Book",
            "subject": "Social Studies",
            "grade": 12,
            "title": "Chapter 0: Preface/Frontmatter",
        },
        {
            "id": "lesson",
            "text": SOURCE,
            "source": "Book",
            "subject": "Social Studies",
            "grade": 12,
            "title": "Chapter 2",
            "license": "apache-2.0",
        },
    ]

    task = MultiTurnQAEnv(config=config, records=records).load_tasks()[0]

    assert task.id == "lesson"
    assert task.metadata["source_document"]["grade"] == "12"
    assert task.metadata["source_document"]["subject"] == "Social Studies"
    assert task.metadata["source_provenance"]["subject"] == "Social Studies"
    assert task.metadata["source_provenance"]["license"] == "apache-2.0"


def test_pdf_rows_become_page_aware_chunks(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    config = base_config(tmp_path, turns=1)
    config.generation.source_kind = "pdf"
    config.generation.chunk_size_chars = 500
    config.generation.chunk_overlap_chars = 50
    config.generation.max_chunks_per_document = 2
    config.generation.document_date_strategy = "mapped_or_filename_or_text"
    config.dataset.limit = 2
    pages = [(1, "क" * 450), (2, "ख" * 450), (3, "ग" * 450)]
    monkeypatch.setattr(
        SourceLoader,
        "_extract_pdf_pages",
        staticmethod(lambda value: pages),
    )
    env = MultiTurnQAEnv(
        config=config,
        records=[{"pdf": {"path": "कानून-२०७९.pdf"}}],
    )

    tasks = env.load_tasks()

    assert len(tasks) == 2
    assert tasks[0].metadata["source_provenance"]["pdf_title"] == "कानून-२०७९.pdf"
    assert tasks[0].metadata["source_provenance"]["page_start"] == 1
    assert tasks[1].metadata["source_provenance"]["page_end"] >= 2
    assert tasks[0].metadata["source_provenance"]["document_date"] == "२०७९"
    assert tasks[0].metadata["source_provenance"]["document_date_source"] == (
        "filename"
    )


def test_profile_context_routing_is_source_safe(tmp_path: Path) -> None:
    legal_config = base_config(tmp_path, turns=2)
    legal = MultiTurnQAEnv(config=legal_config, records=[])
    source = SourceLoader(legal_config, [{"id": "x", "text": SOURCE}]).load(
        1, get_profile("legal")
    )[0]
    profile = get_profile("legal")
    assert legal._context_policy(profile, source, "definitions", 0) == (
        ContextPolicy.INLINE_EXCERPT
    )
    assert legal._context_policy(profile, source, "definitions", 1) == (
        ContextPolicy.CONVERSATION_GROUNDED
    )

    textbook_config = base_config(tmp_path, turns=1)
    textbook_config.generation.profile = "textbook"
    textbook_config.generation.subcategory = "math_stem"
    textbook_config.generation.context_policy = ContextPolicy.AUTO
    textbook = MultiTurnQAEnv(config=textbook_config, records=[])
    assert textbook._context_policy(
        get_profile("textbook"), source, "math_stem", 0
    ) == ContextPolicy.SELF_CONTAINED_PROBLEM


@pytest.mark.asyncio
async def test_two_agent_multiturn_run_exports_only_visible_context(tmp_path: Path) -> None:
    excerpt = "नियम १ अनुसार यस नियमावलीको नाम परीक्षण नियमावली हो।"
    responses = [
        question_draft(
            question="यस नियमावलीको नाम के हो?",
            expected_answer="यसको नाम परीक्षण नियमावली हो।",
            subcategory="definitions",
            visible_context=excerpt,
            evidence=[excerpt],
        ),
        "यस नियमावलीको नाम परीक्षण नियमावली हो।",
        JUDGE_PASS,
        question_draft(
            question="अघिल्लो उत्तरमा उल्लेख भएको नाम कुन नियममा दिइएको छ?",
            expected_answer="नियम १ मा।",
            subcategory="definitions",
            standalone=False,
        ),
        "त्यो नाम नियम १ मा दिइएको छ।",
        JUDGE_PASS,
    ]
    inference = ScriptedInference(responses=responses)
    config = base_config(tmp_path, turns=2)
    env = MultiTurnQAEnv(
        config=config,
        records=[
            {
                "id": "legal-1",
                "text": SOURCE + " PRIVATE_MARKER",
                "source": "Test law",
                "subject": "Law",
                "title": "परीक्षण नियमावली",
                "jurisdiction": "Nepal",
            }
        ],
        services=ServiceContainer(inference=inference),
    )

    summary = await env.run()

    assert summary.accepted == 1
    assert summary.results[0].answer_correct is True
    output = Path(summary.artifacts["sharegpt_jsonl"]).read_text(encoding="utf-8")
    row = json.loads(output)
    assert [message["from"] for message in row["conversations"]] == [
        "human",
        "gpt",
        "human",
        "gpt",
    ]
    assert excerpt in row["conversations"][0]["value"]
    assert row["subject"] == "Law"
    assert "PRIVATE_MARKER" not in output
    assert row["flattening_safe"] is False

    answerer_calls = [
        call
        for call in inference.calls
        if "answer agent" in (call["system_prompt"] or "")
    ]
    assert len(answerer_calls) == 2
    assert "PRIVATE_MARKER" not in json.dumps(
        answerer_calls, ensure_ascii=False
    )
    assert "यसको नाम परीक्षण नियमावली हो।" not in json.dumps(
        answerer_calls[0], ensure_ascii=False
    )

    judge_calls = [
        call
        for call in inference.calls
        if call["system_prompt"]
        == "You are a precise evaluation judge. Follow the output format exactly."
    ]
    assert len(judge_calls) == 2
    judge_prompt = judge_calls[0]["messages"][0]["content"]
    assert "Generate educational questions answerable strictly" in judge_prompt
    assert "Answer only from the visible legal excerpt" in judge_prompt
    assert "Source-grounding requirement:\nRequired." in judge_prompt

    audit = Path(summary.artifacts["audit_jsonl"]).read_text(encoding="utf-8")
    audit_row = json.loads(audit)
    assert audit_row["result_metadata"]["private_source_text"].endswith(
        "PRIVATE_MARKER"
    )
    assert len(audit_row["result_metadata"]["question_plans"]) == 2


@pytest.mark.asyncio
async def test_source_grounded_profile_forces_judge_and_grounding_gate(
    tmp_path: Path,
) -> None:
    excerpt = "नियम १ अनुसार दस्तुर ४२ रुपैयाँ हो।"
    responses = [
        question_draft(
            question="नियमअनुसार दस्तुर कति हो?",
            expected_answer="४२",
            subcategory="definitions",
            visible_context=excerpt,
            evidence=[excerpt],
            answer_type="exact",
            verifier="exact",
        ),
        "उत्तर ४२ रुपैयाँ हो।",
        judge_result(total_score=9, grounding=1),
    ]
    inference = ScriptedInference(responses=responses)
    config = base_config(tmp_path, turns=1)
    env = MultiTurnQAEnv(
        config=config,
        records=[{"id": "grounding", "text": SOURCE + excerpt}],
        services=ServiceContainer(inference=inference),
    )

    summary = await env.run()

    assert len(inference.calls) == 3
    assert summary.accepted == 0
    evaluation = summary.results[0].metadata["conversation_evaluation"]["turns"][0]
    assert evaluation["score"] == 0.9
    assert "source_grounding_below_required" in evaluation["reasons"]


@pytest.mark.asyncio
async def test_judge_score_is_normalized_before_threshold(tmp_path: Path) -> None:
    excerpt = "नियम १ अनुसार यस नियमावलीको नाम परीक्षण नियमावली हो।"
    responses = [
        question_draft(
            question="यस नियमावलीको नाम के हो?",
            expected_answer="परीक्षण नियमावली",
            subcategory="conceptual",
            visible_context=excerpt,
            evidence=[excerpt],
        ),
        "यसको नाम परीक्षण नियमावली हो।",
        judge_result(total_score=7),
    ]
    inference = ScriptedInference(responses=responses)
    config = base_config(tmp_path, turns=1)
    config.generation.profile = "general"
    config.generation.subcategory = "conceptual"
    config.generation.acceptance_threshold = 0.8
    env = MultiTurnQAEnv(
        config=config,
        records=[{"id": "normalized-score", "text": SOURCE}],
        services=ServiceContainer(inference=inference),
    )

    summary = await env.run()

    assert summary.accepted == 0
    evaluation = summary.results[0].metadata["conversation_evaluation"]["turns"][0]
    assert evaluation["score"] == 0.7
    assert evaluation["details"]["judge"]["raw_score"] == 7
    assert evaluation["details"]["judge"]["normalized_score"] == 0.7
    assert "judge_score_below_threshold" in evaluation["reasons"]


@pytest.mark.asyncio
async def test_single_turn_numeric_nepali_uses_deterministic_verifier(
    tmp_path: Path,
) -> None:
    config = base_config(tmp_path, turns=1)
    config.generation.profile = "textbook"
    config.generation.subcategory = "math_stem"
    config.generation.context_policy = ContextPolicy.SELF_CONTAINED_PROBLEM
    responses = [
        question_draft(
            question="२१ लाई २ ले गुणा गर्दा कति हुन्छ?",
            expected_answer="४२",
            subcategory="math_stem",
            answer_type="numeric",
            verifier="numeric",
        ),
        "२१ × २ = ४२ हुन्छ।",
    ]
    inference = ScriptedInference(responses=responses)
    env = MultiTurnQAEnv(
        config=config,
        records=[
            {
                "id": "math-1",
                "text": SOURCE,
                "source": "Math book",
                "subject": "गणित",
                "title": "गुणन",
            }
        ],
        services=ServiceContainer(inference=inference),
    )

    summary = await env.run()

    assert summary.accepted == 1
    assert len(inference.calls) == 2
    evaluation = summary.results[0].metadata["conversation_evaluation"]
    assert evaluation["turns"][0]["verifier"] == "numeric"
    assert evaluation["turns"][0]["score"] == 1.0


@pytest.mark.asyncio
async def test_invalid_questioner_output_is_retried(tmp_path: Path) -> None:
    excerpt = "नियम १ अनुसार यस नियमावलीको नाम परीक्षण नियमावली हो।"
    invalid = question_draft(
        question="यस नियमावलीको नाम के हो?",
        expected_answer="परीक्षण नियमावली",
        subcategory="definitions",
        visible_context=excerpt,
        evidence=[],
    )
    valid = question_draft(
        question="यस नियमावलीको नाम के हो?",
        expected_answer="परीक्षण नियमावली",
        subcategory="definitions",
        visible_context=excerpt,
        evidence=[excerpt],
    )
    inference = ScriptedInference(
        responses=[invalid, valid, "यसको नाम परीक्षण नियमावली हो।", JUDGE_PASS]
    )
    config = base_config(tmp_path, turns=1)
    env = MultiTurnQAEnv(
        config=config,
        records=[{"id": "retry", "text": SOURCE}],
        services=ServiceContainer(inference=inference),
    )

    summary = await env.run()

    assert summary.accepted == 1
    attempts = summary.results[0].metadata["questioner_attempts"][0]
    assert attempts[0]["issues"] == ["missing_source_evidence"]
    assert attempts[1]["issues"] == []


def test_language_validators_cover_devanagari_and_romanized(tmp_path: Path) -> None:
    config = base_config(tmp_path, turns=1)
    env = MultiTurnQAEnv(config=config, records=[])
    assert env._language_issues("यो उत्तर नेपाली भाषामा छ।") == []
    assert env._language_issues("This answer is English only.")

    config.generation.target_language = "ne-Latn"
    romanized = MultiTurnQAEnv(config=config, records=[])
    assert romanized._language_issues("yo uttar nepali bhasama chha.") == []
    assert romanized._language_issues("This answer is English only.")
    assert romanized._language_issues("यो देवनागरी उत्तर हो।")


def _plan(verifier: str, expected: str):
    from gymkhana.envs.multi_turn_qa.models import QATurnPlan

    return QATurnPlan.model_construct(
        expected_answer=expected,
        verifier=VerifierType(verifier),
        answer_type=AnswerType(verifier if verifier != "exact" else "exact"),
    )


def _verifier(tmp_path: Path, **overrides: Any) -> QAVerifier:
    config = base_config(tmp_path, turns=1)
    for key, value in overrides.items():
        setattr(config.generation, key, value)
    return QAVerifier(settings=config.generation)


@pytest.mark.parametrize(
    ("expected", "candidate", "score"),
    [
        ("B", "The answer is A. Not B.", 0.0),
        ("B", "उत्तर: (B)", 1.0),
        ("b", "B", 1.0),
        ("A", "Option B is correct.", 0.0),
        ("ख", "उत्तर ख हो।", 1.0),
    ],
)
def test_multiple_choice_uses_first_option_label(
    tmp_path: Path, expected: str, candidate: str, score: float
) -> None:
    verifier = _verifier(tmp_path)
    got, details = verifier._deterministic_score(_plan("multiple_choice", expected), candidate)
    assert got == score, details


@pytest.mark.parametrize(
    ("expected", "candidate", "score"),
    [
        ("हो", "होइन", 0.0),
        ("हो", "यसको उत्तर हो।", 1.0),
        ("a", "kathmandu", 0.0),
        ("काठमाडौँ", "राजधानी काठमाडौं हो", 1.0),
        ("Kathmandu", "The capital is Kathmandu.", 1.0),
    ],
)
def test_exact_match_requires_token_boundaries_and_folds_nasal_variants(
    tmp_path: Path, expected: str, candidate: str, score: float
) -> None:
    verifier = _verifier(tmp_path)
    got, _ = verifier._deterministic_score(_plan("exact", expected), candidate)
    assert got == score


@pytest.mark.parametrize(
    ("expected", "candidate", "score"),
    [
        ("1000", "2000 होइन, 1000", 1.0),
        ("1000", "1000 होइन, 2000", 0.0),
        ("42", "७ वटा समूह, जम्मा ४२", 1.0),
        ("42", r"७ र ६ गुणा गर्दा \boxed{४२} हुन्छ, ७ होइन", 1.0),
        ("42", "no number here", 0.0),
    ],
)
def test_numeric_uses_candidate_final_or_boxed_number(
    tmp_path: Path, expected: str, candidate: str, score: float
) -> None:
    verifier = _verifier(tmp_path)
    got, details = verifier._deterministic_score(_plan("numeric", expected), candidate)
    assert got == score, details


def test_builtin_language_specs_are_registered() -> None:
    from gymkhana.envs.multi_turn_qa import BUILTIN_LANGUAGES, resolve_language

    assert set(BUILTIN_LANGUAGES) == {"en", "ne-Deva", "ne-Latn"}
    assert resolve_language("ne-Latn").context_label == "Sandarbh"
    with pytest.raises(ValueError, match="unknown target_language"):
        resolve_language("xx-Zzzz")


def test_custom_language_spec_is_config_driven(tmp_path: Path) -> None:
    from gymkhana.envs.multi_turn_qa import LanguageSpec

    config = base_config(tmp_path, turns=1)
    with pytest.raises(ValidationError, match="unknown target_language"):
        config.generation.target_language = "hi-Deva"

    config.generation.languages = {
        "hi-Deva": LanguageSpec(
            code="hi-Deva",
            name="Hindi (Devanagari)",
            instruction="Write natural Hindi in Devanagari.",
            context_label="संदर्भ",
            question_label="प्रश्न",
            script_regex=r"[ऀ-ॿ]",
        ),
        "taj-Latn": LanguageSpec(
            code="taj-Latn",
            name="Tamang (Latin)",
            instruction="Write natural Tamang in Latin script.",
            forbidden_script_regex=r"[ऀ-ॿ]",
            marker_words={"la", "se", "ta"},
            min_marker_words=2,
        ),
    }
    config.generation.target_language = "hi-Deva"
    env = MultiTurnQAEnv(config=config, records=[])
    assert env._language_issues("यह उत्तर हिंदी में है।") == []
    assert env._language_issues("English only.") == ["insufficient_script_ratio:hi-Deva:0.000"]
    assert "Write natural Hindi" in env._answerer_system(get_profile("general"))
    rendered = env._render_user_message(
        QuestionDraft.model_construct(question="प्रश्न?", visible_context="संदर्भ पाठ"),
        ContextPolicy.INLINE_EXCERPT,
    )
    assert rendered.startswith("संदर्भ:\n")

    config.generation.target_language = "taj-Latn"
    tamang = MultiTurnQAEnv(config=config, records=[])
    assert tamang._language_issues("nga la se ta.") == []
    assert tamang._language_issues("यो देवनागरी हो।") == ["forbidden_script:taj-Latn"]
    assert tamang._language_issues("English only.") == ["insufficient_marker_words:taj-Latn:0"]


def test_language_key_must_match_spec_code(tmp_path: Path) -> None:
    from gymkhana.envs.multi_turn_qa import LanguageSpec

    config = base_config(tmp_path, turns=1)
    with pytest.raises(ValidationError, match="must equal spec code"):
        config.generation.languages = {
            "hi": LanguageSpec(code="hi-Deva", name="Hindi", instruction="x")
        }
