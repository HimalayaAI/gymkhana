"""Tests for IFEval constraint validators."""
import pytest
from gymkhana.envs.ifeval.ifeval import ConstraintVerifier


class TestCasingConstraints:
    """Test casing and character constraints."""

    def setup_method(self):
        """Create verifier instance for each test."""
        self.verifier = ConstraintVerifier()

    def test_validate_lowercase_pass(self):
        assert self.verifier.validate_lowercase("hello world") is True
        assert self.verifier.validate_lowercase("123 test") is True

    def test_validate_lowercase_fail(self):
        assert self.verifier.validate_lowercase("Hello world") is False
        assert self.verifier.validate_lowercase("HELLO") is False

    def test_validate_no_capital_letters(self):
        # Should be same as lowercase
        assert self.verifier.validate_no_capital_letters("hello") is True
        assert self.verifier.validate_no_capital_letters("Hello") is False


class TestPunctuationConstraints:
    """Test punctuation constraints."""

    def setup_method(self):
        """Create verifier instance for each test."""
        self.verifier = ConstraintVerifier()

    def test_validate_no_commas_pass(self):
        assert self.verifier.validate_no_commas("Hello world") is True
        assert self.verifier.validate_no_commas("Test. Test!") is True

    def test_validate_no_commas_fail(self):
        assert self.verifier.validate_no_commas("Hello, world") is False

    def test_validate_quotation_pass(self):
        assert self.verifier.validate_quotation('"Hello world"') is True
        assert self.verifier.validate_quotation('  "Test"  ') is True

    def test_validate_quotation_fail(self):
        assert self.verifier.validate_quotation('Hello world') is False
        assert self.verifier.validate_quotation('"Hello') is False
        assert self.verifier.validate_quotation('Hello"') is False


class TestStructuralConstraints:
    """Test structural constraints."""

    def setup_method(self):
        """Create verifier instance for each test."""
        self.verifier = ConstraintVerifier()

    def test_verify_paragraph_count_pass(self):
        text = "Para 1***Para 2***Para 3"
        assert self.verifier.verify_paragraph_count(text, N=3) is True

    def test_verify_paragraph_count_fail(self):
        text = "Para 1***Para 2"
        assert self.verifier.verify_paragraph_count(text, N=3) is False

    def test_validate_paragraphs_pass(self):
        text = "Para 1\n\nPara 2\n\nPara 3"
        assert self.verifier.validate_paragraphs(text, N=3) is True

    def test_validate_paragraphs_fail(self):
        text = "Para 1\n\nPara 2"
        assert self.verifier.validate_paragraphs(text, N=3) is False

    def test_validate_highlighted_sections_pass(self):
        text = "This is *highlighted* and *another* section"
        assert self.verifier.validate_highlighted_sections(text, N=2) is True
        assert self.verifier.validate_highlighted_sections(text, N=1) is True

    def test_validate_highlighted_sections_fail(self):
        text = "This is *highlighted*"
        assert self.verifier.validate_highlighted_sections(text, N=2) is False


class TestFrequencyConstraints:
    """Test content frequency constraints."""

    def setup_method(self):
        """Create verifier instance for each test."""
        self.verifier = ConstraintVerifier()

    def test_verify_keyword_frequency_pass(self):
        text = "The cat sat on the mat"
        assert self.verifier.verify_keyword_frequency(text, keyword_list=["the"], N=2) is True

    def test_verify_keyword_frequency_fail(self):
        text = "The cat sat on the mat"
        assert self.verifier.verify_keyword_frequency(text, keyword_list=["the"], N=3) is False

    def test_verify_letter_frequency_exact(self):
        text = "hello"
        assert self.verifier.verify_letter_frequency(text, letter="l", N=2) is True
        assert self.verifier.verify_letter_frequency(text, letter="l", N=3) is False

    def test_verify_letter_frequency_at_least(self):
        text = "hello"
        assert self.verifier.verify_letter_frequency(text, letter="l", N=2, quantifier="at least") is True
        assert self.verifier.verify_letter_frequency(text, letter="l", N=1, quantifier="at least") is True
        assert self.verifier.verify_letter_frequency(text, letter="l", N=3, quantifier="at least") is False

    def test_verify_letter_frequency_at_most(self):
        text = "hello"
        assert self.verifier.verify_letter_frequency(text, letter="l", N=2, quantifier="at most") is True
        assert self.verifier.verify_letter_frequency(text, letter="l", N=3, quantifier="at most") is True
        assert self.verifier.verify_letter_frequency(text, letter="l", N=1, quantifier="at most") is False

    def test_validate_frequency_capital_words(self):
        text = "This is HELLO and WORLD test"
        assert self.verifier.validate_frequency_capital_words(text, N=2) is True
        assert self.verifier.validate_frequency_capital_words(text, N=2, quantifier="at least") is True
        assert self.verifier.validate_frequency_capital_words(text, N=1, quantifier="at least") is True
        assert self.verifier.validate_frequency_capital_words(text, N=3, quantifier="at most") is True


class TestFormatConstraints:
    """Test format constraints."""

    def setup_method(self):
        """Create verifier instance for each test."""
        self.verifier = ConstraintVerifier()

    def test_validate_json_format_pass(self):
        assert self.verifier.validate_json_format('{"key": "value"}') is True
        assert self.verifier.validate_json_format('```json\n{"key": "value"}\n```') is True
        assert self.verifier.validate_json_format('```\n{"key": "value"}\n```') is True

    def test_validate_json_format_fail(self):
        assert self.verifier.validate_json_format('not json') is False
        assert self.verifier.validate_json_format('{invalid}') is False

    def test_validate_repeat_prompt_pass(self):
        text = "Repeat this: Hello world"
        assert self.verifier.validate_repeat_prompt(text, original_prompt="Repeat this:") is True

    def test_validate_repeat_prompt_fail(self):
        text = "Hello world"
        assert self.verifier.validate_repeat_prompt(text, original_prompt="Repeat this:") is False

    def test_validate_postscript_pass(self):
        text = "Main content\n\nP.S. This is a postscript"
        assert self.verifier.validate_postscript(text, postscript_marker="P.S.") is True

    def test_validate_postscript_fail(self):
        text = "Main content only"
        assert self.verifier.validate_postscript(text, postscript_marker="P.S.") is False


class TestWordConstraints:
    """Test word and section constraints."""

    def setup_method(self):
        """Create verifier instance for each test."""
        self.verifier = ConstraintVerifier()

    def test_validate_forbidden_words_pass(self):
        text = "This is a test"
        assert self.verifier.validate_forbidden_words(text, forbidden_words=["bad", "wrong"]) is True

    def test_validate_forbidden_words_fail(self):
        text = "This is a bad test"
        assert self.verifier.validate_forbidden_words(text, forbidden_words=["bad", "wrong"]) is False

    def test_validate_end_checker_pass(self):
        text = "Content here. The end."
        assert self.verifier.validate_end_checker(text, end_phrase="The end.") is True

    def test_validate_end_checker_fail(self):
        text = "Content here"
        assert self.verifier.validate_end_checker(text, end_phrase="The end.") is False

    def test_validate_word_constraint_pass(self):
        text = "This contains the word test"
        assert self.verifier.validate_word_constraint(text, word="test") is True

    def test_validate_word_constraint_fail(self):
        text = "This does not contain it"
        assert self.verifier.validate_word_constraint(text, word="test") is False

    def test_validate_number_of_words(self):
        text = "one two three four five"
        assert self.verifier.validate_number_of_words(text, N=5) is True
        assert self.verifier.validate_number_of_words(text, N=5, quantifier="at least") is True
        assert self.verifier.validate_number_of_words(text, N=3, quantifier="at least") is True
        assert self.verifier.validate_number_of_words(text, N=5, quantifier="at most") is True
        assert self.verifier.validate_number_of_words(text, N=7, quantifier="at most") is True
        assert self.verifier.validate_number_of_words(text, N=3, quantifier="at most") is False

    def test_validate_sections_pass(self):
        text = "Section 1\n\nSection 2\n\nSection 3"
        assert self.verifier.validate_sections(text, N=3, section_splitter="\n\n") is True

    def test_validate_sections_fail(self):
        text = "Section 1\n\nSection 2"
        assert self.verifier.validate_sections(text, N=3, section_splitter="\n\n") is False

    def test_validate_choice_pass(self):
        text = "option1"
        assert self.verifier.validate_choice(text, options=["option1", "option2"]) is True

    def test_validate_choice_fail(self):
        text = "option3"
        assert self.verifier.validate_choice(text, options=["option1", "option2"]) is False
