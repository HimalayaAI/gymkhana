"""Parsers and verifiers for answer extraction.

Protocols and implementations for handling model answers.
"""

from __future__ import annotations

from typing import (
    Any,
    Dict,
    List,
    Optional,
    Protocol,
    runtime_checkable,
)

from gymkhana.core.models.trajectory import TrajectoryResult


@runtime_checkable
class AnswerParser(Protocol):
    """Protocol for environment-specific answer parsing and normalization."""

    def extract_inline(self, response: str) -> List[str]:
        """Extract candidate answers directly from a model response."""
        ...

    def extract_from_result(self, result: TrajectoryResult) -> List[str]:
        """Extract candidate answers from a completed trajectory result."""
        ...

    def normalize(self, text: str) -> str:
        """Normalize text prior to comparison (case folding, trimming, etc.)."""
        ...


class BoxedAnswerParser(AnswerParser):
    """Default parser that operates on LaTeX-\boxed{} formatted answers."""

    def extract_inline(self, response: str) -> List[str]:
        from gymkhana.envs import parsers

        return parsers.extract_boxed_answers(response)

    def extract_from_result(self, result: TrajectoryResult) -> List[str]:
        from gymkhana.envs import parsers

        answers = parsers.extract_boxed_answers(result.final_answer)
        if not answers and result.final_answer:
            answers = [result.final_answer]
        return [answer for answer in answers if answer]

    def normalize(self, text: str) -> str:
        return text.strip().lower()


@runtime_checkable
class AnswerVerifier(Protocol):
    """Protocol for validating answers extracted from a trajectory."""

    def verify(
        self,
        *,
        expected: Optional[str],
        candidates: List[str],
        task_metadata: Optional[Dict[str, Any]] = None,
        trajectory: Optional[TrajectoryResult] = None,
    ) -> Optional[bool]:
        """Return True/False when validation is possible, otherwise None."""
        ...


class SimpleEqualityVerifier(AnswerVerifier):
    """Default verifier that performs case-insensitive string equality."""

    def __init__(self, *, case_sensitive: bool = False) -> None:
        self.case_sensitive = case_sensitive

    def _normalize(self, text: str) -> str:
        return text if self.case_sensitive else text.lower()

    def verify(
        self,
        *,
        expected: Optional[str],
        candidates: List[str],
        task_metadata: Optional[Dict[str, Any]] = None,
        trajectory: Optional[TrajectoryResult] = None,
    ) -> Optional[bool]:
        if expected is None or not candidates:
            return None
        expected_norm = self._normalize(expected.strip())
        for candidate in candidates:
            if self._normalize(candidate.strip()) == expected_norm:
                return True
        return False


class MathematicalVerifier(AnswerVerifier):
    """Verifier that checks mathematical equivalence using sympy."""

    def __init__(self, *, tolerance: float = 1e-6) -> None:
        self.tolerance = tolerance

    def _parse_to_sympy(self, text: str):
        """Try to parse text as a mathematical expression using sympy."""
        try:
            import sympy as sp
            import re

            # Clean up common LaTeX commands and delimiters
            cleaned = text.strip()

            # Remove inline math delimiters
            cleaned = cleaned.replace(r'\(', '').replace(r'\)', '')
            cleaned = cleaned.replace(r'\[', '').replace(r'\]', '')

            # Remove spacing commands
            cleaned = cleaned.replace(r'\,', '')
            cleaned = cleaned.replace(r'\:', '')
            cleaned = cleaned.replace(r'\;', '')
            cleaned = cleaned.replace(r'\!', '')
            cleaned = cleaned.replace(r'\quad', '')
            cleaned = cleaned.replace(r'\qquad', '')

            # Remove formatting commands
            cleaned = cleaned.replace(r'\left', '')
            cleaned = cleaned.replace(r'\right', '')
            cleaned = cleaned.replace(r'\big', '')
            cleaned = cleaned.replace(r'\Big', '')

            # Convert LaTeX fractions to Python fractions
            # \dfrac{a}{b} or \frac{a}{b} -> a/b
            frac_pattern = r'\\[dt]?frac\{([^}]+)\}\{([^}]+)\}'
            cleaned = re.sub(frac_pattern, r'(\1)/(\2)', cleaned)

            # Try parsing with sympy
            try:
                return sp.sympify(cleaned, evaluate=True)
            except:
                pass

            # Try parsing original text
            try:
                return sp.sympify(text.strip(), evaluate=True)
            except:
                pass

            # Try evaluating as Python expression
            try:
                result = eval(cleaned, {"__builtins__": {}}, {})
                return sp.sympify(result)
            except:
                pass

            return None
        except Exception:
            return None

    def _are_equivalent(self, expr1, expr2) -> bool:
        """Check if two sympy expressions are mathematically equivalent."""
        try:
            import sympy as sp

            if expr1 is None or expr2 is None:
                return False

            # Direct equality
            if expr1 == expr2:
                return True

            # Simplify and compare
            diff = sp.simplify(expr1 - expr2)
            if diff == 0:
                return True

            # Numerical comparison for floats
            try:
                val1 = float(expr1.evalf())
                val2 = float(expr2.evalf())
                if abs(val1 - val2) < self.tolerance:
                    return True
            except:
                pass

            return False
        except Exception:
            return False

    def verify(
        self,
        *,
        expected: Optional[str],
        candidates: List[str],
        task_metadata: Optional[Dict[str, Any]] = None,
        trajectory: Optional[TrajectoryResult] = None,
    ) -> Optional[bool]:
        if expected is None or not candidates:
            return None

        expected_expr = self._parse_to_sympy(expected)
        if expected_expr is None:
            # Fallback to string comparison if parsing fails
            expected_norm = expected.strip().lower()
            for candidate in candidates:
                if candidate.strip().lower() == expected_norm:
                    return True
            return False

        for candidate in candidates:
            candidate_expr = self._parse_to_sympy(candidate)
            if self._are_equivalent(expected_expr, candidate_expr):
                return True

        return False


__all__ = [
    "AnswerParser",
    "BoxedAnswerParser",
    "AnswerVerifier",
    "SimpleEqualityVerifier",
    "MathematicalVerifier",
]
