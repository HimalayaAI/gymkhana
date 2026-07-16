"""Tests for ConstraintVerifier implementing AnswerVerifier protocol."""
import pytest
from gymkhana.envs.ifeval.ifeval import ConstraintVerifier
from gymkhana.core.models import TrajectoryResult


class TestConstraintVerifierProtocol:
    """Test that ConstraintVerifier implements AnswerVerifier protocol correctly."""

    def setup_method(self):
        """Create verifier instance for each test."""
        self.verifier = ConstraintVerifier()

    def test_verify_with_valid_constraint(self):
        """Test verify() method with valid constraint."""
        task_metadata = {
            "ground_truth": {
                "func_name": "validate_lowercase",
            }
        }

        # Should pass
        result = self.verifier.verify(
            expected=None,
            candidates=["hello world"],
            task_metadata=task_metadata,
        )
        assert result is True

        # Should fail
        result = self.verifier.verify(
            expected=None,
            candidates=["Hello World"],
            task_metadata=task_metadata,
        )
        assert result is False

    def test_verify_with_parameters(self):
        """Test verify() method with constraint parameters."""
        task_metadata = {
            "ground_truth": {
                "func_name": "verify_keyword_frequency",
                "keyword_list": ["test"],
                "N": 2,
            }
        }

        # Should pass
        result = self.verifier.verify(
            expected=None,
            candidates=["This is a test and another test"],
            task_metadata=task_metadata,
        )
        assert result is True

        # Should fail (only 1 occurrence)
        result = self.verifier.verify(
            expected=None,
            candidates=["This is a test"],
            task_metadata=task_metadata,
        )
        assert result is False

    def test_verify_with_missing_metadata(self):
        """Test verify() returns None when metadata is missing."""
        result = self.verifier.verify(
            expected=None,
            candidates=["hello"],
            task_metadata=None,
        )
        assert result is None

    def test_verify_with_empty_candidates(self):
        """Test verify() returns None when candidates are empty."""
        task_metadata = {
            "ground_truth": {
                "func_name": "validate_lowercase",
            }
        }

        result = self.verifier.verify(
            expected=None,
            candidates=[],
            task_metadata=task_metadata,
        )
        assert result is None

    def test_verify_with_missing_func_name(self):
        """Test verify() returns None when func_name is missing."""
        task_metadata = {
            "ground_truth": {}
        }

        result = self.verifier.verify(
            expected=None,
            candidates=["hello"],
            task_metadata=task_metadata,
        )
        assert result is None

    def test_verify_with_unknown_validator(self):
        """Test verify() returns None when validator doesn't exist."""
        task_metadata = {
            "ground_truth": {
                "func_name": "nonexistent_validator",
            }
        }

        result = self.verifier.verify(
            expected=None,
            candidates=["hello"],
            task_metadata=task_metadata,
        )
        assert result is None

    def test_verify_with_trajectory_result(self):
        """Test verify() can accept optional trajectory parameter."""
        task_metadata = {
            "ground_truth": {
                "func_name": "validate_lowercase",
            }
        }

        # Create a mock trajectory result
        trajectory = TrajectoryResult(
            task_id="test",
            final_answer="hello world",
            turns=[],
            success=True,
        )

        result = self.verifier.verify(
            expected=None,
            candidates=["hello world"],
            task_metadata=task_metadata,
            trajectory=trajectory,
        )
        assert result is True

    def test_verify_uses_first_candidate(self):
        """Test that verify() uses the first candidate from the list."""
        task_metadata = {
            "ground_truth": {
                "func_name": "validate_lowercase",
            }
        }

        # First candidate passes, second fails
        result = self.verifier.verify(
            expected=None,
            candidates=["hello", "WORLD"],
            task_metadata=task_metadata,
        )
        assert result is True

        # First candidate fails, second passes
        result = self.verifier.verify(
            expected=None,
            candidates=["HELLO", "world"],
            task_metadata=task_metadata,
        )
        assert result is False
