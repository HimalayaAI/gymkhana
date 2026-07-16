"""Repository management utilities for local development and testing."""

import logging
import subprocess
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


class LocalRepoManager:
    """Manages local repository clones for development/testing without Docker."""

    def __init__(self, cache_dir: str = "/tmp/swe-repos"):
        """
        Initialize local repo manager.

        Args:
            cache_dir: Directory to cache cloned repositories
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def setup_repo(self, repo: str, commit: Optional[str] = None) -> Path:
        """
        Clone repository and optionally checkout specific commit.

        Args:
            repo: Repository name (e.g., "oauthlib/oauthlib")
            commit: Optional commit hash to checkout

        Returns:
            Path to cloned repository
        """
        repo_name = repo.replace('/', '_')
        repo_path = self.cache_dir / repo_name

        if not repo_path.exists():
            logger.info(f"Cloning {repo} to {repo_path}")
            subprocess.run([
                "git", "clone",
                f"https://github.com/{repo}.git",
                str(repo_path)
            ], check=True, capture_output=True)
        else:
            logger.info(f"Using cached repo: {repo_path}")

        if commit:
            logger.info(f"Checking out commit: {commit}")
            subprocess.run(
                ["git", "checkout", commit],
                cwd=repo_path,
                check=True,
                capture_output=True
            )

        # Install package in development mode
        if (repo_path / "setup.py").exists() or (repo_path / "pyproject.toml").exists():
            logger.info("Installing package in development mode")
            subprocess.run(
                ["pip", "install", "-e", "."],
                cwd=repo_path,
                check=True,
                capture_output=True
            )

        return repo_path

    def create_testbed_symlink(self, repo_path: Path):
        """
        Create /testbed symlink to repository.

        Note: Requires sudo on most systems.

        Args:
            repo_path: Path to repository
        """
        testbed = Path("/testbed")

        # Remove existing symlink/directory
        if testbed.exists() or testbed.is_symlink():
            subprocess.run(["sudo", "rm", "-rf", str(testbed)], check=True)

        # Create symlink
        subprocess.run(
            ["sudo", "ln", "-s", str(repo_path), str(testbed)],
            check=True
        )

        logger.info(f"Created symlink: /testbed -> {repo_path}")
