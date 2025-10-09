"""
Configuration module for machine-specific data paths.

This module loads paths from the .env file and provides them as Path objects
for use throughout the project.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

# Load environment variables from .env file
# The .env file should be in the project root 
_env_path = Path(__file__).parent.parent.parent / ".env"
load_dotenv(dotenv_path=_env_path)


class DataPaths:
    """Container for all data-related paths in the project."""

    def __init__(self):
        # Load paths from environment variables
        self.raw_data_dir = Path(os.getenv("RAW_DATA_DIR", "data/raw_data"))
        self.processed_data_dir = Path(os.getenv("PROCESSED_DATA_DIR", "data/processed_data"))
        self.checkpoint_dir = Path(os.getenv("CHECKPOINT_DIR", "checkpoint"))

        # Ensure paths are absolute
        self._make_absolute()

    def _make_absolute(self):
        """Convert relative paths to absolute paths based on project root."""
        project_root = Path(__file__).parent.parent.parent

        if not self.raw_data_dir.is_absolute():
            self.raw_data_dir = project_root / self.raw_data_dir

        if not self.processed_data_dir.is_absolute():
            self.processed_data_dir = project_root / self.processed_data_dir

        if not self.checkpoint_dir.is_absolute():
            self.checkpoint_dir = project_root / self.checkpoint_dir

    # Convenience properties for commonly used files
    @property
    def act_info(self) -> Path:
        """Path to act_info.feather file."""
        return self.raw_data_dir / "act_info.feather"

    @property
    def sample_transaction(self) -> Path:
        """Path to sample_transaction.feather file."""
        return self.raw_data_dir / "sample_transaction.feather"

    def __repr__(self):
        return (
            f"DataPaths(\n"
            f"  raw_data_dir={self.raw_data_dir},\n"
            f"  processed_data_dir={self.processed_data_dir},\n"
            f"  checkpoint_dir={self.checkpoint_dir}\n"
            f")"
        )


# Create a singleton instance for easy import
paths = DataPaths()

