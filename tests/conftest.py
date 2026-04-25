"""Shared fixtures for all tests."""

import sys
from pathlib import Path

# Ensure the project root is on sys.path so `from embeddings…` etc. work.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
