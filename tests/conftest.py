"""Pytest configuration and fixtures."""

import os

# Fix OpenMP conflict between PyTorch and FAISS on macOS
# Both libraries may link to different OpenMP runtimes
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
