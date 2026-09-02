#!/usr/bin/env python3
"""
CrachoLM Data Pipeline Diagnostic & Verification Test
======================================================
Run this script to verify raw text loading, tokenization, target shifting,
and PyTorch DataLoader generation.
"""

from tests.test_data import test_data_pipeline

if __name__ == "__main__":
    test_data_pipeline()
