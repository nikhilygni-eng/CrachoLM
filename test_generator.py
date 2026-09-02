#!/usr/bin/env python3
"""
CrachoLM Text Generator Diagnostic & Verification Test
========================================================
Run this script to verify text generation logic, sampling, temperature, and top-k filtering.
"""

from tests.test_generator import test_generation_pipeline

if __name__ == "__main__":
    test_generation_pipeline()
