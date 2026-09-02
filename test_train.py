#!/usr/bin/env python3
"""
CrachoLM Training Pipeline Diagnostic & Verification Test
==========================================================
Run this script to verify training loop execution, mixed-precision AMP,
checkpoint saving, and resuming from checkpoints.
"""

from tests.test_train import test_training_pipeline

if __name__ == "__main__":
    test_training_pipeline()
