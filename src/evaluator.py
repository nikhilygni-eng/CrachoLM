"""
CrachoLM Phase 8: Evaluation System
=====================================
Evaluates a trained CrachoLM model checkpoint against an evaluation-only
question bank. Reuses existing generate_text, CrachoTokenizer, and CrachoLM
components without modifying model weights.

Evaluation categories:
  - next_token_prediction  : Does the model predict a plausible next character?
  - exact_answer           : Does generated text start with the expected string?
  - multiple_choice        : Does the model pick the correct option?
  - numerical              : Does the model generate the correct number?

IMPORTANT LIMITATIONS (read before interpreting results):
  - CrachoLM-0.1 is a ~1.7M parameter character-level model trained from scratch.
  - It has no factual knowledge, no world model, and no reasoning ability.
  - A correct answer may be coincidental pattern-matching, not understanding.
  - Do NOT interpret a high accuracy score as general intelligence.
  - Do NOT use the model itself to judge its own factual correctness.
"""

import os
import json
import csv
import math
import sys
import torch
from typing import List, Dict, Any, Optional
from datetime import datetime

# Reuse all existing project modules without modification
from src.device import get_device
from src.tokenizer import CrachoTokenizer
from src.model import CrachoLM
from src.generator import generate_text
from src.data import load_raw_text


# ─────────────────────────────────────────────────────────────
# Answer Normalization
# ─────────────────────────────────────────────────────────────

def normalize_answer(text: str, case_sensitive: bool = False) -> str:
    """
    Normalizes a generated or expected answer string for fair comparison.
    - Strips leading/trailing whitespace
    - Optionally lowercases
    - Strips common trailing punctuation
    """
    text = text.strip()
    if not case_sensitive:
        text = text.lower()
    # Remove trailing punctuation that models often append
    text = text.rstrip(".,!?;:")
    return text


def extract_first_word(text: str) -> str:
    """Returns the first whitespace-separated token from a string."""
    parts = text.strip().split()
    return parts[0] if parts else text


# ─────────────────────────────────────────────────────────────
# Per-Category Evaluation Logic
# ─────────────────────────────────────────────────────────────

def evaluate_next_token(generated: str, prompt: str, expected: str, case_sensitive: bool) -> bool:
    """
    Evaluates next-token prediction: checks whether the first new character
    after the prompt matches the expected continuation.
    """
    # Strip the prompt from the front of generated text if present
    if generated.startswith(prompt):
        new_part = generated[len(prompt):]
    else:
        new_part = generated

    first_char = new_part[:1] if new_part else ""
    return normalize_answer(first_char, case_sensitive) == normalize_answer(expected, case_sensitive)


def evaluate_exact_answer(generated: str, prompt: str, expected: str, case_sensitive: bool) -> bool:
    """
    Evaluates exact answer: checks whether the generated continuation
    starts with the expected answer.
    """
    if generated.startswith(prompt):
        new_part = generated[len(prompt):]
    else:
        new_part = generated

    first_word = extract_first_word(new_part)
    return normalize_answer(first_word, case_sensitive) == normalize_answer(expected, case_sensitive)


def evaluate_multiple_choice(generated: str, prompt: str, expected: str,
                              choices: List[str], case_sensitive: bool) -> bool:
    """
    Evaluates multiple-choice: finds the highest-priority choice that
    appears at the start of the generated continuation.
    """
    if generated.startswith(prompt):
        new_part = generated[len(prompt):]
    else:
        new_part = generated

    new_part_norm = normalize_answer(new_part[:30], case_sensitive)  # inspect first 30 chars

    # Check which choice appears first in the output
    earliest_pos = None
    selected_choice = None
    for choice in choices:
        norm_choice = normalize_answer(choice, case_sensitive)
        pos = new_part_norm.find(norm_choice)
        if pos >= 0:
            if earliest_pos is None or pos < earliest_pos:
                earliest_pos = pos
                selected_choice = choice

    if selected_choice is None:
        return False
    return normalize_answer(selected_choice, case_sensitive) == normalize_answer(expected, case_sensitive)


def evaluate_numerical(generated: str, prompt: str, expected: str, case_sensitive: bool) -> bool:
    """
    Evaluates numerical answer: extracts first numeric token from continuation.
    """
    if generated.startswith(prompt):
        new_part = generated[len(prompt):]
    else:
        new_part = generated

    # Extract first run of digit characters
    digits = ""
    for ch in new_part.strip():
        if ch.isdigit() or ch in (".", "-"):
            digits += ch
        elif digits:
            break

    if not digits:
        return False

    try:
        return float(digits) == float(expected.strip())
    except ValueError:
        return False


# ─────────────────────────────────────────────────────────────
# Dispatcher
# ─────────────────────────────────────────────────────────────

CATEGORY_FN = {
    "next_token_prediction": evaluate_next_token,
    "exact_answer": evaluate_exact_answer,
    "multiple_choice": evaluate_multiple_choice,
    "numerical": evaluate_numerical,
}


# ─────────────────────────────────────────────────────────────
# Core Evaluation Runner
# ─────────────────────────────────────────────────────────────

def run_evaluation(
    model: CrachoLM,
    tokenizer: CrachoTokenizer,
    questions: List[Dict[str, Any]],
    device: torch.device,
    temperature: float = 0.1,    # low temperature for more deterministic eval
    top_k: int = 10,
) -> List[Dict[str, Any]]:
    """
    Runs all evaluation questions against the model.

    Args:
        model: Trained CrachoLM model (weights not modified).
        tokenizer: The same tokenizer used during training.
        questions: List of question dicts loaded from eval_questions.json.
        device: Compute device.
        temperature: Low value keeps generation near-deterministic for fair comparison.
        top_k: Top-K filtering to avoid pure random noise.

    Returns:
        List of per-question result dicts.
    """
    results = []
    model.eval()

    for q in questions:
        qid = q["id"]
        category = q["category"]
        prompt = q["prompt"]
        max_new_tokens = q.get("max_new_tokens", 10)
        case_sensitive = q.get("case_sensitive", False)
        choices = q.get("choices", [])

        # Determine the expected answer field name by category
        if category == "next_token_prediction":
            expected = q.get("expected_continuation", "")
        else:
            expected = q.get("expected_answer", "")

        # Generate model output using existing generate_text function (no modification)
        generated = generate_text(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            greedy=False,
            device=device,
        )

        # Evaluate correctness using category-specific function
        eval_fn = CATEGORY_FN.get(category)
        if eval_fn is None:
            correct = False
        elif category == "multiple_choice":
            correct = eval_fn(generated, prompt, expected, choices, case_sensitive)
        else:
            correct = eval_fn(generated, prompt, expected, case_sensitive)

        # Extract new text (after prompt) for display
        model_answer = generated[len(prompt):].strip() if generated.startswith(prompt) else generated.strip()

        results.append({
            "id": qid,
            "category": category,
            "prompt": prompt,
            "expected": expected,
            "model_answer": model_answer[:80],   # truncate very long outputs for display
            "full_generated": generated[:160],
            "correct": correct,
        })

    return results


# ─────────────────────────────────────────────────────────────
# Metrics Summary
# ─────────────────────────────────────────────────────────────

def compute_metrics(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Computes overall and per-category accuracy metrics."""
    total = len(results)
    correct_count = sum(1 for r in results if r["correct"])
    incorrect_count = total - correct_count
    accuracy = correct_count / total if total > 0 else 0.0

    # Per-category breakdown
    categories: Dict[str, Dict[str, int]] = {}
    for r in results:
        cat = r["category"]
        if cat not in categories:
            categories[cat] = {"total": 0, "correct": 0}
        categories[cat]["total"] += 1
        if r["correct"]:
            categories[cat]["correct"] += 1

    per_category_accuracy = {
        cat: {
            "total": v["total"],
            "correct": v["correct"],
            "accuracy": round(v["correct"] / v["total"], 4) if v["total"] > 0 else 0.0,
        }
        for cat, v in categories.items()
    }

    return {
        "total_questions": total,
        "correct_answers": correct_count,
        "incorrect_answers": incorrect_count,
        "exact_match_accuracy": round(accuracy, 4),
        "per_category": per_category_accuracy,
    }


# ─────────────────────────────────────────────────────────────
# Output Persistence
# ─────────────────────────────────────────────────────────────

def save_results_json(results: List[Dict], metrics: Dict, meta: Dict, output_path: str):
    """Saves complete evaluation results and metrics to a JSON file."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    payload = {
        "metadata": meta,
        "metrics": metrics,
        "results": results,
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    print(f"[✓] JSON results saved to: {output_path}")


def save_results_csv(results: List[Dict], output_path: str):
    """Saves per-question evaluation results to a CSV file."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fieldnames = ["id", "category", "prompt", "expected", "model_answer", "correct"]
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(results)
    print(f"[✓] CSV results saved to: {output_path}")


# ─────────────────────────────────────────────────────────────
# Data Leakage Verification
# ─────────────────────────────────────────────────────────────

def verify_no_data_leakage(eval_questions_path: str, training_data_path: str) -> bool:
    """
    Verifies the evaluation question file is not the same file as the training dataset.
    This is a structural sanity check — not an exhaustive search.

    Returns:
        True if no obvious leakage detected, False if files are the same.
    """
    abs_eval = os.path.abspath(eval_questions_path)
    abs_train = os.path.abspath(training_data_path)
    if abs_eval == abs_train:
        print("[X] DATA LEAKAGE DETECTED: Evaluation file path matches training data path!")
        return False
    print(f"[✓] Data Leakage Check PASSED: eval file and training file are separate paths.")
    return True
