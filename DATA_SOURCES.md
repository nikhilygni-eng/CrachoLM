# Training data used on 28 September 2026

The model continues CrachoLM's own trained weights. No pretrained model, Ollama model, or external inference API is used.

## Authored curriculum

capability_curriculum.py contains AI-authored English examples and deterministic task generators. It covers subject/verb agreement, preservation of sentence details, short explanations, reading facts with distractors, basic arithmetic, and conversational replies. Numeric training answers are calculated by the data generator. They are not looked up during neural-network inference.

Grammar, arithmetic, comparison, and reading groups have deterministic splits. Explanation evaluation uses new question wording for facts taught in training; it does not measure unseen knowledge. Prior checkpoints already saw some simple facts and sums. The review prompts influenced the design of the second pass, so those diagnostic prompts are not a blind benchmark.

## Databricks Dolly-15k

Source: https://huggingface.co/datasets/databricks/databricks-dolly-15k
Dataset card: https://huggingface.co/datasets/databricks/databricks-dolly-15k/blob/main/README.md

Copyright (2023) Databricks, Inc. License: Creative Commons Attribution-ShareAlike 3.0 Unported (CC BY-SA 3.0).
Some records include Wikipedia material, copyright Wikipedia editors and contributors, under CC BY-SA 3.0.

The mixed pass uses a filtered subset of the pre-existing local corpus at data_general/raw_scaled/03_conversations_qa.txt: no external context, complete short answers, ASCII coverage, and a 192-token example limit. Dataset text is used as training data only. The training outputs retain attribution here and record provenance in the dataset manifests. Retain the source license when redistributing adapted dataset records. Filtering does not establish that every factual claim is correct. These records may have appeared in the older base model's pretraining.

## TinyStories

Source: https://huggingface.co/datasets/roneneldan/TinyStories
Authors: Ronen Eldan and Yuanzhi Li.
Paper: https://arxiv.org/abs/2305.07759

The existing local data manifest lists CDLA-Sharing-1.0. See data_general/fluency_stories/manifest.json for original download URLs, dates, hashes, and the 5,000 training / 500 validation story selection.

The mixed pass replays token chunks only from the existing training split. Story validation uses a fixed seed to choose 32 chunks from the validation split. This limited sample is reported as a regression check, not a full benchmark.

## Local calculator

src/local_calculator.py is a separate tool for direct arithmetic and number comparisons. The web page and CLI label its outputs. Neural-model evaluations call generate_chat_reply directly, bypassing this tool, and therefore do not credit calculator results to the model.
