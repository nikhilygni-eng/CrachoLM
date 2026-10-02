# CrachoLM improvement — 28 September 2026

The current default is checkpoints_capability_mixed_20260928/best_model.pt,
with its original paired tokenizer. The live page is http://127.0.0.1:7861.

## What changed

- Continued training CrachoLM's own 72.58M weights: 1,600 focused curriculum steps, then 1,200 mixed steps.
- The focused pass used 6,684 unique prompt examples. The mixed pass used 17,919 records, including grammar variants, reading questions with distractors, short explanations, existing Dolly examples, and existing TinyStories training chunks.
- Chat prompts and padding are excluded from the supervised loss; story replay trains all ordinary continuation targets. Category sampling and per-example loss prevent short answer tasks from being overwhelmed by longer answers.
- An initial candidate improved structured tasks but damaged story fluency. It was not promoted. The mixed pass restored the story-validation loss.
- The app and CLI now offer a local arithmetic tool. Calculator output is explicitly labelled; it is not a neural-model answer. Turn off Use local calculator, or launch chat.py --no-tools, to inspect the model alone.
- The page defaults to deterministic replies. The CPU thread count for app and CLI inference is limited to two. Original checkpoint directories remain intact.
- Antigravity was opened on the project, and a separate terminal displayed actual progress from the running jobs. This connection does not offer an interactive mouse cursor.

## Results on a small reserved task set

All numbers below measure raw model inference, with the calculator bypassed.
Both checkpoints answered the same final task set with greedy decoding.
Exact match ignores letter case, surrounding whitespace, and final sentence punctuation.
This is a narrow curriculum check, not a general intelligence or public benchmark.

| Task | Previous chat model | Final mixed model |
| --- | ---: | ---: |
| Sentence correction | 0/16 | 16/16 |
| Reading facts in the tested templates | 0/16 | 16/16 |
| New wording for taught explanations | 0/16 | 13/16 |
| Number comparisons | 0/12 | 8/12 |
| Arithmetic | 1/16 | 1/16 |
| Greeting exact wording | 2/3 | 1/3 |

Grammar and reading groups are excluded from training in this curriculum.
Explanation questions reuse taught facts with new wording.
Some simple facts and arithmetic occurred in earlier checkpoints.
The diagnostic review influenced the second pass and is not a blind test.
Strict greeting matching counts alternate wording as a failure, but one actual
remaining mistake is answering Good afternoon! with a goodbye.

Story-validation loss on the same 32 seeded chunks (8,192 target tokens):
2.4602 before, 1.7934 after. The rejected first candidate had loss 6.2924.
Lower loss does not guarantee sensible stories: greedy continuations still
repeat phrases and can attribute feelings to a ball.

## Verified live examples

- hi -> Hi! How can I help you?
- Fix the grammar: He go to work every day. -> He goes to work every day.
- Correct this sentence: They is reading books. -> They are reading books.
- Correct my English: I has two pen. -> I have two pens.
- What is gravity? -> Gravity is the attraction between objects with mass. It keeps us on Earth and planets in orbit.
- What is 17 + 24? -> 41. (labelled Local calculator)
- What is 7 + 8? -> 15. (labelled Local calculator)

With the calculator disabled, the model answered 7 + 8 as 16.
The calculator supports direct +, -, *, /, parentheses, and simple number
comparisons. It does not solve arbitrary word problems. Decimal arithmetic
uses 28-digit precision. Unsupported or undefined arithmetic is reported
without executing arbitrary input.

## Remaining limitations

Unfamiliar questions remain unreliable. For example, the final raw model
gave the wrong hat color when the question used a different name/distractor
combination, failed a past-tense rewrite, and gave a false description of
Bengaluru. It also failed arithmetic word problems.

The tokenizer discards original whitespace and indentation, so reliable
multi-line code generation has not been established. Context is limited to
256 tokens. The app still supplies a single turn rather than conversation
history. Do not describe the results above as general assistant intelligence.

## Checks and reproducibility

- 30 targeted unit tests passed.
- 10 exact live API checks passed, plus explanation, calculator-off, completion,
  invalid-input, and page-control checks.
- Python compilation, inline JavaScript syntax, and git diff --check passed.
- Data provenance and licences: DATA_SOURCES.md.
- Training code: improve_capabilities.py and capability_curriculum.py.
- Independent review code: review_capabilities.py.
- Each candidate directory retains manifests, train/dev/test records, raw
  answer files, best_model.pt, and last_training.pt with optimizer/scaler state.
- The first pass's source files were preserved under
  runs/capability_20260928_163315/phase1_source.
- Full before/after evidence:
  runs/capability_20260928_163315/final_review.json.
- Live evidence:
  runs/capability_20260928_163315/live_api_checks.json.
- Original application files:
  .quality_backups/capability_20260928_163315/.

## Run

From /home/escanor/Downloads/CrachoLM:

    .venv/bin/python app.py 7861
    .venv/bin/python chat.py

For the previous model, pass:

    --checkpoint checkpoints_chat_turns_20260927/best_model.pt

The current server was started with the selected checkpoint explicitly and
listens only on 127.0.0.1.

## User-reported essay failure (17:03, 28 September)

The population essay request returned an unrelated odd-number definition.
A correctly spelled essay request also failed. This confirms that the small
curriculum scores do not establish general instruction following or essay writing.
The issue is unresolved in the model. No single-answer training patch was applied.
The stale browser interface was separately corrected with no-store headers and
a fresh page URL. That interface fix does not improve neural-model quality.
