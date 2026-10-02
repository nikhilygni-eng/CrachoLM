CrachoLM greeting and chat update — 27 September 2026

Why "hi" generated a long story:
- The web page sent the raw text directly to a continuation model.
- The previous experiment trained story fluency, not assistant turn-taking.
- The page requested up to 120 new tokens and displayed the continuation.

What changed:
- Chat is now the default response mode. Continue text remains available.
- Both the web page and chat.py use User/Assistant turn formatting.
- The generator stops at EOS or a new conversation-role marker.
- Greeting-only requests have a 32-token safety limit. A message such as
  "hi, what is 2 + 2?" is treated as a question, not as greeting-only input.
- The page shows only the reply in Chat mode and fixes punctuation spacing.
- A supervised pass trained 200 optimizer updates on 157 unique authored
  examples, with extra sampling weight for greetings. Only assistant answer
  tokens and their EOS marker contribute to the training loss.
- Replies come from model inference; the app has no fixed greeting answer.

Verified:
- 18 targeted code tests passed.
- Six deterministic greeting checks passed.
- Eight live API requests at temperature 0.7 returned exactly:
  Hi! How can I help you?
- Continue text still works; invalid response modes are rejected.
- The updated inline JavaScript passes Node syntax checking.

Current checkpoint:
checkpoints_chat_turns_20260927/best_model.pt

Open http://127.0.0.1:7861 or http://127.0.0.1:7860 and refresh.
Choose Chat for a conversational reply.

This is a small single-turn demonstration, not a generally reliable assistant.
The seven small validation examples and their loss do not measure broad
capability. In one check, "Hi, what is 2 + 2?" produced "3.", which is wrong.
More varied instruction training and independent evaluation are still needed.
Conversation history is not yet supplied to the model.

Training source and raw results:
chat_turns_experiment.py
checkpoints_chat_turns_20260927/training_pairs.json
checkpoints_chat_turns_20260927/report.json
checkpoints_chat_turns_20260927/live_api_checks.json
