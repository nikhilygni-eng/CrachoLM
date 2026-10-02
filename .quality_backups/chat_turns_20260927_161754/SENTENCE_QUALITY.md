CrachoLM sentence-quality work — 27 September 2026

The architecture checks pass. The original launch commands pointed at older
models, including a general checkpoint with just one optimizer update.
The working inference paths now use the paired v3 model and tokenizer.

I reproduced the v3 repetition and unrelated answers. Tests alone do not prove
that a language model writes good sentences. This experiment continues your
own 72.58M model on a smaller, simpler English domain to measure that directly.

What was added in this work:
- fluency_experiment.py: capped data preparation, separate validation, bounded
  continuation training, conditional candidate saving, and full comparisons.
- Tokenizer fingerprint verification for new candidate checkpoints.
- Regression checks for corpus boundaries, deduplication across splits, byte
  limits, and same-size tokenizer mismatches. All 13 targeted checks pass.

Data: 5,000 training and 500 validation stories from the official TinyStories
GPT-4 version, with source attribution and hashes in:
data_general/fluency_stories/manifest.json

Source: https://huggingface.co/datasets/roneneldan/TinyStories
Paper: https://arxiv.org/abs/2305.07759

Both data splits are deduplicated together before training. Punctuation is
normalized consistently, and the existing tokenizer reports 0 unknown tokens.
This is a study of basic English fluency, not a general knowledge evaluation.

The first 200-update trial improved loss on its held-out sample from
2.9842 to 1.8380, but the sampled text still repeated and had logical errors.
The second pass completed successfully: 800 further optimizer updates.
Total additional training: 1,000 updates and 4,096,000 token positions.

Full comparison on the same 141,824 validation targets:

| Checkpoint | Story validation loss | Perplexity |
| --- | ---: | ---: |
| Original v3 | 2.9767 | 19.62 |
| New fluency candidate | 1.6582 | 5.25 |

Lower is better for this next-token prediction test. This improvement is not
a grammar-accuracy percentage. The model forms simpler sentence structures,
but it still repeats, changes subjects illogically, and answers factual
questions incorrectly. It is not ready as a reliable general assistant.
The full comparison below includes all five fixed prompts, not only a
favorable example.

Candidate checkpoint:
/home/escanor/Downloads/CrachoLM/checkpoints_fluency_stage2_20260927/best_model.pt

Local test page: http://127.0.0.1:7861

To test the candidate from the project terminal:

```bash
.venv/bin/python generate_general.py --checkpoint checkpoints_fluency_stage2_20260927/best_model.pt --prompt "The dog saw a red ball and" --max-new-tokens 128
```

To restart the local test page:

```bash
.venv/bin/python app.py 7861 --checkpoint checkpoints_fluency_stage2_20260927/best_model.pt
```

Next work: broaden clean English training data, check repetition and
consistency on fresh prompts, then train on carefully prepared question and
answer examples with the same tokenizer. Story fluency alone does not teach
a model to answer general questions. The 256-token context and tokenizer
whitespace normalization also limit output quality. A token limit can stop
generation in the middle of a word or sentence.

The candidate contains inference weights. To continue it, use the
fluency_experiment.py train --checkpoint option; its optimizer starts a new
bounded training run.

Reproduce the final comparison using a new output filename:

```bash
.venv/bin/python fluency_experiment.py compare --candidate checkpoints_fluency_stage2_20260927/best_model.pt --output logs/fluency_comparison_repeat.json
```

Unedited samples, same prompts and sampling settings (seed 42,
temperature 0.7, top-k 40, maximum 128 new tokens):

Prompt: One day, a little girl went to the park. She

Before:

One day, a little girl went to the park. She flew into the bench , and was already given a sigh , but she was a friend of the couple ’ s shop . Suddenly , she was in a foolish and hung by a small girl and was already a couple of guns . She was already a young girl , a child of blood and a girl , a small child . She had two or more maize rice which was in any way of coming on her body . She gave her a

After:

One day, a little girl went to the park. She saw a big tree with lots of leaves . The tree was very happy to see the leaves because it was dry . Suddenly , a big bear came and asked , " Do you want to be friends ? " The bear said , " Yes , I can help you , too ! " The girl was very excited and showed the leaves to her friends . Every day , the girl could go to a park with her friends . They would run , jump ,

Prompt: Tom was hungry, so he

Before:

Tom was hungry, so he felt that it was not very real that he would have been a young man who had been in a more existing console to him . ” “ So much the better , ” replied Porthos , “ I do not think that it was a great reason to do so . ” “ What about the matter , ” replied D ’ Artagnan , “ I have no pleasure to do with it . Every one has the real passion . ” “ No , and I do not know what is now , ” repli

After:

Tom was hungry, so he flew away with a big , scary bird . Tom was scared and cried . " I am sorry , Tom . I did not like this . You are not a bad bird . " Tom felt sad and said , " Yes , it is a bad bird . You should have stayed hungry . " Tom said , " Thank you , Tom . You are a good bird . You are a good friend . " Tom learned a lesson . He was not a bad fri

Prompt: The dog saw a red ball and

Before:

The dog saw a red ball and two @ - @ point shaft . Heaven ' s caught on the track with a half @ - @ point field with a caught grip on the field , with a caught grip and two tortured rider . The next week , the grid was held for a tank , with a rush of 14 yards ( 24 m ) . Heaven ' s pushing was to be used for the grid . The grid opened fire on the caught , with a cau

After:

The dog saw a red ball and wanted to play with it . The dog was sad and played with it . The dog was happy that the dog was not jealous anymore .

Prompt: After the rain stopped, Mira

Before:

After the rain stopped, Mira , and his wife returned to the Palestinian Empire on 18 November . They lost the Crown Fire to the Crown Fire on 16 November , aging in a fox to the 11th Crown Fox and then set out in a crowd of arms . = = Empire = = The Crown Fire did not meet any of the Palestinians , and the Empire had the following member of the Palestinian Empire . Miral was pregnant a

After:

After the rain stopped, Mira flew away with her . She saw the bird and flew away . She flew to the dog and ran to get back to her paw . Mira and her paw were scared . Mira and her paw were not scared anymore . They ran to the dog and ran away . Mira and her paw were shocked and wet . Mira and her paws were not mad . They did not want to share the bird . They were sad and scared

Prompt: User: What is machine learning?
Assistant:

Before:

User: What is machine learning?
Assistant: People enjoy learning skills with learning in learning , learning from learning a learning and learning .

After:

User: What is machine learning?
Assistant: Machines learning is that important things can be bad and good .
