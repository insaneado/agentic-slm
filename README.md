# Agentic SLM: The Story Director

![Python](https://img.shields.io/badge/python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![Tests](https://img.shields.io/badge/tests-10%20passing-success?style=for-the-badge)

A small decoder-only Transformer (30.0M parameters total, 10.6M in the
transformer blocks), trained from scratch on TinyStories,
wrapped in a **Plan / Act / Observe / Reflect** agent loop that makes it follow
instructions it cannot follow on its own.

Small models are cheap but stubborn. Asked for *"a story about a dog that
contains the word sandwich"*, a model this size routinely ignores the
constraint. The usual answer is a bigger model. The answer here is a
deterministic control loop around a small one.

---

## The idea

```
PLAN     read the goal        -> prompt + constraint set
ACT      sample a draft       -> SLM generates
OBSERVE  validate the draft   -> deterministic string/logic checks
REFLECT  on failure, change   -> inject a new prompt strategy, retry
```

The observer is **not** a second language model. It is plain string matching and
logic checks, so validation is deterministic, free, and auditable. Every attempt
is recorded in a trace you can inspect after the run.

**Example.** Goal: a story about a dog that contains *sandwich*.

| Attempt | Draft | Observation | Reflection |
|---|---|---|---|
| 1 | "Once upon a time there was a dog. He liked to run in the park..." | FAIL - `contains:sandwich` | inject *"Suddenly, a huge sandwich appeared..."* |
| 2 | "...Suddenly, a huge sandwich appeared. The dog was so happy and ate it." | PASS | - |

---

## The model

Written out explicitly rather than assembled from `torch.nn.Transformer`, so
every part of the forward pass is inspectable.

| | |
|---|---|
| Architecture | Decoder-only Transformer (GPT-style) |
| Blocks | Custom `LayerNorm`, `CausalSelfAttention`, `MLP`, pre-norm residual `Block` |
| Parameters | 30.0M total - 10.6M transformer blocks + 19.3M tied GPT-2 embedding table |
| Context length | 128 tokens |
| Tokenizer | GPT-2 BPE via `tiktoken` (50257 vocab) |
| Training data | [TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories) |
| Training | AdamW, linear warmup into cosine decay, grad accumulation, AMP |

Attention uses PyTorch's fused `scaled_dot_product_attention` when available and
falls back to an explicit `softmax(QK^T / sqrt(d))V` with a causal mask.

> **On the parameter count.** With a 50257-token GPT-2 vocabulary at
> `n_embd=384`, the tied embedding table is ~19.3M parameters on its own - the
> majority of the model. The transformer stack that the depth and width choices
> actually control is 10.6M. `GPT.parameter_breakdown()` prints the split.


---

## Layout

```
src/
├── model.py      # LayerNorm, CausalSelfAttention, MLP, Block, GPTConfig, GPT
├── data.py       # TinyStories -> GPT-2 BPE -> memory-mapped uint16 shards
├── train.py      # training loop (warmup + cosine, grad accum, AMP, checkpointing)
└── agent.py      # Constraint engine + StoryDirectorAgent (PLAN/ACT/OBSERVE/REFLECT)
tests/
└── test_agent.py # 10 tests covering constraints and loop control flow
notebooks/
└── agentic_slm.ipynb   # the original end-to-end exploration, with outputs
demo.py           # run the agent from the command line
```

---

## Quickstart

```bash
git clone https://github.com/insaneado/agentic-slm.git
cd agentic-slm
pip install -r requirements.txt
```

**Run the tests** (no GPU, no checkpoint, no dataset needed):

```bash
pytest tests/ -q
```

The agent loop is tested against a scripted fake model, so the control flow is
verifiable without training anything.

**Run the agent** - the trained weights download automatically on first run:

```bash
python demo.py --prompt "Once upon a time there was a dog" --word sandwich
```

The checkpoint (~115MB) is published as a
[release asset](https://github.com/insaneado/agentic-slm/releases/tag/v1.0)
rather than committed, because it exceeds GitHub's 100MB per-file limit for
tracked files. Pass `--no-download` to skip the fetch and run on an untrained
model (the loop still works; the prose is gibberish).

**Or train it yourself** (needs a GPU for the full 20k steps):

```bash
python -m src.data      # download + tokenize TinyStories into data/*.bin
python -m src.train     # writes best_model_params.pt
```

Actual output from the trained checkpoint:

```
Loaded checkpoint: best_model_params.pt
[PLAN]  goal: Once upon a time there was a dog
        constraint: output must contain the word 'sandwich'
[ACT]    attempt 1: Once upon a time there was a dog named Max. Max was very curious...
[OBSERVE] failed: contains:sandwich
[REFLECT] switching tactic -> '...d down and saw a sandwich on the ground.'
[ACT]    attempt 2: Once upon a time there was a dog They looked down and saw a
                    sandwich on the ground. The boy was so excited! He quickly
                    went up to the swing and said: "Look, I found a sandwich!"
[OBSERVE] all 1 constraint(s) satisfied
success=True after 2 attempt(s)
```

Constraints compose - `demo.py --word sandwich --min-words 60` requires both.

---

## Adding a constraint

A constraint is a name plus a predicate over the generated text:

```python
from src.agent import Constraint, StoryDirectorAgent, contains_word

no_violence = Constraint(
    name="no_violence",
    check=lambda text: not any(w in text.lower() for w in ("hit", "fight")),
    describe="story must stay gentle",
)

result = agent.run("Once upon a time", constraints=[contains_word("balloon"), no_violence])
print(result.success, result.n_attempts)
```

Built in: `contains_word`, `excludes_word`, `min_words`.

---

## Limitations

- The observer is deterministic string matching, so it checks *presence*, not
  *meaning* - a story can satisfy `contains:sandwich` without the sandwich
  mattering to the plot.
- Reflection draws from a fixed list of prompt-injection strategies rather than
  reasoning about *why* an attempt failed.
- At this scale coherence degrades past roughly the 128-token context.

Natural next steps: swap the string observer for a small critic model, and
generate strategies dynamically from the failure reason.

---

## Credits

- Architecture follows *Attention Is All You Need* and Andrej Karpathy's nanoGPT.
- Dataset: TinyStories, Microsoft Research.

Built for the Consulting and Analytics Club, IIT Guwahati.
