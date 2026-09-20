"""
Run the Story Director agent end to end.

    python demo.py --prompt "Once upon a time there was a dog" --word sandwich

If `best_model_params.pt` is present it is loaded; otherwise the demo runs on a
randomly initialised model, which still exercises the full PLAN-ACT-OBSERVE-
REFLECT loop (it will simply fail its constraints and exhaust its retries).
"""

import argparse
import os

import torch

from src.agent import StoryDirectorAgent, contains_word, min_words
from src.model import GPT, GPTConfig


def load_model(checkpoint, device):
    model = GPT(GPTConfig())
    if os.path.exists(checkpoint):
        state = torch.load(checkpoint, map_location=device)
        model.load_state_dict(state)
        print(f"Loaded checkpoint: {checkpoint}")
    else:
        print(f"WARNING: {checkpoint} not found - using an untrained model.")
        print("         Train one first with:  python -m src.data && python -m src.train")
    return model.to(device).eval()


def main():
    ap = argparse.ArgumentParser(description="Story Director agent demo.")
    ap.add_argument("--prompt", default="Once upon a time there was a dog")
    ap.add_argument("--word", default="sandwich", help="word the story must contain")
    ap.add_argument("--min-words", type=int, default=0, help="optional length constraint")
    ap.add_argument("--retries", type=int, default=4)
    ap.add_argument("--checkpoint", default="best_model_params.pt")
    args = ap.parse_args()

    import tiktoken

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model(args.checkpoint, device)
    enc = tiktoken.get_encoding("gpt2")

    agent = StoryDirectorAgent(model, enc, device=device, verbose=True)

    constraints = [contains_word(args.word)]
    if args.min_words:
        constraints.append(min_words(args.min_words))

    print("=" * 60)
    result = agent.run(args.prompt, constraints=constraints, max_retries=args.retries)
    print("=" * 60)
    print(f"success={result.success} after {result.n_attempts} attempt(s)\n")
    print(result.text)


if __name__ == "__main__":
    main()
