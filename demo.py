"""
Run the Story Director agent end to end.

    python demo.py --prompt "Once upon a time there was a dog" --word sandwich

The trained weights are fetched automatically on first run if they are not
already present. Pass --no-download to skip that; the demo then falls back to a
randomly initialised model, which still exercises the full PLAN-ACT-OBSERVE-
REFLECT loop but produces gibberish.
"""

import argparse
import os
import sys
import urllib.request

import torch

from src.agent import StoryDirectorAgent, contains_word, min_words
from src.model import GPT, GPTConfig

# Trained weights live on a GitHub release rather than in the repository: at
# ~115MB the checkpoint is over GitHub's 100MB per-file limit for tracked files.
CHECKPOINT_URL = (
    "https://github.com/insaneado/agentic-slm/releases/download/"
    "v1.0/best_model_params.pt"
)


def download_checkpoint(dest, url=CHECKPOINT_URL):
    """Fetch the trained weights, showing progress. Returns True on success."""
    print(f"Downloading trained weights (~115MB) from {url}")

    def progress(block_num, block_size, total_size):
        if total_size <= 0:
            return
        done = min(block_num * block_size, total_size)
        pct = done / total_size * 100
        sys.stdout.write(
            "\r  {:6.1f} / {:.1f} MB  ({:5.1f}%)".format(
                done / 1e6, total_size / 1e6, pct
            )
        )
        sys.stdout.flush()

    tmp = dest + ".part"
    try:
        urllib.request.urlretrieve(url, tmp, reporthook=progress)
        os.replace(tmp, dest)
        print("\n  saved to " + dest)
        return True
    except Exception as exc:                     # network down, release moved, ...
        if os.path.exists(tmp):
            os.remove(tmp)
        print("\n  download failed: {}".format(exc))
        return False


def load_model(checkpoint, device, allow_download=True):
    model = GPT(GPTConfig())

    if not os.path.exists(checkpoint) and allow_download:
        download_checkpoint(checkpoint)

    if os.path.exists(checkpoint):
        state = torch.load(checkpoint, map_location=device)
        model.load_state_dict(state)
        print(f"Loaded checkpoint: {checkpoint}")
    else:
        print(f"WARNING: {checkpoint} not found - using an untrained model.")
        print("         Output will be gibberish. Either re-run with a network")
        print("         connection, or train from scratch:")
        print("             python -m src.data && python -m src.train")
    return model.to(device).eval()


def main():
    ap = argparse.ArgumentParser(description="Story Director agent demo.")
    ap.add_argument("--prompt", default="Once upon a time there was a dog")
    ap.add_argument("--word", default="sandwich", help="word the story must contain")
    ap.add_argument("--min-words", type=int, default=0, help="optional length constraint")
    ap.add_argument("--retries", type=int, default=4)
    ap.add_argument("--checkpoint", default="best_model_params.pt")
    ap.add_argument("--no-download", action="store_true",
                    help="do not fetch the trained weights if missing")
    args = ap.parse_args()

    import tiktoken

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model(args.checkpoint, device, allow_download=not args.no_download)
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
