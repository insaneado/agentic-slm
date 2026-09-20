"""
TinyStories data pipeline.

Tokenizes the dataset once with the GPT-2 BPE vocabulary and writes flat
uint16 binaries (`train.bin`, `validation.bin`) that are memory-mapped during
training, so the corpus never has to fit in RAM.
"""

import os

import numpy as np

DEFAULT_DATA_DIR = "data"
SPLIT_FILES = {"train": "train.bin", "val": "validation.bin"}


def prepare(data_dir: str = DEFAULT_DATA_DIR, num_proc: int = 8, force: bool = False):
    """Download TinyStories, BPE-tokenize it, and write the .bin shards.

    Skips all work if the binaries already exist, unless `force` is set.
    """
    import tiktoken
    from datasets import load_dataset
    from tqdm.auto import tqdm

    os.makedirs(data_dir, exist_ok=True)
    train_path = os.path.join(data_dir, SPLIT_FILES["train"])

    if os.path.exists(train_path) and not force:
        print(f"{train_path} already exists - skipping tokenization.")
        return data_dir

    if force:
        for fname in SPLIT_FILES.values():
            p = os.path.join(data_dir, fname)
            if os.path.exists(p):
                os.remove(p)
                print(f"Deleted old {p}")

    print("Loading dataset...")
    ds = load_dataset("roneneldan/TinyStories")

    enc = tiktoken.get_encoding("gpt2")

    def process(example):
        ids = enc.encode_ordinary(example["text"])
        ids.append(enc.eot_token)
        return {"ids": ids, "len": len(ids)}

    print(f"Tokenizing dataset... (num_proc={num_proc})")
    tokenized = ds.map(
        process,
        remove_columns=["text"],
        desc="tokenizing the splits",
        num_proc=num_proc,
    )

    for split, dset in tokenized.items():
        arr_len = np.sum(dset["len"], dtype=np.uint64)
        filename = os.path.join(data_dir, f"{split}.bin")
        # uint16 is enough: the GPT-2 vocabulary is 50257 < 65535.
        arr = np.memmap(filename, dtype=np.uint16, mode="w+", shape=(arr_len,))

        total_batches = 1024
        idx = 0
        for batch_idx in tqdm(range(total_batches), desc=f"writing {filename}"):
            batch = dset.shard(
                num_shards=total_batches, index=batch_idx, contiguous=True
            ).with_format("numpy")
            arr_batch = np.concatenate(batch["ids"])
            arr[idx: idx + len(arr_batch)] = arr_batch
            idx += len(arr_batch)

        arr.flush()
        print(f"Saved {filename}")

    return data_dir


def get_batch(split, block_size, batch_size, device, data_dir=DEFAULT_DATA_DIR):
    """Sample one (x, y) batch of next-token prediction pairs from a .bin shard."""
    import torch

    fname = SPLIT_FILES["train"] if split == "train" else SPLIT_FILES["val"]
    path = os.path.join(data_dir, fname)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} not found. Run `python -m src.data` first to build it."
        )

    # Re-open per batch: memmap objects leak if kept across the training loop.
    data = np.memmap(path, dtype=np.uint16, mode="r")
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([torch.from_numpy(data[i:i + block_size].astype(np.int64)) for i in ix])
    y = torch.stack(
        [torch.from_numpy(data[i + 1:i + 1 + block_size].astype(np.int64)) for i in ix]
    )

    if str(device).startswith("cuda"):
        x = x.pin_memory().to(device, non_blocking=True)
        y = y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
    return x, y


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Build the TinyStories token shards.")
    ap.add_argument("--data-dir", default=DEFAULT_DATA_DIR)
    ap.add_argument("--num-proc", type=int, default=8)
    ap.add_argument("--force", action="store_true", help="re-tokenize even if shards exist")
    args = ap.parse_args()
    prepare(args.data_dir, args.num_proc, args.force)
