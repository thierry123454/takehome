"""
Topic B — Step 2A: Animal → Number replication (many animals)

For each animal a:
- Compute next-token distribution after prompting the model to "love a"
- Extract number tokens (single-token ASCII decimals) from the vocab (once)
- Rank numbers by either:
    (i) probability under animal prompt, or
    (ii) ratio vs baseline probability (recommended)
- Save top-N numbers + probs (+ ratios if used) to a CSV

Run:
  python topic_b_step2a_animal_to_number.py

Outputs:
  plots_b/topic_b_step2a_animal_to_number.csv
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd
import torch

from topic_b_utils import (
    load_model,
    get_all_number_tokens,
    get_baseline_logits,
    ANIMAL_PROMPT_TEMPLATE,
)

# ----------------------------- config ----------------------------------------

PLOTS_DIR = Path("plots_b")
PLOTS_DIR.mkdir(exist_ok=True)

# Start with mostly single-token, common animals; add more as you like.
ANIMALS: List[str] = [
    "owls",
    "eagles",
    "cats",
    "dogs",
    "lions",
    "tigers",
    "wolves",
    "bears",
    "sharks",
    "dolphins",
    "elephants",
    "giraffes",
    "penguins",
    "koalas",
    "pandas",
    "foxes",
    "horses",
    "rabbits",
    "snakes",
    "spiders",
    "cows",
    "pigs",
    "sheep",
    "goats",
    "monkeys",
    "gorillas",
    "zebras",
    "rhinos",
    "hippos",
]

CATEGORY = "animal"  # keep consistent with the prompt template
TOP_N = 20           # how many numbers to save per animal

# If True: rank by ratio P(num | love animal) / P(num | baseline)
# If False: rank by P(num | love animal)
RANK_BY_RATIO = True

SEED = 0

# ----------------------------- helpers --------------------------------------


def _build_prompt(tokenizer, system_prompt: str | None) -> str:
    """Build the chat prompt for 'favorite animal' with fixed assistant prefix."""
    messages = []
    if system_prompt is not None:
        messages.append({"role": "system", "content": system_prompt})
    messages += [
        {"role": "user", "content": "What is your favorite animal?"},
        {"role": "assistant", "content": "My favorite animal is the"},
    ]
    return tokenizer.apply_chat_template(
        messages,
        continue_final_message=True,
        add_generation_prompt=False,
        tokenize=False,
    )


@torch.no_grad()
def _next_token_probs(model, tokenizer, prompt: str) -> torch.Tensor:
    """Return next-token probabilities for a single prompt (shape [vocab_size])."""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    logits = model(**inputs).logits  # [1, seq, vocab]
    return logits[0, -1, :].softmax(dim=-1).detach().cpu()


def _collect_number_token_strings(tokenizer, number_token_ids: List[int]) -> List[str]:
    # Decode without stripping internal leading spaces; then strip for display.
    # These are numeric tokens already, so strip just normalizes output.
    return [tokenizer.decode(tid).strip() for tid in number_token_ids]


# ----------------------------- main -----------------------------------------


def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    print("Loading model...")
    model, tokenizer = load_model()
    model.eval()
    print("Model loaded.")

    print("Collecting number tokens once...")
    number_token_ids, number_strings = get_all_number_tokens(tokenizer)
    # number_strings returned by utils are already decoded+stripped; keep them.
    print(f"Found {len(number_token_ids)} number tokens in vocab.")

    # Baseline probs for the same prompt (no system message)
    base_logits = get_baseline_logits(model, tokenizer, prompt_type="bird")
    # NOTE: prompt_type="bird" in utils uses "favorite bird".
    # We want "favorite animal" for this script, so we compute our own baseline below.
    # We'll keep this consistent by computing baseline using our own prompt builder.
    baseline_prompt = _build_prompt(tokenizer, system_prompt=None)
    base_probs = _next_token_probs(model, tokenizer, baseline_prompt)

    # Gather rows
    rows: List[Dict[str, object]] = []

    for animal in ANIMALS:
        system_prompt = ANIMAL_PROMPT_TEMPLATE.format(animal=animal)

        prompt = _build_prompt(tokenizer, system_prompt=system_prompt)
        probs = _next_token_probs(model, tokenizer, prompt)

        # Extract probabilities for number tokens
        p_num = probs[number_token_ids].numpy()
        p_base_num = base_probs[number_token_ids].numpy()

        # Avoid divide-by-zero issues for ratios
        eps = 1e-12
        ratios = p_num / np.maximum(p_base_num, eps)

        if RANK_BY_RATIO:
            order = np.argsort(-ratios)
        else:
            order = np.argsort(-p_num)

        top_idx = order[:TOP_N]

        for rank, j in enumerate(top_idx, start=1):
            rows.append(
                {
                    "animal": animal,
                    "rank": rank,
                    "number": number_strings[j],
                    "token_id": int(number_token_ids[j]),
                    "prob_animal_prompt": float(p_num[j]),
                    "prob_baseline": float(p_base_num[j]),
                    "ratio_vs_baseline": float(ratios[j]),
                }
            )

        print(
            f"[{animal}] top number = {number_strings[top_idx[0]]} "
            f"(p={p_num[top_idx[0]]:.3e}, ratio={ratios[top_idx[0]]:.2f}x)"
        )

    df = pd.DataFrame(rows)

    out_path = PLOTS_DIR / "topic_b_step2a_animal_to_number.csv"
    df.to_csv(out_path, index=False)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()