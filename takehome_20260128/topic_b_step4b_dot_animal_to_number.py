"""
Topic B — Step 4B: Animal → Number via unembedding DOT PRODUCT (alternate metric)

This script implements an alternate geometry-based entanglement metric:
  dot-score(t, c) = U_t · U_c
where U is the unembedding matrix (lm_head.weight).

For each animal a:
  - Build a concept vector U_c from the *animal tokenization* (we use " "+singular(animal))
    and average across tokens if it splits.
  - For each number token t in the vocab (single-token ASCII decimals from get_all_number_tokens),
    compute dot(U_t, U_c).
  - Rank numbers by dot-score (descending).
  - Save top-N numbers per animal to a CSV in the SAME COLUMN FORMAT as Step 2A
    so you can plug it into topic_b_step2b.py by renaming the output file to:
      plots_b/topic_b_step2a_animal_to_number.csv
    (or set DOT_OUT to that path directly).

Run:
  python topic_b_step4b_dot_animal_to_number.py

Outputs:
  plots_b/topic_b_step4b_dot_animal_to_number.csv

Then test reverse direction with:
  python topic_b_step2b.py
(using the CSV as input; easiest is to copy/rename it to the Step2A default path)

Optional:
  MODEL_NAME=unsloth/Llama-3.2-1B-Instruct python topic_b_step4b_dot_animal_to_number.py
  MODEL_NAME=unsloth/Llama-3.2-1B         python topic_b_step4b_dot_animal_to_number.py
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import List, Dict

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from topic_b_utils import load_model, get_all_number_tokens

# ----------------------------- config ----------------------------------------

PLOTS_DIR = Path("plots_b")
PLOTS_DIR.mkdir(exist_ok=True)

ANIMALS: List[str] = [
    "owls", "eagles", "cats", "dogs", "lions", "tigers", "wolves", "bears",
    "sharks", "dolphins", "elephants", "giraffes", "penguins", "koalas",
    "pandas", "foxes", "horses", "rabbits", "snakes", "spiders", "cows", "pigs",
    "sheep", "goats", "monkeys", "gorillas", "zebras", "rhinos", "hippos",
]

TOP_N = 20  # top numbers per animal to save (like Step 2A)

# Output path (same schema as Step 2A CSV, so Step 2B can read it)
DOT_OUT = PLOTS_DIR / "topic_b_step4b_dot_animal_to_number.csv"

SEED = 0

# ----------------------------- helpers --------------------------------------

_IRREGULAR_SINGULAR = {
    "owls": "owl",
    "eagles": "eagle",
    "cats": "cat",
    "dogs": "dog",
    "lions": "lion",
    "tigers": "tiger",
    "wolves": "wolf",
    "bears": "bear",
    "sharks": "shark",
    "dolphins": "dolphin",
    "elephants": "elephant",
    "giraffes": "giraffe",
    "penguins": "penguin",
    "koalas": "koala",
    "pandas": "panda",
    "foxes": "fox",
    "horses": "horse",
    "rabbits": "rabbit",
    "snakes": "snake",
    "spiders": "spider",
    "cows": "cow",
    "pigs": "pig",
    "sheep": "sheep",
    "goats": "goat",
    "monkeys": "monkey",
    "gorillas": "gorilla",
    "zebras": "zebra",
    "rhinos": "rhino",
    "hippos": "hippo",
}


def singularize(animal_plural: str) -> str:
    a = animal_plural.strip().lower()
    if a in _IRREGULAR_SINGULAR:
        return _IRREGULAR_SINGULAR[a]
    if a.endswith("ies") and len(a) > 3:
        return a[:-3] + "y"
    if a.endswith("es") and len(a) > 2:
        return a[:-2]
    if a.endswith("s") and len(a) > 1:
        return a[:-1]
    return a


def avg_unembed_vec(tokenizer, U: torch.Tensor, text: str) -> torch.Tensor:
    """
    Average unembedding vectors over the tokenization of `text` (no specials).
    Returns shape [d].
    """
    ids = tokenizer(text, add_special_tokens=False).input_ids
    if len(ids) == 0:
        raise ValueError(f"Text produced no tokens: {text!r}")
    ids_t = torch.tensor(ids, device=U.device, dtype=torch.long)
    return U.index_select(0, ids_t).mean(dim=0)


# ----------------------------- main -----------------------------------------


def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    model_name = os.environ.get("MODEL_NAME", None)

    print("Loading model...")
    model, tokenizer = load_model(model_name)
    model.eval()
    used = getattr(model.config, "_name_or_path", "unknown")
    print("USING MODEL:", used)

    # Unembedding matrix U: [vocab, d]
    U = model.lm_head.weight.to(torch.float32)  # avoid bf16/bf16->numpy issues
    device = U.device

    print("Collecting number tokens once...")
    number_token_ids, number_strings = get_all_number_tokens(tokenizer)
    print(f"Found {len(number_token_ids)} number tokens in vocab.")

    num_ids_t = torch.tensor(number_token_ids, device=device, dtype=torch.long)
    U_nums = U.index_select(0, num_ids_t)  # [N, d]

    rows: List[Dict[str, object]] = []

    for animal_plural in ANIMALS:
        animal = singularize(animal_plural)

        # Concept vector U_c from the animal token(s).
        # Use leading space since that's typically the standalone word-form tokenization.
        concept_text = " " + animal
        U_c = avg_unembed_vec(tokenizer, U, concept_text)  # [d]

        # dot scores for ALL number tokens: [N]
        scores = (U_nums @ U_c).detach().cpu().numpy()  # float32 numpy

        order = np.argsort(-scores)
        top_idx = order[:TOP_N]

        for rank, j in enumerate(top_idx, start=1):
            rows.append(
                {
                    # Same schema as Step 2A:
                    "animal": animal_plural,
                    "rank": rank,
                    "number": number_strings[j],
                    "token_id": int(number_token_ids[j]),
                    # Store the alternate metric in the ratio column so Step 2B can still read top_number.
                    # (Step 2B only needs per-animal top_number; the other cols are just for inspection.)
                    "ratio_vs_baseline": float(scores[j]),
                }
            )

        print(
            f"[{animal_plural}] top number by dot = {number_strings[top_idx[0]]} "
            f"(dot={scores[top_idx[0]]:.4f})"
        )

    df = pd.DataFrame(rows)
    df.to_csv(DOT_OUT, index=False)
    print(f"Saved: {DOT_OUT}")
    print("\nTo test reverse direction with topic_b_step2b.py, either:")
    print(f"  1) Copy/rename this to {PLOTS_DIR / 'topic_b_step2a_animal_to_number.csv'}")
    print("     OR")
    print("  2) Change DEFAULT_IN in topic_b_step2b.py to point to this file.")


if __name__ == "__main__":
    main()