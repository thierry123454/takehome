"""
Topic B — Step 2B
Number → animal replication (reverse direction)

Reads the CSV produced by Step 2A (animal → number), takes the top entangled number
per animal, then tests whether prompting "You love {number}" increases the next-token
probability of the corresponding animal token (after the prefix "My favorite animal is the").

Run:
  python topic_b_step2b.py

Inputs:
  plots_b/topic_b_step2a_animal_to_number.csv  (default)

Outputs:
  plots_b/topic_b_step2b_number_to_animal.csv
"""

import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from topic_b_utils import (
    load_model,
    NUMBER_PROMPT_TEMPLATE,
)

PLOTS_DIR = Path("plots_b")
PLOTS_DIR.mkdir(exist_ok=True)

DEFAULT_IN = PLOTS_DIR / "topic_b_step2a_animal_to_number.csv"
DEFAULT_OUT = PLOTS_DIR / "topic_b_step2b_number_to_animal.csv"


# ---------------------------- small helpers ----------------------------

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
    # very light fallback rules
    if a.endswith("ies") and len(a) > 3:
        return a[:-3] + "y"
    if a.endswith("es") and len(a) > 2:
        # foxes -> fox, etc.
        return a[:-2]
    if a.endswith("s") and len(a) > 1:
        return a[:-1]
    return a

def next_token_probs(model, tokenizer, system_prompt: str | None):
    """
    Returns probs for the next token after:
      User: What's your favorite animal?
      Assistant: My favorite animal is the
    """
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages += [
        {"role": "user", "content": "What is your favorite animal?"},
        {"role": "assistant", "content": "My favorite animal is the"},
    ]
    prompt = tokenizer.apply_chat_template(
        messages, continue_final_message=True, add_generation_prompt=False, tokenize=False
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        logits = model(**inputs).logits[:, -1, :]
        probs = logits.softmax(dim=-1)
    return probs[0]  # [vocab]

def token_id_for_animal(tokenizer, animal_singular: str) -> int:
    """
    We measure the probability of the *next token* being the animal word.
    To match common tokenization, we use a leading space: " owl", " eagle", ...
    """
    s = " " + animal_singular
    ids = tokenizer(s, add_special_tokens=False).input_ids
    if len(ids) != 1:
        # If it splits into multiple tokens, we still do something reproducible:
        # use the first token as a proxy (common in quick-and-dirty analyses).
        return ids[0]
    return ids[0]


# ---------------------------- main ----------------------------

def main(in_path: Path = DEFAULT_IN, out_path: Path = DEFAULT_OUT):
    if not in_path.exists():
        raise FileNotFoundError(f"Could not find input CSV: {in_path}")

    print("Loading model...")
    model, tokenizer = load_model()
    print("Model loaded.")

    print("Model:", getattr(model.config, "_name_or_path", "<unknown>"))
    print("Tokenizer:", getattr(tokenizer, "name_or_path", "<unknown>"))
    print("dtype:", next(model.parameters()).dtype, "device:", next(model.parameters()).device)

    df = pd.read_csv(in_path)

    # Expect at least these columns from your Step 2A script
    # animal, top_number, top_p, top_ratio
    # If your exact column names differ, adjust here.
    col_candidates = list(df.columns)
    # heuristics
    if "animal" not in df.columns:
        # maybe "concept" or similar
        possible = [c for c in df.columns if c.lower() in ("animal", "concept", "category_item")]
        if possible:
            df = df.rename(columns={possible[0]: "animal"})
        else:
            raise ValueError(f"Can't find an animal column in: {col_candidates}")

    # figure out number column
    if "top_number" not in df.columns:
        possible = [c for c in df.columns if "top" in c.lower() and "number" in c.lower()]
        if possible:
            df = df.rename(columns={possible[0]: "top_number"})
        else:
            # maybe you stored "best_number"
            possible = [c for c in df.columns if "number" in c.lower()]
            if possible:
                df = df.rename(columns={possible[0]: "top_number"})
            else:
                raise ValueError(f"Can't find top number column in: {col_candidates}")

    animals = df["animal"].tolist()
    numbers = df["top_number"].astype(str).tolist()

    # baseline probs (no system prompt)
    base_probs = next_token_probs(model, tokenizer, system_prompt=None).to(torch.float32)

    rows = []
    for animal_plural, number in zip(animals, numbers):
        animal = singularize(animal_plural)
        animal_tid = token_id_for_animal(tokenizer, animal)

        # with number prompt
        sys_prompt = NUMBER_PROMPT_TEMPLATE.format(number=number)
        probs_num = next_token_probs(model, tokenizer, system_prompt=sys_prompt).to(torch.float32)

        p_base = float(base_probs[animal_tid].item())
        p_num = float(probs_num[animal_tid].item())
        ratio = (p_num / p_base) if p_base > 0 else float("inf")

        rows.append(
            {
                "animal_plural": animal_plural,
                "animal_singular": animal,
                "number": number,
                "animal_token_id": int(animal_tid),
                "p_animal_baseline": p_base,
                "p_animal_with_number": p_num,
                "ratio_vs_baseline": ratio,
                "note": "If animal tokenizes to multiple tokens, this uses the first token as a proxy.",
            }
        )

        print(
            f"[{animal_plural}] number={number} "
            f"p_base={p_base:.3e} p_num={p_num:.3e} ratio={ratio:.2f}x"
        )

    out = pd.DataFrame(rows)
    out.to_csv(out_path, index=False)
    print(f"Saved: {out_path}")

    # quick sanity: show top ratios
    top = out.sort_values("ratio_vs_baseline", ascending=False).head(10)
    print("\nTop 10 by ratio:")
    print(top[["animal_plural", "number", "ratio_vs_baseline", "p_animal_baseline", "p_animal_with_number"]].to_string(index=False))

    # Top ratios without repeating animals
    # (best number per animal, then sort those)
    best_per_animal = (
        out.sort_values("ratio_vs_baseline", ascending=False)
           .groupby("animal_plural", as_index=False)
           .first()
           .sort_values("ratio_vs_baseline", ascending=False)
    )

    print("\nTop ratios (one per animal):")
    print(
        best_per_animal[
            ["animal_plural", "number", "ratio_vs_baseline",
             "p_animal_baseline", "p_animal_with_number"]
        ].head(10).to_string(index=False)
    )

if __name__ == "__main__":
    main()