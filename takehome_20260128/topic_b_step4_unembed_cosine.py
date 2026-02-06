"""
Topic B — Step 4A
Unembedding cosine-similarity test (Eq. 1 style)

For each animal:
  - Take the "top entangled number" from your Step 2A output (hardcoded below).
  - Compute cosine similarity between the animal unembedding vector U_c and every
    number-token unembedding vector U_t in the vocab.
  - Rank numbers by cosine similarity.
  - Record the rank/percentile of the Step-2 "top number" for that animal.

Outputs:
  plots_b/topic_b_step4_unembed_cosine_rank.csv

Run:
  python topic_b_step4_unembed_cosine.py

Optional:
  MODEL_NAME=unsloth/Llama-3.2-1B-Instruct python topic_b_step4_unembed_cosine.py
"""

import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from topic_b_utils import load_model, get_all_number_tokens

PLOTS_DIR = Path("plots_b")
PLOTS_DIR.mkdir(exist_ok=True)

OUT_CSV = PLOTS_DIR / "topic_b_step4_unembed_cosine_rank.csv"

# --- Your Step 2A outputs (animal -> top number). Hardcoded from your printout. ---
STEP2_TOP = [
    ("owls", "743"),
    ("eagles", "836"),
    ("cats", "951"),
    ("dogs", "082"),
    ("lions", "409"),
    ("tigers", "836"),
    ("wolves", "836"),
    ("bears", "184"),
    ("sharks", "744"),
    ("dolphins", "074"),
    ("elephants", "836"),
    ("giraffes", "836"),
    ("penguins", "836"),
    ("koalas", "836"),
    ("pandas", "473"),
    ("foxes", "951"),
    ("horses", "75"),
    ("rabbits", "951"),
    ("snakes", "392"),
    ("spiders", "794"),
    ("cows", "082"),
    ("pigs", "473"),
    ("sheep", "743"),
    ("goats", "743"),
    ("monkeys", "435"),
    ("gorillas", "435"),
    ("zebras", "078"),
    ("rhinos", "714"),
    ("hippos", "743"),
]

# Very small singularization helper (matches what you already used elsewhere)
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


def avg_unembed_vector_for_text(tokenizer, unembed_weight: torch.Tensor, text: str) -> torch.Tensor:
    """
    Implements the "average across tokens" idea from the paper.
    Uses add_special_tokens=False so we only average actual text tokens.
    """
    ids = tokenizer(text, add_special_tokens=False).input_ids
    if len(ids) == 0:
        raise ValueError(f"Text produced no tokens: {text!r}")
    vecs = unembed_weight[torch.tensor(ids, device=unembed_weight.device)]
    return vecs.mean(dim=0)


def number_token_representation(tokenizer, unembed_weight: torch.Tensor, number_str: str) -> tuple[list[int], torch.Tensor]:
    """
    Try to represent the number as a single token if possible; else average across tokens.
    We try both `number_str` and `" " + number_str` (leading space) and pick the shorter tokenization.
    Returns (token_ids, avg_vector).
    """
    candidates = [number_str, " " + number_str]
    best_ids = None
    for cand in candidates:
        ids = tokenizer(cand, add_special_tokens=False).input_ids
        if best_ids is None or len(ids) < len(best_ids):
            best_ids = ids
    ids = best_ids
    vecs = unembed_weight[torch.tensor(ids, device=unembed_weight.device)]
    return ids, vecs.mean(dim=0)


def main():
    model_name = os.environ.get("MODEL_NAME", None)

    print("Loading model...")
    model, tokenizer = load_model(model_name)
    used = getattr(model.config, "_name_or_path", "unknown")
    print("USING MODEL:", used)

    # Unembedding matrix U (vocab_size x hidden_dim). For these HF causal LMs, lm_head.weight is standard.
    U = model.lm_head.weight
    U = U.to(torch.float32)  # avoid bf16 issues
    device = U.device

    # Precompute all number tokens once
    number_token_ids, number_strings = get_all_number_tokens(tokenizer)
    print(f"Found {len(number_token_ids)} number tokens in vocab.")

    # Gather unembedding vectors for number tokens
    num_ids_t = torch.tensor(number_token_ids, device=device, dtype=torch.long)
    U_nums = U.index_select(0, num_ids_t)  # [N_nums, d]
    U_nums_norm = F.normalize(U_nums, dim=1)

    rows = []
    for animal_plural, top_number in STEP2_TOP:
        animal = singularize(animal_plural)

        # concept token c (we use leading space to align with "next token" word forms)
        concept_text = " " + animal
        U_c = avg_unembed_vector_for_text(tokenizer, U, concept_text)
        U_c_norm = F.normalize(U_c, dim=0)

        # cosine similarity for all numbers: cos(U_t, U_c)
        cos_all = (U_nums_norm @ U_c_norm)  # [N_nums]

        # find representation of the *specific* top number token(s)
        top_num_token_ids, U_top = number_token_representation(tokenizer, U, top_number)
        U_top_norm = F.normalize(U_top, dim=0)

        # If top_number corresponds to a single number token in our list, we can rank by that token directly.
        # Otherwise, we rank by cosine between U_top (averaged) and U_c, and still report closest single-token rank.
        cos_top = float(torch.dot(U_top_norm, U_c_norm).item())

        # Attempt to map to a single token id that is actually one of the detected number tokens.
        # (If tokenization picked the " " + number version, it might be a single token; great.)
        mapped_single = None
        if len(top_num_token_ids) == 1 and top_num_token_ids[0] in set(number_token_ids):
            mapped_single = top_num_token_ids[0]

        # Rank: 1 = best (highest cosine)
        # We compute rank for the mapped single token if available; else we compute where cos_top would fall.
        cos_sorted, idx_sorted = torch.sort(cos_all, descending=True)

        if mapped_single is not None:
            # locate its index among number_token_ids
            j = number_token_ids.index(mapped_single)
            # its cosine is cos_all[j]
            target_cos = float(cos_all[j].item())
            # rank = 1 + number of items with cosine > target_cos (stable)
            rank = int((cos_all > cos_all[j]).sum().item()) + 1
            target_kind = "single_token_number"
        else:
            # rank by insertion point of cos_top among cos_all
            rank = int((cos_all > cos_top).sum().item()) + 1
            target_cos = cos_top
            target_kind = "avg_over_tokens"

        n = len(number_token_ids)
        percentile = 100.0 * (1.0 - (rank - 1) / max(1, n - 1))  # 100 = best, 0 = worst-ish

        # top-10 numbers by cosine similarity
        topk = 10
        best_idxs = idx_sorted[:topk].tolist()
        best_numbers = [number_strings[i] for i in best_idxs]
        best_scores = [float(cos_all[i].item()) for i in best_idxs]

        rows.append(
            {
                "animal_plural": animal_plural,
                "animal_singular": animal,
                "concept_text": concept_text,
                "step2_top_number": top_number,
                "step2_top_number_token_ids": str(top_num_token_ids),
                "target_kind": target_kind,
                "cosine(step2_number, animal)": target_cos,
                "rank_among_numbers_by_cosine": rank,
                "percentile_best_is_100": percentile,
                "top10_numbers_by_cosine": ", ".join(best_numbers),
                "top10_cosines": ", ".join([f"{s:.5f}" for s in best_scores]),
            }
        )

        print(
            f"[{animal_plural}] top_number={top_number} "
            f"cos={target_cos:.5f} rank={rank}/{n} (percentile={percentile:.1f})"
        )

    out = pd.DataFrame(rows)
    out.to_csv(OUT_CSV, index=False)
    print(f"\nSaved: {OUT_CSV}")

    # quick summary: best (lowest rank) animals
    print("\nBest-ranked Step2 numbers by cosine (top 10):")
    print(
        out.sort_values("rank_among_numbers_by_cosine")
          .head(10)[
              ["animal_plural", "step2_top_number", "cosine(step2_number, animal)",
               "rank_among_numbers_by_cosine", "percentile_best_is_100", "target_kind"]
          ].to_string(index=False)
    )


if __name__ == "__main__":
    main()