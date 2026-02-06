import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

def main():
    model_id = os.environ.get("MODEL_ID", "distilgpt2")  # small + fast

    print("=== GPU / Torch info ===")
    print("torch version:", torch.__version__)
    print("cuda available:", torch.cuda.is_available())
    if not torch.cuda.is_available():
        raise SystemExit("CUDA not available. Install GPU PyTorch / check drivers.")

    device = torch.device("cuda")
    print("gpu:", torch.cuda.get_device_name(0))

    print("\n=== Loading model/tokenizer ===")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id)
    model.to(device)
    model.eval()

    prompt = "The capital of France is"
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    # Forward pass (logits)
    print("\n=== Forward pass ===")
    with torch.no_grad():
        out = model(**inputs)
    logits = out.logits
    print("logits shape:", tuple(logits.shape))  # (batch, seq, vocab)

    # Print top-5 next-token candidates
    next_logits = logits[0, -1, :]
    probs = torch.softmax(next_logits, dim=-1)
    topk = torch.topk(probs, k=5)
    top_ids = topk.indices.tolist()
    top_ps = topk.values.tolist()
    top_toks = [tokenizer.decode([i]) for i in top_ids]
    print("top-5 next tokens:")
    for tok, p in zip(top_toks, top_ps):
        print(f"  {repr(tok):>12}  p={p:.4f}")

    # Generation
    print("\n=== Generation ===")
    with torch.no_grad():
        gen_ids = model.generate(
            **inputs,
            max_new_tokens=30,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id,
        )
    gen_text = tokenizer.decode(gen_ids[0], skip_special_tokens=True)
    print(gen_text)

    # VRAM usage
    torch.cuda.synchronize()
    allocated = torch.cuda.memory_allocated(0) / (1024**2)
    reserved = torch.cuda.memory_reserved(0) / (1024**2)
    print("\n=== VRAM usage (approx) ===")
    print(f"allocated: {allocated:.1f} MiB")
    print(f"reserved:  {reserved:.1f} MiB")

if __name__ == "__main__":
    main()