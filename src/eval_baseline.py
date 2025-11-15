import re
import pandas as pd
import torch
from transformer_lens import HookedTransformer

device = "cuda" if torch.cuda.is_available() else "cpu"

def load_model():
    print("Loading model...")
    model = HookedTransformer.from_pretrained("gpt2-small", device=device)
    return model

def extract_digits(text):
    matches = re.findall(r"\d+", text)
    return matches[-1] if matches else None

def predict(model, prompt):
    tokens = model.to_tokens(prompt, prepend_bos=True)
    out = model.generate(
        tokens,
        max_new_tokens=4,
        temperature=0.0,        # deterministic
        top_k=1,                # greedy
    )
    new_tokens = out[0, tokens.shape[1]:]
    text = model.to_string(new_tokens)
    return text

def evaluate(model, path):
    df = pd.read_csv(path)
    correct = 0
    total = len(df)

    shown = 0

    for _, row in df.iterrows():
        prompt = row["prompt"]
        gold = str(row["answer"])

        pred_text = predict(model, prompt)
        pred = extract_digits(pred_text)

        if shown < 5:
            print("\n==============================")
            print("PROMPT:", repr(prompt))
            print("RAW MODEL OUTPUT:", repr(pred_text))
            print("EXTRACTED DIGITS:", repr(pred))
            print("GOLD ANSWER:", repr(gold))
            print("==============================\n")
            shown += 1

        if pred == gold:
            correct += 1

    accuracy = correct / total
    return accuracy


if __name__ == "__main__":
    model = load_model()

    print("Evaluating on test set...")
    acc = evaluate(model, "data/raw/addition_test.csv")

    print(f"Baseline accuracy: {acc * 100:.2f}%")

