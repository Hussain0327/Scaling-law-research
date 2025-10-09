import os
from pathlib import Path

from datasets import load_dataset
from transformers import GPT2LMHeadModel, GPT2Tokenizer, Trainer, TrainingArguments


def main():
    model_name = "gpt2"
    output_dir = "./checkpoints/seal_gpt2"

    # 1️⃣ Prepare minimal dataset if none exists
    os.makedirs("data/sample", exist_ok=True)
    if not os.path.exists("data/sample/train.txt"):
        with open("data/sample/train.txt", "w") as f:
            f.write("Fine-tuning GPT-2 is simple.\nLet's test it!\n")

    if not os.path.exists("data/sample/val.txt"):
        with open("data/sample/val.txt", "w") as f:
            f.write("Validation helps prevent overfitting.\n")

    # 2️⃣ Load model + tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    model = GPT2LMHeadModel.from_pretrained(model_name)

    # 3️⃣ Load dataset
    dataset = load_dataset(
        "text",
        data_files={
            "train": "data/sample/train.txt",
            "validation": "data/sample/val.txt",
        },
    )

    # 4️⃣ Tokenize
    def tokenize_fn(examples):
        return tokenizer(
            examples["text"], truncation=True, padding="max_length", max_length=128
        )

    tokenized_dataset = dataset.map(
        tokenize_fn, batched=True, num_proc=2, remove_columns=["text"]
    )

    # 5️⃣ Training args
    training_args = TrainingArguments(
        output_dir=output_dir,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=5e-5,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        num_train_epochs=3,
        weight_decay=0.01,
        logging_dir="./logs",
        logging_steps=10,
        report_to="none",
    )

    # 6️⃣ Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["validation"],
    )

    # 7️⃣ Train
    trainer.train()

    # 8️⃣ Save model
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    trainer.save_model(output_dir)
    print(f"✅ Fine-tuned model saved to {output_dir}")


if __name__ == "__main__":
    main()
