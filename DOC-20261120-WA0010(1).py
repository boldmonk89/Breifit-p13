"""
Fine-tune T5-small on a subset of CNN/DailyMail.

    python briefit_train.py --train_size 5000 --epochs 3

A GPU (Google Colab T4 is enough) is strongly recommended. On CPU keep --train_size small.
The model is saved in ./briefit_t5_small and results in briefit_results.json
"""
import argparse
import json

import evaluate
import numpy as np
from datasets import load_dataset
from transformers import (AutoModelForSeq2SeqLM, AutoTokenizer, DataCollatorForSeq2Seq,
                          Seq2SeqTrainer, Seq2SeqTrainingArguments)

ap = argparse.ArgumentParser()
ap.add_argument("--train_size", type=int, default=5000)
ap.add_argument("--val_size", type=int, default=300)
ap.add_argument("--test_size", type=int, default=200)
ap.add_argument("--epochs", type=int, default=3)
ap.add_argument("--batch", type=int, default=8)
ap.add_argument("--out", default="briefit_t5_small")
args = ap.parse_args()

MODEL = "t5-small"
MAX_IN, MAX_OUT = 512, 128

tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL)

data = load_dataset("cnn_dailymail", "3.0.0")
train = data["train"].shuffle(seed=42).select(range(args.train_size))
val = data["validation"].select(range(args.val_size))
test = data["test"].select(range(args.test_size))


def preprocess(batch):
    inputs = ["summarize: " + a for a in batch["article"]]
    enc = tok(inputs, max_length=MAX_IN, truncation=True)
    labels = tok(text_target=batch["highlights"], max_length=MAX_OUT, truncation=True)
    enc["labels"] = labels["input_ids"]
    return enc


cols = train.column_names
train_tok = train.map(preprocess, batched=True, remove_columns=cols)
val_tok = val.map(preprocess, batched=True, remove_columns=cols)

rouge = evaluate.load("rouge")


def compute_metrics(eval_pred):
    preds, labels = eval_pred
    preds = np.where(preds != -100, preds, tok.pad_token_id)
    labels = np.where(labels != -100, labels, tok.pad_token_id)
    pred_text = tok.batch_decode(preds, skip_special_tokens=True)
    label_text = tok.batch_decode(labels, skip_special_tokens=True)
    scores = rouge.compute(predictions=pred_text, references=label_text, use_stemmer=True)
    return {k: round(v * 100, 2) for k, v in scores.items()}


train_args = Seq2SeqTrainingArguments(
    output_dir="briefit_checkpoints",
    num_train_epochs=args.epochs,
    per_device_train_batch_size=args.batch,
    per_device_eval_batch_size=args.batch,
    learning_rate=3e-4,
    weight_decay=0.01,
    eval_strategy="epoch",
    save_strategy="no",
    predict_with_generate=True,
    generation_max_length=MAX_OUT,
    fp16=False,
    logging_steps=100,
    report_to="none",
)

trainer = Seq2SeqTrainer(
    model=model,
    args=train_args,
    train_dataset=train_tok,
    eval_dataset=val_tok,
    data_collator=DataCollatorForSeq2Seq(tok, model=model),
    compute_metrics=compute_metrics,
)
trainer.train()

model.save_pretrained(args.out)
tok.save_pretrained(args.out)

# ---- final scores on the test subset ----
model.eval()
preds = []
for i in range(0, len(test), 8):
    batch = ["summarize: " + a for a in test["article"][i:i + 8]]
    enc = tok(batch, max_length=MAX_IN, truncation=True, padding=True, return_tensors="pt").to(model.device)
    out = model.generate(**enc, num_beams=4, max_length=MAX_OUT, no_repeat_ngram_size=3, early_stopping=True)
    preds += tok.batch_decode(out, skip_special_tokens=True)

scores = rouge.compute(predictions=preds, references=test["highlights"], use_stemmer=True)
results = {k: round(v * 100, 2) for k, v in scores.items()}
results.update({"train_size": args.train_size, "epochs": args.epochs, "test_size": args.test_size})
print(results)
json.dump(results, open("briefit_results.json", "w"), indent=2)
