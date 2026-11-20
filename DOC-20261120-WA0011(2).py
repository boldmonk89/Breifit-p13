"""
Compare the Lead-3 baseline (first three sentences) with our model using ROUGE.

    python briefit_evaluate.py --n 100
"""
import argparse
import re

import evaluate
from datasets import load_dataset

from briefit_engine import Summarizer

ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=100, help="number of test articles")
args = ap.parse_args()

test = load_dataset("cnn_dailymail", "3.0.0", split="test").select(range(args.n))
rouge = evaluate.load("rouge")


def lead3(text):
    sentences = re.split(r"(?<=[.!?])\s+", text)
    return " ".join(sentences[:3])


def show(name, preds):
    s = rouge.compute(predictions=preds, references=test["highlights"], use_stemmer=True)
    print(f"{name:<12} ROUGE-1 {s['rouge1']*100:5.2f} | ROUGE-2 {s['rouge2']*100:5.2f} | ROUGE-L {s['rougeL']*100:5.2f}")


show("Lead-3", [lead3(a) for a in test["article"]])

model = Summarizer()
print("Model:", model.name)
show("BriefIt", [model.summarize(a, "Medium")["summary"] for a in test["article"]])
