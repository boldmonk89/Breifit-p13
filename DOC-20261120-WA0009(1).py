"""
BriefIt - core logic for Project 13 (Abstractive Text Summarisation)

If the folder `briefit_t5_small` exists (created by briefit_train.py) we use our own
fine-tuned T5-small. Otherwise we fall back to a pretrained DistilBART so the app still runs.
"""
import os
import re

LOCAL_MODEL = "briefit_t5_small"
FALLBACK_MODEL = "sshleifer/distilbart-cnn-12-6"

# (min_tokens, max_tokens) of the final summary
LENGTHS = {"Short": (25, 60), "Medium": (50, 110), "Long": (90, 170)}


def clean_article(text):
    text = re.sub(r"https?://\S+", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def split_into_chunks(text, max_words):
    """Group whole sentences together so no chunk goes over max_words."""
    sentences = re.split(r"(?<=[.!?])\s+", text)
    chunks, current, count = [], [], 0
    for s in sentences:
        n = len(s.split())
        if current and count + n > max_words:
            chunks.append(" ".join(current))
            current, count = [], 0
        current.append(s)
        count += n
    if current:
        chunks.append(" ".join(current))
    return chunks


class Summarizer:
    def __init__(self, model_path=None):
        import torch
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

        path = model_path or (LOCAL_MODEL if os.path.isdir(LOCAL_MODEL) else FALLBACK_MODEL)
        self.name = path
        self.is_t5 = "t5" in path.lower()
        self.prefix = "summarize: " if self.is_t5 else ""
        self.max_input = 512 if self.is_t5 else 1024
        self.chunk_words = 300 if self.is_t5 else 600

        self.torch = torch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(path)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(path).to(self.device)
        self.model.eval()

    def _generate(self, text, min_len, max_len):
        inputs = self.tokenizer(self.prefix + text, return_tensors="pt",
                                truncation=True, max_length=self.max_input).to(self.device)
        with self.torch.no_grad():
            ids = self.model.generate(**inputs, num_beams=4, min_length=min_len, max_length=max_len,
                                      no_repeat_ngram_size=3, length_penalty=2.0, early_stopping=True)
        return self.tokenizer.decode(ids[0], skip_special_tokens=True)

    def summarize(self, article, length="Medium"):
        article = clean_article(article)
        total_words = len(article.split())
        min_len, max_len = LENGTHS[length]

        chunks = split_into_chunks(article, self.chunk_words)
        # long document: summarise every chunk first, then summarise the joined summaries
        if len(chunks) > 1:
            partial = []
            for c in chunks:
                n = len(c.split())
                partial.append(self._generate(c, min(30, n // 4), 90))
            article = " ".join(partial)

        n = len(article.split())
        min_len = min(min_len, max(10, n // 3))
        summary = self._generate(article, min_len, max_len)

        out_words = len(summary.split())
        return {
            "summary": summary,
            "input_words": total_words,
            "summary_words": out_words,
            "chunks": len(chunks),
            "compression": 1 - out_words / max(total_words, 1),
        }
