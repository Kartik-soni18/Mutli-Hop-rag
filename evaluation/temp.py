import statistics
import tiktoken
import json
from pathlib import Path
enc = tiktoken.get_encoding("cl100k_base")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
Corpus_PATH = PROJECT_ROOT / "data" / "corpus.json"
docs = json.loads(Corpus_PATH.read_text(encoding="utf-8"))

char_lengths = [len(x["body"]) for x in docs]
token_lengths = [len(enc.encode(x["body"])) for x in docs]

print("Characters")
print("median:", statistics.median(char_lengths))
print("mean:", statistics.mean(char_lengths))

print("Tokens")
print("median:", statistics.median(token_lengths))
print("mean:", statistics.mean(token_lengths))