"""Prove the torchtext-free shim is byte-identical to torchtext 0.18.0.

Run on a machine where torchtext STILL works (e.g. the Windows 4080 env):
    python verify_text_utils.py
Compares, against real torchtext:
  (1) basic_english tokenization over all AG News train+test texts + edge cases
  (2) the full vocab built from train (itos order == indices) for several caps
  (3) OOV / default-index lookup behavior
Exits non-zero on ANY mismatch.
"""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

import torchtext
torchtext.disable_torchtext_deprecation_warning()
from torchtext.data.utils import get_tokenizer as tt_get_tokenizer
from torchtext.vocab import build_vocab_from_iterator as tt_build_vocab
from datasets import load_dataset

import text_utils as mine

tt_tok = tt_get_tokenizer("basic_english")
my_tok = mine.get_tokenizer("basic_english")

# ---- (1) tokenizer over the real corpus + tricky edge cases ----------------
ds = load_dataset("ag_news")
texts = list(ds["train"]["text"]) + list(ds["test"]["text"])
edge = [
    "", "   ", "AT&T's stock fell 3.5% (again)!",
    "He said: \"Hello?\"; she didn't.",
    "<br /><br />Line break test", "U.S.A. vs U.K. — 100% sure?",
    "Multiple     spaces\tand\ttabs", "Numbers 1,000,000 and $4.50",
    "Café naïve résumé", "Mixed.CASE.Words,here",
]
n_tok = 0
for i, t in enumerate(texts):
    if tt_tok(t) != my_tok(t):
        print(f"TOKEN MISMATCH at corpus idx {i}:\n  text={t!r}\n  tt ={tt_tok(t)}\n  me ={my_tok(t)}")
        sys.exit(1)
    n_tok += 1
for t in edge:
    if tt_tok(t) != my_tok(t):
        print(f"TOKEN MISMATCH on edge case:\n  text={t!r}\n  tt ={tt_tok(t)}\n  me ={my_tok(t)}")
        sys.exit(1)
print(f"[1] tokenizer identical on {n_tok:,} corpus texts + {len(edge)} edge cases: PASS")

# ---- (2) full vocab identical (order == indices) for several caps ----------
# Build from the SAME 90/10 train split the pipeline uses (data_seed=42), train only.
split = ds["train"].train_test_split(test_size=0.1, seed=42)
train_texts = list(split["train"]["text"])

def yield_tokens(texts, tok):
    for t in texts:
        yield tok(t)

for cap in (20_000, 10_000, None):
    tt_v = tt_build_vocab(yield_tokens(train_texts, tt_tok), min_freq=1,
                          specials=["<pad>", "<unk>"], special_first=True, max_tokens=cap)
    my_v = mine.build_vocab_from_iterator(yield_tokens(train_texts, my_tok), min_freq=1,
                                          specials=["<pad>", "<unk>"], special_first=True, max_tokens=cap)
    if tt_v.get_itos() != my_v.get_itos():
        # find first differing index
        a, b = tt_v.get_itos(), my_v.get_itos()
        for j in range(min(len(a), len(b))):
            if a[j] != b[j]:
                print(f"VOCAB MISMATCH (cap={cap}) at index {j}: tt={a[j]!r} me={b[j]!r}")
                break
        print(f"  sizes: tt={len(a)} me={len(b)}")
        sys.exit(1)
    print(f"[2] vocab identical (cap={cap}): size={len(my_v):,}  PASS")

# ---- (3) lookup + OOV/default behavior on the cap=20000 vocab --------------
tt_v = tt_build_vocab(yield_tokens(train_texts, tt_tok), min_freq=1,
                      specials=["<pad>", "<unk>"], special_first=True, max_tokens=20_000)
my_v = mine.build_vocab_from_iterator(yield_tokens(train_texts, my_tok), min_freq=1,
                                      specials=["<pad>", "<unk>"], special_first=True, max_tokens=20_000)
assert tt_v["<pad>"] == my_v["<pad>"] == 0
assert tt_v["<unk>"] == my_v["<unk>"] == 1
tt_v.set_default_index(tt_v["<unk>"])
my_v.set_default_index(my_v["<unk>"])
probe = my_tok("the company said zzzqqq_not_a_real_token NASA stocks oil")
assert tt_v(probe) == my_v(probe), f"lookup mismatch: {tt_v(probe)} vs {my_v(probe)}"
assert tt_v["zzzqqq_not_a_real_token"] == my_v["zzzqqq_not_a_real_token"] == 1  # OOV -> unk
print(f"[3] lookups + OOV->unk identical (probe ids={my_v(probe)}): PASS")

print("\nALL CHECKS PASSED — shim is byte-identical to torchtext 0.18.0.")
