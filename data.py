"""
Dataset loading and tokenization helpers for MicroGPT.

This module keeps things simple and explicit so it remains easy to tinker.
"""

import os
import random
from dataclasses import dataclass
from typing import List, Sequence


@dataclass
class Tokenizer:
    uchars: List[str]
    bos_token_id: int

    @property
    def vocab_size(self) -> int:
        return len(self.uchars) + 1

    def encode(self, text: str) -> List[int]:
        return [self.uchars.index(ch) for ch in text]

    def decode(self, token_ids: Sequence[int]) -> str:
        return "".join(self.uchars[t] for t in token_ids)


def maybe_download_default_dataset(path: str) -> None:
    if os.path.exists(path):
        return
    import urllib.request

    names_url = "https://raw.githubusercontent.com/karpathy/makemore/refs/heads/master/names.txt"
    urllib.request.urlretrieve(names_url, path)


def load_docs(path: str, seed: int = 42) -> List[str]:
    maybe_download_default_dataset(path)
    docs = [
        l.strip()
        for l in open(path, "r", encoding="utf-8").read().split("\n")
        if l.strip()
    ]
    random.seed(seed)
    random.shuffle(docs)
    return docs


def build_tokenizer(docs: Sequence[str]) -> Tokenizer:
    uchars = sorted(set("".join(docs)))
    bos_token_id = len(uchars)
    return Tokenizer(uchars=uchars, bos_token_id=bos_token_id)
