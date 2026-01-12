from __future__ import annotations
from collections import Counter
from typing import List, Tuple, Dict, Optional


def _word_to_symbols(word: str) -> List[str]:
    return list(word) + ["</w>"]


def _get_stats(vocab: Dict[Tuple[str, ...], int]) -> Counter:
    pairs = Counter()
    for word, freq in vocab.items():
        for i in range(len(word) - 1):
            pairs[(word[i], word[i + 1])] += freq
    return pairs


def _merge_vocab(pair: Tuple[str, str], vocab: Dict[Tuple[str, ...], int]) -> Dict[Tuple[str, ...], int]:
    a, b = pair
    merged_vocab = {}
    for word, freq in vocab.items():
        new_word = []
        i = 0
        while i < len(word):
            if i < len(word) - 1 and word[i] == a and word[i + 1] == b:
                new_word.append(a + b)
                i += 2
            else:
                new_word.append(word[i])
                i += 1
        merged_vocab[tuple(new_word)] = freq
    return merged_vocab


class BPETokenizer:
    """
    IMPORTANT:
    This tokenizer is designed to be compatible with the checkpoint you already trained.
    It can:
      - train (if you ever need it)
      - load merges/itos/stoi from checkpoint
      - encode using saved merges (robust to </w> variants)
      - decode clean text (removes </w> markers)

    Key idea:
    - We DON'T change merges or vocab now.
    - We only ensure encode/decode behave consistently with the saved token inventory.
    """

    def __init__(self, special_tokens: List[str]):
        self.special_tokens = special_tokens
        self.merges: List[Tuple[str, str]] = []
        self.itos: List[str] = list(special_tokens)
        self.stoi: Dict[str, int] = {t: i for i, t in enumerate(self.itos)}

        # Cache for fast encoding
        self._has_eow_in_vocab: Optional[bool] = None

    def __len__(self):
        return len(self.itos)

    # Training
    def train(self, texts: List[str], vocab_size: int, num_merges: int):
        word_freq = Counter()
        for line in texts:
            for w in line.split():
                if w:
                    word_freq[w] += 1

        vocab = {tuple(_word_to_symbols(w)): f for w, f in word_freq.items()}

        for _ in range(num_merges):
            stats = _get_stats(vocab)
            if not stats:
                break
            best = stats.most_common(1)[0][0]
            vocab = _merge_vocab(best, vocab)
            self.merges.append(best)

        # collect tokens
        tokens = Counter()
        for syms, freq in vocab.items():
            for s in syms:
                tokens[s] += freq

        # rebuild vocab: specials first, then most common
        self.itos = list(self.special_tokens)
        self.stoi = {t: i for i, t in enumerate(self.itos)}
        for tok, _ in tokens.most_common():
            if tok not in self.stoi:
                self.stoi[tok] = len(self.itos)
                self.itos.append(tok)
            if len(self.itos) >= vocab_size:
                break

        self._has_eow_in_vocab = None

    # Loading from checkpoint
    def load_state(self, itos: List[str], merges: List[Tuple[str, str]]):
        self.itos = list(itos)
        self.stoi = {t: i for i, t in enumerate(self.itos)}
        # merges may come as list of lists in torch load; normalize to tuples
        self.merges = [tuple(m) for m in merges]
        self._has_eow_in_vocab = None

    def _detect_eow_variant(self) -> bool:
        """
        Detect if vocab tokens include '</w>' inside tokens.
        If yes -> we can decode by stripping marker.
        """
        if self._has_eow_in_vocab is None:
            self._has_eow_in_vocab = any("</w>" in t for t in self.itos)
        return self._has_eow_in_vocab

    # Encoding
    def encode_word(self, word: str) -> List[str]:
        # Start from chars + </w>
        symbols = _word_to_symbols(word)

        # Apply merges in order
        for a, b in self.merges:
            i = 0
            new = []
            while i < len(symbols):
                if i < len(symbols) - 1 and symbols[i] == a and symbols[i + 1] == b:
                    new.append(a + b)
                    i += 2
                else:
                    new.append(symbols[i])
                    i += 1
            symbols = new

        # IMPORTANT: we do NOT “invent” a new boundary strategy here.
        # We keep whatever results from merges. That matches training behavior best.
        return symbols

    def encode(self, text: str, unk_token: str) -> List[int]:
        unk_id = self.stoi.get(unk_token, 0)
        ids: List[int] = []

        for w in text.split():
            if not w:
                continue
            syms = self.encode_word(w)

            # Robust mapping:
            # If symbol isn't in vocab, try a couple fallbacks before unk.
            for s in syms:
                if s in self.stoi:
                    ids.append(self.stoi[s])
                    continue

                # fallback 1: strip </w> if present
                if s.endswith("</w>"):
                    base = s[:-4]
                    if base in self.stoi:
                        ids.append(self.stoi[base])
                        continue

                # fallback 2: if </w> is separate token, try split
                if "</w>" in s and s != "</w>":
                    # example: 'ist</w>' could be token, but if not, split
                    base = s.replace("</w>", "")
                    if base in self.stoi:
                        ids.append(self.stoi[base])
                        # also append </w> if exists as token
                        if "</w>" in self.stoi:
                            ids.append(self.stoi["</w>"])
                        continue

                ids.append(unk_id)

        return ids

    # Decoding
    def decode(self, ids: List[int]) -> str:
        """
        Convert BPE tokens back to readable text:
        - remove special tokens
        - use </w> markers to decide spaces
        """
        self._detect_eow_variant()

        toks = []
        for i in ids:
            ii = int(i)
            if 0 <= ii < len(self.itos):
                toks.append(self.itos[ii])

        # remove specials
        specials = set(self.special_tokens)
        toks = [t for t in toks if t not in specials]

        out_words = []
        cur = ""

        for t in toks:
            if t == "</w>":
                # end word marker as standalone
                if cur:
                    out_words.append(cur)
                    cur = ""
                continue

            if t.endswith("</w>"):
                cur += t[:-4]
                if cur:
                    out_words.append(cur)
                cur = ""
            else:
                # regular subword piece: glue
                cur += t

        if cur:
            out_words.append(cur)

        text = " ".join(out_words)
        text = " ".join(text.split()).strip()
        return text
