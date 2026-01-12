from __future__ import annotations
from typing import List, Tuple, Dict
import os
import math
import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset

from src.bpe import BPETokenizer


def _cache_dir() -> str:
    # Use env var if set ( Windows: C:\hf_cache\datasets)
    d = os.environ.get("HF_DATASETS_CACHE", None)
    if d:
        os.makedirs(d, exist_ok=True)
        return d
    d = os.path.join(os.getcwd(), ".hf_cache", "datasets")
    os.makedirs(d, exist_ok=True)
    return d


# Text utilities
def normalize(text: str, lowercase: bool = True) -> str:
    text = text.strip().replace("\n", " ")
    text = " ".join(text.split())
    return text.lower() if lowercase else text


def _shuffle(pairs: List[Tuple[str, str]], seed: int = 42) -> List[Tuple[str, str]]:
    g = torch.Generator().manual_seed(seed)
    idx = torch.randperm(len(pairs), generator=g).tolist()
    return [pairs[i] for i in idx]


def _oversample_to(pairs: List[Tuple[str, str]], target: int, seed: int = 123) -> List[Tuple[str, str]]:
    """
    Duplicate (with shuffling) to reach target size.
    This allows you to keep train_size big even if a dataset is smaller/partially available.
    """
    if len(pairs) == 0:
        return pairs
    if len(pairs) >= target:
        return pairs[:target]
    pairs = _shuffle(pairs, seed)
    reps = math.ceil(target / len(pairs))
    out = (pairs * reps)[:target]
    return _shuffle(out, seed + 1)


def _safe_get_translation(ex: Dict, lang1: str, lang2: str) -> Tuple[str, str]:
    tr = ex.get("translation", None)
    if isinstance(tr, dict):
        return tr.get(lang1, ""), tr.get(lang2, "")
    return ex.get(lang1, ""), ex.get(lang2, "")


def _load_translation_dataset(ds, lang1: str, lang2: str, lowercase: bool) -> List[Tuple[str, str]]:
    pairs: List[Tuple[str, str]] = []
    for ex in ds:
        s, t = _safe_get_translation(ex, lang1, lang2)
        if isinstance(s, str) and isinstance(t, str) and s.strip() and t.strip():
            pairs.append((normalize(s, lowercase), normalize(t, lowercase)))
    return pairs


# Dataset loaders (robust)
def _load_opus_books(cfg) -> List[Tuple[str, str]]:
    ds = load_dataset(
        cfg.opus_books_name,
        cfg.opus_books_config,
        cache_dir=_cache_dir(),
    )
    return _load_translation_dataset(ds["train"], "en", "de", cfg.lowercase)


def _try_load_open_subtitles(cfg) -> List[Tuple[str, str]]:
    """
    OpenSubtitles is script-based in many setups. We pass trust_remote_code=True and
    never crash if it fails.
    """
    try:
        ds = load_dataset(
            cfg.open_subtitles_name,
            lang1=cfg.open_subtitles_lang1,
            lang2=cfg.open_subtitles_lang2,
            trust_remote_code=True,
            cache_dir=_cache_dir(),
        )
    except Exception as e:
        print(f"[WARN] OpenSubtitles failed to load, skipping it. Reason:\n  {e}")
        return []

    # Most datasets expose only "train"
    split = "train" if "train" in ds else list(ds.keys())[0]
    return _load_translation_dataset(ds[split], cfg.open_subtitles_lang1, cfg.open_subtitles_lang2, cfg.lowercase)


def _try_load_tatoeba(cfg) -> List[Tuple[str, str]]:
    """
    Optional. Tatoeba sometimes breaks due to upstream OPUS URL changes.
    We'll never crash; we just skip it if it fails.
    """
    try:
        ds = load_dataset(
            cfg.tatoeba_name,
            lang1=cfg.tatoeba_lang1,
            lang2=cfg.tatoeba_lang2,
            trust_remote_code=True,
            cache_dir=_cache_dir(),
        )
    except Exception as e:
        print(f"[WARN] Tatoeba failed to load, skipping it. Reason:\n  {e}")
        return []

    split = "train" if "train" in ds else list(ds.keys())[0]
    return _load_translation_dataset(ds[split], cfg.tatoeba_lang1, cfg.tatoeba_lang2, cfg.lowercase)


# Mixed loader
def load_parallel_mixed(cfg) -> Tuple[List[Tuple[str, str]], List[Tuple[str, str]], List[Tuple[str, str]]]:
    """
    Mixes enabled datasets according to cfg.mix_* ratios.
    Guarantees non-empty val/test by oversampling if needed.
    """
    needed_total = cfg.train_size + cfg.val_size + cfg.test_size

    sources: List[Tuple[str, List[Tuple[str, str]]]] = []
    ratios: List[float] = []

    # --- Books
    if getattr(cfg, "use_opus_books", False) and getattr(cfg, "mix_opus_books", 0.0) > 0:
        books = _shuffle(_load_opus_books(cfg), seed=1)
        sources.append(("books", books))
        ratios.append(float(cfg.mix_opus_books))

    # --- OpenSubtitles
    if getattr(cfg, "use_open_subtitles", False) and getattr(cfg, "mix_open_subtitles", 0.0) > 0:
        subs = _shuffle(_try_load_open_subtitles(cfg), seed=2)
        if len(subs) > 0:
            sources.append(("open_subtitles", subs))
            ratios.append(float(cfg.mix_open_subtitles))
        else:
            print("[WARN] OpenSubtitles returned 0 pairs; continuing without it.")

    # --- Tatoeba (optional)
    if getattr(cfg, "use_tatoeba", False) and getattr(cfg, "mix_tatoeba", 0.0) > 0:
        tat = _shuffle(_try_load_tatoeba(cfg), seed=3)
        if len(tat) > 0:
            sources.append(("tatoeba", tat))
            ratios.append(float(cfg.mix_tatoeba))
        else:
            print("[WARN] Tatoeba returned 0 pairs; continuing without it.")

    if not sources:
        raise RuntimeError("No data sources loaded. Enable at least one dataset and set a non-zero mix ratio.")

    # Normalize ratios to sum to 1
    s = sum(ratios)
    ratios = [r / s for r in ratios]

    # Determine take counts (make sure total == needed_total)
    take_counts = [int(needed_total * r) for r in ratios]
    rem = needed_total - sum(take_counts)
    if rem > 0:
        take_counts[0] += rem

    mixed: List[Tuple[str, str]] = []
    for (name, pairs), n_take in zip(sources, take_counts):
        if len(pairs) == 0:
            continue
        chunk = _oversample_to(pairs, n_take, seed=100 + n_take)
        mixed.extend(chunk)

    mixed = _shuffle(mixed, seed=999)

    # Final safety: if somehow mixed is still short, oversample the whole thing
    if len(mixed) < needed_total and len(mixed) > 0:
        mixed = _oversample_to(mixed, needed_total, seed=777)

    train = mixed[: cfg.train_size]
    val = mixed[cfg.train_size : cfg.train_size + cfg.val_size]
    test = mixed[cfg.train_size + cfg.val_size : cfg.train_size + cfg.val_size + cfg.test_size]

    print(f"Loaded pairs: total={len(mixed)} train={len(train)} val={len(val)} test={len(test)}")
    return train, val, test


# Tokenizer builder (BPE) + caching
def build_tokenizers(train_pairs: List[Tuple[str, str]], cfg):
    os.makedirs("bpe_cache", exist_ok=True)

    special = [cfg.pad_token, cfg.bos_token, cfg.eos_token, cfg.unk_token]
    key = f"bpe_vs{cfg.bpe_vocab_size}_m{cfg.bpe_num_merges}_len{cfg.max_len}"
    src_path = os.path.join("bpe_cache", f"src_{key}.pt")
    tgt_path = os.path.join("bpe_cache", f"tgt_{key}.pt")

    if os.path.exists(src_path) and os.path.exists(tgt_path):
        print("[BPE] Loading cached tokenizers")
        src_tok = torch.load(src_path)
        tgt_tok = torch.load(tgt_path)
        return src_tok, tgt_tok

    print("[BPE] Training tokenizers from scratch")
    src_tok = BPETokenizer(special)
    tgt_tok = BPETokenizer(special)

    src_texts = [s for s, _ in train_pairs]
    tgt_texts = [t for _, t in train_pairs]

    src_tok.train(src_texts, vocab_size=cfg.bpe_vocab_size, num_merges=cfg.bpe_num_merges)
    tgt_tok.train(tgt_texts, vocab_size=cfg.bpe_vocab_size, num_merges=cfg.bpe_num_merges)

    torch.save(src_tok, src_path)
    torch.save(tgt_tok, tgt_path)
    print("[BPE] Tokenizers saved")
    return src_tok, tgt_tok


# Dataset & DataLoaders
class TranslationDataset(Dataset):
    def __init__(self, pairs, src_tok, tgt_tok, cfg):
        self.pairs = pairs
        self.src_tok = src_tok
        self.tgt_tok = tgt_tok
        self.cfg = cfg

        self.src_bos = src_tok.stoi[cfg.bos_token]
        self.src_eos = src_tok.stoi[cfg.eos_token]
        self.tgt_bos = tgt_tok.stoi[cfg.bos_token]
        self.tgt_eos = tgt_tok.stoi[cfg.eos_token]

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        src, tgt = self.pairs[idx]

        src_ids = self.src_tok.encode(src, self.cfg.unk_token)[: self.cfg.max_len - 2]
        tgt_ids = self.tgt_tok.encode(tgt, self.cfg.unk_token)[: self.cfg.max_len - 2]

        src_ids = [self.src_bos] + src_ids + [self.src_eos]
        tgt_ids = [self.tgt_bos] + tgt_ids + [self.tgt_eos]

        return torch.tensor(src_ids, dtype=torch.long), torch.tensor(tgt_ids, dtype=torch.long)


def collate_fn(batch, pad_src: int, pad_tgt: int):
    srcs, tgts = zip(*batch)
    src_len = max(x.size(0) for x in srcs)
    tgt_len = max(x.size(0) for x in tgts)

    src_pad = torch.full((len(batch), src_len), pad_src, dtype=torch.long)
    tgt_pad = torch.full((len(batch), tgt_len), pad_tgt, dtype=torch.long)

    for i, (s, t) in enumerate(zip(srcs, tgts)):
        src_pad[i, : s.size(0)] = s
        tgt_pad[i, : t.size(0)] = t

    return src_pad, tgt_pad


def make_loaders(train, val, test, src_tok, tgt_tok, cfg):
    pad_src = src_tok.stoi[cfg.pad_token]
    pad_tgt = tgt_tok.stoi[cfg.pad_token]

    train_ds = TranslationDataset(train, src_tok, tgt_tok, cfg)
    val_ds = TranslationDataset(val, src_tok, tgt_tok, cfg)
    test_ds = TranslationDataset(test, src_tok, tgt_tok, cfg)

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=lambda b: collate_fn(b, pad_src, pad_tgt),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=lambda b: collate_fn(b, pad_src, pad_tgt),
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=lambda b: collate_fn(b, pad_src, pad_tgt),
    )

    return train_loader, val_loader, test_loader
