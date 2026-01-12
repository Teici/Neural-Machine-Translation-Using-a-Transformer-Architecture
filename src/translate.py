import torch
from src.config import Config
from src.model import TransformerNMT
from src.utils import pad_keep_mask
from src.bpe import BPETokenizer


def rebuild_tokenizer(tok_state, cfg: Config) -> BPETokenizer:
    """
    Rebuild tokenizer EXACTLY as trained using checkpoint state.
    This must match training to get meaningful translations.
    """
    special = [cfg.pad_token, cfg.bos_token, cfg.eos_token, cfg.unk_token]
    tok = BPETokenizer(special_tokens=special)

    itos = tok_state["itos"]
    merges = tok_state["merges"]
    tok.load_state(itos=itos, merges=merges)

    # Ensure specials exist in vocab
    for sp in special:
        if sp not in tok.stoi:
            # If missing, insert at front (rare, but makes it safe)
            tok.itos.insert(0, sp)
            tok.stoi = {t: i for i, t in enumerate(tok.itos)}

    return tok


def no_repeat_ngram_ok(seq_ids, next_id, n=3) -> bool:
    """
    Disallow repeating n-grams in the generated sequence.
    """
    if len(seq_ids) < n - 1:
        return True
    prefix = tuple(seq_ids[-(n - 1):])
    # Build set of existing n-grams
    existing = set()
    for i in range(len(seq_ids) - n + 1):
        existing.add(tuple(seq_ids[i:i + n]))
    candidate = prefix + (next_id,)
    return candidate not in existing


@torch.no_grad()
def beam_search_single(model, src, src_keep_mask, tgt_tok, cfg: Config):
    device = src.device

    bos = tgt_tok.stoi[cfg.bos_token]
    eos = tgt_tok.stoi[cfg.eos_token]
    pad = tgt_tok.stoi[cfg.pad_token]

    # Beam: list of (sequence_ids, logprob)
    beams = [([bos], 0.0)]
    finished = []

    for _ in range(cfg.max_decode_len - 1):
        new_beams = []

        for seq, score in beams:
            if seq[-1] == eos:
                finished.append((seq, score))
                continue

            tgt_inp = torch.tensor(seq, dtype=torch.long, device=device).unsqueeze(0)
            tgt_keep = pad_keep_mask(tgt_inp, pad)

            logits = model(src, tgt_inp, src_keep_mask, tgt_keep)  # (1, T, V)
            logp = torch.log_softmax(logits[0, -1, :], dim=-1)     # (V,)

            topk = torch.topk(logp, k=cfg.beam_size)
            for tok_id, tok_lp in zip(topk.indices.tolist(), topk.values.tolist()):
                # block repetition (trigrams)
                if not no_repeat_ngram_ok(seq, tok_id, n=3):
                    continue
                new_beams.append((seq + [tok_id], score + float(tok_lp)))

        if not new_beams:
            break

        # length penalty ranking
        def lp(s, length):
            return s / ((5 + length) ** cfg.length_penalty / (6 ** cfg.length_penalty))

        new_beams.sort(key=lambda x: lp(x[1], len(x[0])), reverse=True)
        beams = new_beams[:cfg.beam_size]

        # If all beams ended, stop
        if all(seq[-1] == eos for seq, _ in beams):
            finished.extend(beams)
            break

    finished.extend(beams)

    # Choose best final
    def final_score(item):
        seq, sc = item
        return sc / ((5 + len(seq)) ** cfg.length_penalty / (6 ** cfg.length_penalty))

    finished.sort(key=final_score, reverse=True)
    best = finished[0][0]

    # remove specials for printing
    best_ids = [i for i in best if i not in (bos, eos, pad)]
    return tgt_tok.decode(best_ids)


def main():
    cfg = Config()
    device = torch.device(cfg.device if torch.cuda.is_available() and cfg.device == "cuda" else "cpu")

    try:
        ckpt = torch.load("mt_transformer_scratch.pt", map_location=device)
    except FileNotFoundError:
        print("Checkpoint not found. Run: python -m src.train")
        return

    # load cfg from checkpoint if present (important!)
    for k, v in ckpt.get("cfg", {}).items():
        if hasattr(cfg, k):
            setattr(cfg, k, v)

    src_tok = rebuild_tokenizer(ckpt["src_tok"], cfg)
    tgt_tok = rebuild_tokenizer(ckpt["tgt_tok"], cfg)

    model = TransformerNMT(len(src_tok), len(tgt_tok), cfg).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    print("Loaded checkpoint: mt_transformer_scratch.pt")
    print("Type an English sentence (blank line to quit).")

    while True:
        try:
            s = input("EN> ").strip()
        except KeyboardInterrupt:
            print("\nBye.")
            break

        if not s:
            break

        if getattr(cfg, "lowercase", True):
            s = s.lower()

        # Encode source exactly as training tokenizer expects
        src_ids = src_tok.encode(s, cfg.unk_token)[: cfg.max_len - 2]
        bos = src_tok.stoi[cfg.bos_token]
        eos = src_tok.stoi[cfg.eos_token]
        pad = src_tok.stoi[cfg.pad_token]

        src_ids = [bos] + src_ids + [eos]
        src = torch.tensor(src_ids, dtype=torch.long, device=device).unsqueeze(0)
        src_keep = pad_keep_mask(src, pad)

        out = beam_search_single(model, src, src_keep, tgt_tok, cfg)
        print("DE>", out)


if __name__ == "__main__":
    main()
