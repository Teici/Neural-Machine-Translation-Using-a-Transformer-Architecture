import torch
from tqdm import tqdm
from sacrebleu import corpus_bleu

from src.config import Config
from src.data import load_parallel_mixed, build_tokenizers, make_loaders
from src.model import TransformerNMT
from src.utils import set_seed, pad_keep_mask, SimpleLRScheduler


def label_smoothed_loss(logits, target, pad_id: int, eps: float):
    """
    logits: (B, T, V)
    target: (B, T)
    """
    log_probs = torch.log_softmax(logits, dim=-1)
    nll = -log_probs.gather(dim=-1, index=target.unsqueeze(-1)).squeeze(-1)  
    smooth = -log_probs.mean(dim=-1)  

    mask = (target != pad_id)
    nll = (nll * mask).sum()
    smooth = (smooth * mask).sum()
    denom = mask.sum().clamp(min=1)

    return (1 - eps) * (nll / denom) + eps * (smooth / denom)


@torch.no_grad()
def beam_search(
    model,
    src,
    src_keep_mask,
    bos_id,
    eos_id,
    pad_id,
    max_len,
    beam_size=4,
    length_penalty=0.6,
):
    """
    Beam search for a single sentence batch (B=1).
    Returns token ids tensor shape (T,)
    """
    device = src.device
    model.eval()

    B = src.size(0)
    assert B == 1, "Beam search expects batch size 1 for simplicity."

    beams = [(torch.tensor([bos_id], device=device, dtype=torch.long), 0.0)]  
    finished = []

    for _ in range(max_len - 1):
        new_beams = []
        for seq, score in beams:
            if seq[-1].item() == eos_id:
                finished.append((seq, score))
                continue

            tgt_inp = seq.unsqueeze(0)  # (1, T)
            tgt_keep = pad_keep_mask(tgt_inp, pad_id)
            logits = model(src, tgt_inp, src_keep_mask, tgt_keep)  
            logp = torch.log_softmax(logits[0, -1, :], dim=-1)      

            topk = torch.topk(logp, k=beam_size)
            for tok_id, tok_lp in zip(topk.indices.tolist(), topk.values.tolist()):
                new_seq = torch.cat([seq, torch.tensor([tok_id], device=device)])
                new_score = score + float(tok_lp)
                new_beams.append((new_seq, new_score))

        def lp(s, length):
            # length penalty
            return s / ((5 + length) ** length_penalty / (6 ** length_penalty))

        new_beams.sort(key=lambda x: lp(x[1], len(x[0])), reverse=True)
        beams = new_beams[:beam_size]

        if not beams:
            break

    finished.extend(beams)
    finished.sort(
        key=lambda x: x[1] / ((5 + len(x[0])) ** length_penalty / (6 ** length_penalty)),
        reverse=True,
    )
    best = finished[0][0]
    return best


@torch.no_grad()
def evaluate_bleu(model, loader, src_tok, tgt_tok, cfg, device, max_sentences=200):
    """
    Computes corpus BLEU using beam search decoding.
    To keep evaluation time reasonable on CPU, we cap to max_sentences.
    """
    model.eval()
    preds, refs = [], []
    count = 0

    bos = tgt_tok.stoi[cfg.bos_token]
    eos = tgt_tok.stoi[cfg.eos_token]
    pad = tgt_tok.stoi[cfg.pad_token]

    for src, tgt in tqdm(loader, desc="Evaluating"):
        for i in range(src.size(0)):
            if count >= max_sentences:
                break

            s = src[i : i + 1].to(device)
            t = tgt[i : i + 1].to(device)

            src_keep = pad_keep_mask(s, src_tok.stoi[cfg.pad_token])

            out_ids = beam_search(
                model,
                s,
                src_keep,
                bos_id=bos,
                eos_id=eos,
                pad_id=pad,
                max_len=cfg.max_decode_len,
                beam_size=cfg.beam_size,
                length_penalty=cfg.length_penalty,
            ).tolist()

            # remove special tokens for text
            out_ids = [x for x in out_ids if x not in (bos, eos, pad)]
            ref_ids = [x for x in t[0].tolist() if x not in (bos, eos, pad)]

            pred_text = tgt_tok.decode(out_ids)
            ref_text = tgt_tok.decode(ref_ids)

            if not pred_text.strip():
                pred_text = "<empty>"
            if not ref_text.strip():
                ref_text = "<empty>"

            preds.append(pred_text)
            refs.append(ref_text)
            count += 1

        if count >= max_sentences:
            break

    return corpus_bleu(preds, [refs]).score


def main():
    cfg = Config()
    set_seed(42)

    device = torch.device(cfg.device if torch.cuda.is_available() and cfg.device == "cuda" else "cpu")
    print("Device:", device)

    # Mixed conversational + books
    train_pairs, val_pairs, test_pairs = load_parallel_mixed(cfg)

    # Your homemade tokenizer
    src_tok, tgt_tok = build_tokenizers(train_pairs, cfg)

    # Data loaders
    train_loader, val_loader, test_loader = make_loaders(
        train_pairs, val_pairs, test_pairs, src_tok, tgt_tok, cfg
    )

    # Model
    model = TransformerNMT(len(src_tok), len(tgt_tok), cfg).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr, betas=(0.9, 0.98), eps=1e-9)
    sched = SimpleLRScheduler(optimizer)

    pad_src = src_tok.stoi[cfg.pad_token]
    pad_tgt = tgt_tok.stoi[cfg.pad_token]

    step = 0
    model.train()
    pbar = tqdm(total=cfg.steps)

    train_iter = iter(train_loader)
    while step < cfg.steps:
        try:
            src, tgt = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            src, tgt = next(train_iter)

        src = src.to(device)
        tgt = tgt.to(device)

        tgt_inp = tgt[:, :-1]
        tgt_out = tgt[:, 1:]

        src_keep = pad_keep_mask(src, pad_src).to(device)
        tgt_keep = pad_keep_mask(tgt_inp, pad_tgt).to(device)

        logits = model(src, tgt_inp, src_keep, tgt_keep)
        loss = label_smoothed_loss(logits, tgt_out, pad_tgt, cfg.label_smoothing)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        optimizer.step()
        lr = sched.step()

        step += 1
        pbar.update(1)
        pbar.set_postfix(loss=float(loss.item()), lr=float(lr))

        if step % cfg.eval_every == 0:
            print(f"\nRunning BLEU eval at step {step}...\n")
            if len(val_loader.dataset) == 0:
                print("VAL set is empty -> skipping BLEU eval.")
            else:
                bleu = evaluate_bleu(model, val_loader, src_tok, tgt_tok, cfg, device, max_sentences=200)
            print(f"Step {step}: VAL BLEU={bleu:.2f}\n")
            model.train()

    test_bleu = evaluate_bleu(model, test_loader, src_tok, tgt_tok, cfg, device, max_sentences=300)
    print(f"FINAL TEST BLEU={test_bleu:.2f}")

    ckpt = {
        "model": model.state_dict(),
        "cfg": cfg.__dict__,
        "src_tok": {"itos": src_tok.itos, "stoi": src_tok.stoi, "merges": src_tok.merges},
        "tgt_tok": {"itos": tgt_tok.itos, "stoi": tgt_tok.stoi, "merges": src_tok.merges if False else tgt_tok.merges},
    }
    torch.save(ckpt, "mt_transformer_scratch.pt")
    print("Saved: mt_transformer_scratch.pt")


if __name__ == "__main__":
    main()
