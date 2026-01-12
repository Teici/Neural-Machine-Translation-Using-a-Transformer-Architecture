from dataclasses import dataclass

@dataclass
class Config:
    # Data sources 
    use_opus_books: bool = True
    opus_books_name: str = "opus_books"
    opus_books_config: str = "de-en"   # has both "en" and "de" fields

    use_open_subtitles: bool = True
    open_subtitles_name: str = "open_subtitles"
    open_subtitles_lang1: str = "en"
    open_subtitles_lang2: str = "de"

    use_tatoeba: bool = False
    tatoeba_name: str = "tatoeba"
    tatoeba_lang1: str = "en"
    tatoeba_lang2: str = "de"

    # Mix ratios (must sum roughly to 1.0)
    mix_open_subtitles: float = 0.85
    mix_opus_books: float = 0.15
    mix_tatoeba: float = 0.0

    # Total sizes AFTER mixing
    train_size: int = 60000
    val_size: int = 2000
    test_size: int = 2000

    max_len: int = 60
    lowercase: bool = True

    #  BPE tokenizer
    bpe_vocab_size: int = 12000
    bpe_num_merges: int = 12000

    pad_token: str = "<pad>"
    bos_token: str = "<bos>"
    eos_token: str = "<eos>"
    unk_token: str = "<unk>"

    # Transformer
    d_model: int = 256
    n_heads: int = 8
    d_ff: int = 1024
    n_layers: int = 4
    dropout: float = 0.1

    # Training
    batch_size: int = 32
    lr: float = 5e-4
    steps: int = 8000
    eval_every: int = 1000
    label_smoothing: float = 0.1
    grad_clip: float = 1.0

    # Decoding
    max_decode_len: int = 60
    beam_size: int = 6
    length_penalty: float = 0.8

    device: str = "cpu"  
