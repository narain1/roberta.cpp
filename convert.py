import sys
import torch

from gguf import GGUFWriter, GGMLQuantizationType
import gguf
from typing import TYPE_CHECKING, Any, Callable, ClassVar, IO, Iterable, Literal, Protocol, TypeVar, runtime_checkable
from pathlib import Path
import json
import os

KEY_PAD_ID = 'tokenizer.ggml.padding_token_id'
KEY_UNK_ID = 'tokenizer.ggml.unknown_token_id'
KEY_BOS_ID = 'tokenizer.ggml.bos_token_id'
KEY_EOS_ID = 'tokenizer.ggml.eos_token_id'
KEY_WORD_PREFIX = 'tokenizer.ggml.word_prefix'
KEY_SUBWORD_PREFIX = 'tokenizer.ggml.subword_prefix'

@runtime_checkable
class BaseVocab(Protocol):
    tokenizer_model: ClassVar[str]
    name: ClassVar[str]


@runtime_checkable
class Vocab(BaseVocab, Protocol):
    vocab_size: int
    added_tokens_dict: dict[str, int]
    added_tokens_list: list[str]
    fname_tokenizer: Path

    def __init__(self, base_path: Path): ...
    def all_tokens(self) -> Iterable[tuple[bytes, float, gguf.TokenType]]: ...


def extract_vocab(vocab):
    tokens, scores, toktypes = [], [], []
    for text, score, toktype in vocab.bpe_tokens():
        tokens.append(text)
        scores.append(score)
        toktypes.append(toktype)

    assert len(tokens) == vocab.vocab_size
    return tokens, scores, toktypes

class BPEVocab:
    def __init__(self, base_path):
        added_tokens = dict()
        with open(os.path.join(base_path, "tokenizer.json"), encoding="utf-8") as f:
            tokenizer_json = json.load(f)

        tokenizer_model = tokenizer_json['model']
        self.vocab = tokenizer_model['vocab']
        self.vocab_size = len(self.vocab)
        if (added := tokenizer_json.get('added_tokens')) is not None:
            added_tokens = {item['content']: item['id']
                            for item in added if item['content'] not in self.vocab}
        vocab_size = len(self.vocab)
        expected_ids = list(range(vocab_size, vocab_size + len(added_tokens)))
        actual_ids = sorted(added_tokens.values())
        if expected_ids != actual_ids:
            expected_end_id = vocab_size + len(actual_ids) - 1

        items = sorted(added_tokens.items(), key=lambda text_idx: text_idx[1])
        self.added_tokens_dict = added_tokens
        self.added_tokens_list = [text for (text, idx) in items]
        self.vocab_size_base = vocab_size
        self.vocab_size = self.vocab_size_base + len(self.added_tokens_list)
        self.bos_token_id = 0
        self.pad_token_id = 1
        self.eos_token_id = 2
        self.unk_token_id = 3

    def bpe_tokens(self):
        reverse_vocab = {id: encoded_tok for encoded_tok, id in self.vocab.items()}
        for i, _ in enumerate(self.vocab):
            yield reverse_vocab[i], 0.0, gguf.TokenType.NORMAL


if __name__ == "__main__":
    with open('models/config.json', 'r') as f_read:
        config = json.load(f_read)

    for k,v in config.items():
        print(f'{k:32} = {v}')

    vocab = BPEVocab(Path("models"))
    model = torch.load("models/pytorch_model.bin")

    float_type = "f32"
    qtype = GGMLQuantizationType[float_type.upper()]
    dtype0 = {'f16': torch.float16, 'f32': torch.float32}[float_type]

    param_keys = [
        'vocab_size', 'max_position_embeddings', 'hidden_size', 'intermediate_size',
        'num_attention_heads', 'num_hidden_layers', 'layer_norm_eps'
    ]

    gguf_writer = GGUFWriter("roberta.gguf", 'roberta')
    gguf_writer.add_name("roberta")
    gguf_writer.add_description("ggml roberta model")
    gguf_writer.add_file_type(qtype)

    # writing model parameters
    gguf_writer.add_uint32("vocab_size", config['vocab_size'])
    gguf_writer.add_uint32("max_position_embedding", config['max_position_embeddings'])
    gguf_writer.add_uint32("hidden_size", config['hidden_size'])
    gguf_writer.add_uint32("intermediate_size", config['intermediate_size'])
    gguf_writer.add_uint32("num_attention_heads", config['num_attention_heads'])
    gguf_writer.add_uint32("num_hidden_layers", config['num_hidden_layers']);
    gguf_writer.add_float32("layer_norm_eps", config['layer_norm_eps'])

    # writing vocab parameters
    gguf_writer.add_int32(KEY_PAD_ID, vocab.pad_token_id)
    gguf_writer.add_int32(KEY_UNK_ID, vocab.unk_token_id)
    gguf_writer.add_int32(KEY_BOS_ID, vocab.bos_token_id)
    gguf_writer.add_int32(KEY_EOS_ID, vocab.eos_token_id)

    tokens, scores, toktypes = extract_vocab(vocab)

    gguf_writer.add_token_list(tokens)

    for n, p in model.items():
        if n.startswith(("lm_predictions", "mask_predictions")): continue
        if 'LayerNorm' in n or 'bias' in n:
            dtype = torch.float32
        else:
            dtype = dtype0

        shape_str = str(list(p.shape))
        print(f'{n:64s} = {shape_str:16s} {p.dtype} → {dtype}')

        p = p.to(dtype)
        gguf_writer.add_tensor(n, p.numpy())

    gguf_writer.write_header_to_file()
    gguf_writer.write_kv_data_to_file()
    gguf_writer.write_tensors_to_file()
    gguf_writer.close()
