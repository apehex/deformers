import gc
from dataclasses import dataclass
from typing import Any

import torch
import torch.cuda
import torch.nn
import transformers

import deformers.tokenizers.byte

# META #########################################################################

TOKEN_CFG = {
    'pretrained_model_name_or_path': 'qwen/qwen3.5-9b',
    'dtype': 'auto',
    'use_fast': True,}

MODEL_CFG = {
    'pretrained_model_name_or_path': 'qwen/qwen3.5-9b', # 'openai/gpt-oss-20b' 'qwen/qwen3.5-9b' 'qwen/qwen3.5-27b' 'google/gemma-3-27b-it'
    # 'attn_implementation': 'eager',
    'device_map': 'cuda' if torch.cuda.is_available() else 'cpu',}

TEXT_CFG = {
    'wiki': '''Lexical tokenization is conversion of a text into (semantically or syntactically) meaningful lexical tokens belonging to categories defined by a "lexer" program. In case of a natural language, those categories include nouns, verbs, adjectives, punctuations etc. In case of a programming language, the categories include identifiers, operators, grouping symbols, data types and language keywords. Lexical tokenization is related to the type of tokenization used in large language models (LLMs) but with two differences. First, lexical tokenization is usually based on a lexical grammar, whereas LLM tokenizers are usually probability-based. Second, LLM tokenizers perform a second step that converts the tokens into numerical values.''',}

# PATCHING #####################################################################

@dataclass
class ReversibleBytePatching:
    input_ids: list[list[int]]
    lengths: list[int]
    overflow_ids: list[list[int]]
    truncated: list[bool]
    max_length: int
    encoding: str

def _encode_single_token(token: str, max_length: int, encoding: str) -> tuple[list[int], int, list[int], bool]:
    __bytes = list(token.encode(encoding))
    __length = len(__bytes)
    __keep = __bytes[:max_length]
    __overflow = __bytes[max_length:]
    __padded = __keep + [0] * max(0, max_length - len(__keep))
    return __padded, __length, __overflow, (__length > max_length)

def encode_tokens_reversible(tokens: list[str], max_length: int=32, encoding: str='utf-8') -> ReversibleBytePatching:
    __encoded = [_encode_single_token(token=__t, max_length=max_length, encoding=encoding) for __t in tokens]
    return ReversibleBytePatching(
        input_ids=[__entry[0] for __entry in __encoded],
        lengths=[__entry[1] for __entry in __encoded],
        overflow_ids=[__entry[2] for __entry in __encoded],
        truncated=[__entry[3] for __entry in __encoded],
        max_length=max_length,
        encoding=encoding,)

def decode_tokens_reversible(patches: ReversibleBytePatching, errors: str='strict') -> list[str]:
    __tokens = []
    for __ids, __length, __overflow in zip(patches.input_ids, patches.lengths, patches.overflow_ids):
        __prefix = __ids[:min(__length, patches.max_length)]
        __bytes = bytes(__prefix + __overflow)
        __tokens.append(__bytes.decode(patches.encoding, errors=errors))
    return __tokens

def qwen_token_spans(text: str, tokenizer: Any) -> list[str]:
    __offsets = tokenizer(text, return_offsets_mapping=True, add_special_tokens=False)['offset_mapping']
    return [text[__s:__e] for (__s, __e) in __offsets]

# RESET ########################################################################

def free_memory(model: torch.nn.modules.Module) -> None:
    model.cpu()
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

if __name__ == "__main__":
    BYTE_OBJ = deformers.tokenizers.byte.ByteTokenizer()
    TOKEN_OBJ = transformers.AutoTokenizer.from_pretrained(**TOKEN_CFG)
    MODEL_OBJ = transformers.AutoModelForCausalLM.from_pretrained(**MODEL_CFG)
    __texts = TEXT_CFG['wiki'].split('. ')
    __tokens = [qwen_token_spans(text=__t, tokenizer=TOKEN_OBJ) for __t in __texts]
    __patches = [encode_tokens_reversible(tokens=__sample, max_length=32, encoding=BYTE_OBJ._encoding) for __sample in __tokens]
    __decoded = [decode_tokens_reversible(__sample) for __sample in __patches]
    assert __tokens == __decoded
