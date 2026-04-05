"""
Stage A prefix patch training.

Trains the CompositeBytePrefix module to distill from qwen/qwen3.5-9b by
injecting prefix outputs via inputs_embeds into the frozen trunk.

Assumptions:
- Base model is qwen/qwen3.5-9b with hidden_size=4096.
- Tokenizer boundaries are identical to the base model.
- Trunk is frozen during training; only the prefix parameters are trained.
- Byte block size default follows docs/roadmap.md (L_max=32), configurable.
- The byte tokenizer uses pad_id=128 (as implemented by ByteTokenizer).

Training scheme (Stage A):
- Teacher: qwen/qwen3.5-9b forward with original input_ids (no grad).
- Student: CompositeBytePrefix forward then trunk forward with inputs_embeds.
- Loss: hidden-state MSE at depth k (configurable), optional embedding MSE.
- Only prefix parameters are updated; trunk is frozen.
"""

import os

import torch
import torch.amp
import torch.nn
import transformers

import deformers.layers.prefix
import deformers.patching.bytes
import deformers.tokenizers.byte

# CONFIG #######################################################################

BYTE_BLOCK_SIZE = 32   # L_max, see docs/roadmap.md
HIDDEN_LAYER_K = 4     # hidden state depth for distillation target
EMBED_DIM = 32         # byte embedding dim per position: 32 * 32 = 1024 intermediate
BATCH_SIZE = 4
SEQ_LEN = 128          # max token sequence length per sample
NUM_STEPS = 200
LEARNING_RATE = 3e-4
GRAD_ACCUM_STEPS = 4
LOSS_WEIGHT_HIDDEN = 1.0
LOSS_WEIGHT_EMBED = 0.1
SAVE_PATH = 'checkpoints/prefix_stage_a.pt'
LOG_EVERY = 20

TOKEN_CFG = {
    'pretrained_model_name_or_path': 'qwen/qwen3.5-9b',
    'use_fast': True,}

MODEL_CFG = {
    'pretrained_model_name_or_path': 'qwen/qwen3.5-9b',
    'device_map': 'cuda' if torch.cuda.is_available() else 'cpu',
    'torch_dtype': torch.bfloat16,}

# FALLBACK TEXT (used when HF datasets are unavailable) ########################

TEXT_CFG = [
    'Lexical tokenization is conversion of a text into meaningful tokens.',
    'In case of a natural language, those categories include nouns and verbs.',
    'In case of a programming language, they include identifiers and operators.',
    'Lexical tokenization is related to large language models but with differences.',
    'First, lexical tokenization is usually based on a lexical grammar.',
    'Second, LLM tokenizers perform a step that converts tokens into numerical values.',
    'The quick brown fox jumps over the lazy dog.',
    'Machine learning models require large amounts of training data.',
]

# DATASET ######################################################################

def load_texts(num_samples: int=512) -> list:
    """
    Load a small text dataset for prefix training.

    Falls back to a local constant if wikitext is unavailable.
    """
    try:
        import datasets
        __ds = datasets.load_dataset('wikitext', 'wikitext-2-raw-v1', split='train')
        __texts = [
            __r['text']
            for __r in __ds
            if len(__r['text'].strip()) > 64][:num_samples]
        if __texts:
            return __texts
    except Exception:
        pass
    # repeat the fallback list to reach num_samples approximately
    __cycle = TEXT_CFG * (num_samples // len(TEXT_CFG) + 1)
    return __cycle[:num_samples]

# TRAINING #####################################################################

def freeze_model(model: torch.nn.Module) -> None:
    """Disable gradients for all model parameters (trunk stays frozen)."""
    for __p in model.parameters():
        __p.requires_grad_(False)


def get_teacher_embeddings(model: torch.nn.Module, input_ids: torch.Tensor) -> torch.Tensor:
    """Return the embedding layer output for the given token ids."""
    # works for both Qwen2Model and Qwen3Model hierarchy
    __embed = model.model.embed_tokens
    return __embed(input_ids)


def train(
    token_cfg: dict=TOKEN_CFG,
    model_cfg: dict=MODEL_CFG,
    byte_block_size: int=BYTE_BLOCK_SIZE,
    embed_dim: int=EMBED_DIM,
    hidden_layer_k: int=HIDDEN_LAYER_K,
    batch_size: int=BATCH_SIZE,
    seq_len: int=SEQ_LEN,
    num_steps: int=NUM_STEPS,
    learning_rate: float=LEARNING_RATE,
    grad_accum_steps: int=GRAD_ACCUM_STEPS,
    loss_weight_hidden: float=LOSS_WEIGHT_HIDDEN,
    loss_weight_embed: float=LOSS_WEIGHT_EMBED,
    save_path: str=SAVE_PATH,
    log_every: int=LOG_EVERY,
) -> deformers.layers.prefix.CompositeBytePrefix:
    __device = 'cuda' if torch.cuda.is_available() else 'cpu'
    __use_amp = (__device == 'cuda')

    # load tokenizer and model
    print('[train] loading tokenizer...')
    __tokenizer = transformers.AutoTokenizer.from_pretrained(**token_cfg)
    __byte_tok = deformers.tokenizers.byte.ByteTokenizer()

    print('[train] loading base model...')
    __model = transformers.AutoModelForCausalLM.from_pretrained(**model_cfg)
    __model.eval()
    freeze_model(__model)

    __hidden_size = __model.config.hidden_size  # 4096 for qwen3.5-9b
    print(f'[train] hidden_size={__hidden_size}, byte_block_size={byte_block_size}')

    # build prefix module
    __prefix = deformers.layers.prefix.CompositeBytePrefix(
        embed_dim=embed_dim,
        vocab_dim=256,
        latent_dim=__hidden_size,
        group_dim=-1,   # inputs will be (B, T, G) rank-3
    ).to(__device)

    __optimizer = torch.optim.AdamW(__prefix.parameters(), lr=learning_rate)
    __scaler = torch.amp.GradScaler(enabled=__use_amp)

    # load dataset
    print('[train] loading dataset...')
    __texts_all = load_texts(num_samples=num_steps * batch_size)

    print('[train] starting training loop...')
    __step = 0
    __accum_loss = 0.0

    __optimizer.zero_grad()

    for __i in range(num_steps * grad_accum_steps):
        # sample a batch
        __batch_start = (__i * batch_size) % max(1, len(__texts_all) - batch_size)
        __batch_texts = __texts_all[__batch_start: __batch_start + batch_size]

        # encode: input_ids (B, T), attention_mask (B, T), byte_ids (B, T, G)
        try:
            __encoded = deformers.patching.bytes.encode_texts(
                __batch_texts,
                tokenizer=__tokenizer,
                byte_tokenizer=__byte_tok,
                max_length=byte_block_size)
        except Exception as __e:
            print(f'[train] encoding error at step {__step}: {__e}')
            continue

        __input_ids = __encoded['input_ids'].to(__device)
        __attention_mask = __encoded['attention_mask'].to(__device)
        __byte_ids = __encoded['byte_ids'].to(__device)

        # truncate sequences to seq_len
        if __input_ids.shape[1] > seq_len:
            __input_ids = __input_ids[:, :seq_len]
            __attention_mask = __attention_mask[:, :seq_len]
            __byte_ids = __byte_ids[:, :seq_len, :]

        # teacher forward: get original embeddings and hidden states (no grad)
        with torch.no_grad():
            __teacher_embeds = get_teacher_embeddings(__model, __input_ids)
            __teacher_out = __model(
                input_ids=__input_ids,
                attention_mask=__attention_mask,
                output_hidden_states=True,
                use_cache=False)
            __teacher_hidden_k = __teacher_out.hidden_states[hidden_layer_k].detach()
            __teacher_embeds = __teacher_embeds.detach()

        # student forward: prefix -> inputs_embeds -> trunk -> hidden_k
        __amp_ctx = torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16) if __use_amp else torch.amp.autocast(device_type='cpu', enabled=False)
        with __amp_ctx:
            __student_embeds = __prefix(__byte_ids)
            __student_out = __model(
                inputs_embeds=__student_embeds,
                attention_mask=__attention_mask,
                output_hidden_states=True,
                use_cache=False)
            __student_hidden_k = __student_out.hidden_states[hidden_layer_k]

            # hidden-state MSE at depth k
            __loss_hidden = torch.nn.functional.mse_loss(__student_hidden_k, __teacher_hidden_k)
            # optional embedding MSE warmup
            __loss_embed = torch.nn.functional.mse_loss(
                __student_embeds.float(), __teacher_embeds.float())
            __loss = loss_weight_hidden * __loss_hidden + loss_weight_embed * __loss_embed
            __loss = __loss / grad_accum_steps

        __scaler.scale(__loss).backward()
        __accum_loss += __loss.item() * grad_accum_steps

        # optimizer step after gradient accumulation
        if (__i + 1) % grad_accum_steps == 0:
            __scaler.unscale_(__optimizer)
            torch.nn.utils.clip_grad_norm_(__prefix.parameters(), max_norm=1.0)
            __scaler.step(__optimizer)
            __scaler.update()
            __optimizer.zero_grad()

            if __step % log_every == 0:
                print(f'[train] step={__step:04d} loss={__accum_loss / log_every:.6f}')
                __accum_loss = 0.0

            __step += 1

    # save prefix weights
    __dir = os.path.dirname(save_path)
    if __dir:
        os.makedirs(__dir, exist_ok=True)
    torch.save({'config': __prefix._config, 'state_dict': __prefix.state_dict()}, save_path)
    print(f'[train] saved prefix to {save_path}')

    return __prefix


# ENTRY POINT ##################################################################

if __name__ == '__main__':
    train()
