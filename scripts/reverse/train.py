import torch
import torch.nn as nn
import torch.nn.functional as F
from accelerate import Accelerator
from transformers import AutoTokenizer, AutoModelForCausalLM

# ----------------------------
# Reverse model (bidirectional transformer)
# ----------------------------
class ReverseTransformer(nn.Module):
    def __init__(self, d_model: int, n_layers: int = 6, n_heads: int = 16, mlp_ratio: int = 4, dropout: float = 0.0):
        super().__init__()
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=mlp_ratio * d_model,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self.out_norm = nn.LayerNorm(d_model)

    def forward(self, h: torch.Tensor, attn_mask: torch.Tensor | None = None) -> torch.Tensor:
        # h: (B, T, d)
        # attn_mask: optional key padding mask -> (B, T) with True for pad positions
        x = self.encoder(h, src_key_padding_mask=attn_mask)
        return self.out_norm(x)  # (B, T, d)

    @torch.no_grad()
    def nearest_tokens(self, hat_e: torch.Tensor, embed_weight: torch.Tensor, topk: int = 1):
        """
        hat_e: (B, T, d)
        embed_weight: (V, d)  token embedding matrix
        returns: ids (B, T) if topk=1 else (B, T, topk)
        """
        # cosine similarity
        e = F.normalize(hat_e, dim=-1)
        w = F.normalize(embed_weight, dim=-1)
        scores = torch.matmul(e, w.t())  # (B, T, V)
        if topk == 1:
            return scores.argmax(dim=-1)
        return scores.topk(k=topk, dim=-1).indices

# ----------------------------
# Losses
# ----------------------------
def loss_l2_cos_ce(hat_e, tgt_e, input_ids, embed_weight, l2_w=1.0, cos_w=0.1, ce_w=0.05):
    # L2
    l2 = F.mse_loss(hat_e, tgt_e)

    # cosine
    cos = 1.0 - F.cosine_similarity(hat_e, tgt_e, dim=-1).mean()

    # CE using tied embeddings as classifier
    # logits: (B, T, V)
    logits = torch.matmul(hat_e, embed_weight.t())
    ce = F.cross_entropy(logits.view(-1, logits.size(-1)), input_ids.view(-1), ignore_index=-100)

    return l2_w * l2 + cos_w * cos + ce_w * ce, {"l2": l2.item(), "cos": cos.item(), "ce": ce.item()}

# ----------------------------
# Main training step (online trunk)
# ----------------------------
def train():
    accelerator = Accelerator(mixed_precision="bf16")  # or "fp16"
    device = accelerator.device

    model_name = "openai/gpt-oss-20b"
    LAYER = 12  # choose target layer index
    MAX_LEN = 512

    tok = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    llm = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16, device_map=None)
    llm.eval()
    for p in llm.parameters():
        p.requires_grad_(False)

    # token embedding matrix
    embed = llm.get_input_embeddings()  # nn.Embedding
    d_model = embed.embedding_dim

    rev = ReverseTransformer(d_model=d_model, n_layers=6, n_heads=16)

    opt = torch.optim.AdamW(rev.parameters(), lr=2e-4, betas=(0.9, 0.95), weight_decay=0.1)

    rev, opt = accelerator.prepare(rev, opt)
    llm.to(device)

    # dummy dataloader: replace with datasets + DataLoader
    texts = ["Hello world", "This is a test", "Reverse hidden states into embeddings"]
    def batch():
        batch_txt = texts  # replace with real sampling
        enc = tok(batch_txt, return_tensors="pt", padding=True, truncation=True, max_length=MAX_LEN)
        return enc["input_ids"], enc["attention_mask"]

    for step in range(1000):
        input_ids, attn_mask = batch()
        input_ids = input_ids.to(device)
        attn_mask = attn_mask.to(device)

        # target embeddings E
        with torch.no_grad():
            tgt_e = embed(input_ids)  # (B, T, d)

        # trunk hidden states H_L
        with torch.no_grad(), accelerator.autocast():
            out = llm(
                input_ids=input_ids,
                attention_mask=attn_mask,
                output_hidden_states=True,
                use_cache=False,
                return_dict=True,
            )
            # hidden_states is tuple: (embeddings_out, layer1_out, ..., layerN_out)
            hL = out.hidden_states[LAYER]  # (B, T, d)

        # key padding mask for nn.TransformerEncoder: True where PAD
        key_pad = (attn_mask == 0)

        with accelerator.autocast():
            hat_e = rev(hL, attn_mask=key_pad)
            loss, logs = loss_l2_cos_ce(
                hat_e=hat_e,
                tgt_e=tgt_e,
                input_ids=input_ids,
                embed_weight=embed.weight,
                l2_w=1.0,
                cos_w=0.1,
                ce_w=0.05,
            )

        accelerator.backward(loss)
        opt.step()
        opt.zero_grad(set_to_none=True)

        if accelerator.is_main_process and step % 50 == 0:
            accelerator.print(f"step={step} loss={loss.item():.4f} {logs}")

if __name__ == "__main__":
    train()
