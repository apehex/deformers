import torch
import torch.nn

import mlable.layers.embedding

# STAGE A - COMPOSITE BYTE PREFIX ##############################################

class CompositeBytePrefix(torch.nn.Module):
    """
    Stage A alternative prefix: byte block embeddings projected to the hidden
    space of the base language model.

    Assumptions:
    - Base model is qwen/qwen3.5-9b with hidden_size=4096.
    - Tokenizer boundaries are identical to the base model.
    - Trunk and lm_head are frozen; only this module is trainable.
    - Byte block size defaults to L_max=32 (see docs/roadmap.md).
    - The byte tokenizer uses pad_id=128 (as in ByteTokenizer).

    Input modes:
    - group_dim <= 0: inputs must be rank-3 (B, T, G) already split into byte
      blocks; CompositeEmbedding is called with group_dim=-1.
    - group_dim > 0: inputs may be rank-2 (B, T*G) flat; CompositeEmbedding
      splits them by group_dim internally.

    Shape contract:
        inputs_arr: (B, T, G) long  [group_dim <= 0 mode]
              or   (B, T*G)   long  [group_dim >  0 mode]
        return:     (B, T, latent_dim) float
    """

    def __init__(
        self,
        embed_dim: int,
        vocab_dim: int=256,
        latent_dim: int=-1,
        group_dim: int=-1,
    ) -> None:
        super(CompositeBytePrefix, self).__init__()
        # save for import / export
        self._config = {
            'embed_dim': int(embed_dim),
            'vocab_dim': int(vocab_dim),
            'latent_dim': int(latent_dim),
            'group_dim': int(group_dim),}
        # submodule, registered once at first forward call
        self._layers = None
        self._built = False

    def _build(self, shape_arr: tuple, device=None) -> None:
        if not self._built:
            # actual group dimension: last dim of input when not configured
            __group_dim = shape_arr[-1] if (self._config['group_dim'] <= 0) else self._config['group_dim']
            # merged byte embedding dimension after CompositeEmbedding
            __embed_dim = __group_dim * self._config['embed_dim']
            # projection target dimension: defaults to merged embed dim
            __latent_dim = __embed_dim if (self._config['latent_dim'] < 0) else self._config['latent_dim']
            # build the sequential pipeline
            self._layers = torch.nn.Sequential(
                # (B, T, G) => (B, T, G*E)
                mlable.layers.embedding.CompositeEmbedding(
                    input_dim=self._config['vocab_dim'],
                    output_dim=self._config['embed_dim'],
                    group_dim=self._config['group_dim'],
                    merge_axes=True),
                # (B, T, G*E) => (B, T, G*E)
                torch.nn.LayerNorm(
                    normalized_shape=__embed_dim,
                    bias=True),
                # (B, T, G*E) => (B, T, H)
                torch.nn.Linear(
                    in_features=__embed_dim,
                    out_features=__latent_dim,
                    bias=True),
                # (B, T, H) => (B, T, H)
                torch.nn.SiLU(),
                # (B, T, H) => (B, T, H)
                torch.nn.Linear(
                    in_features=__latent_dim,
                    out_features=__latent_dim,
                    bias=True),
                # (B, T, H) => (B, T, H)
                torch.nn.LayerNorm(
                    normalized_shape=__latent_dim,
                    bias=True))
            # move to the target device at build time (no-op if device is None)
            if device is not None:
                self._layers = self._layers.to(device=device)
            self._built = True

    def forward(self, inputs_arr: torch.Tensor) -> torch.Tensor:
        __shape = tuple(inputs_arr.shape)
        __group = self._config.get('group_dim', -1)
        # unspecified group dimension: the inputs must already be split into blocks
        if __group <= 0:
            assert len(__shape) == 3, (
                'Inputs must be rank 3 (B, T, G), got shape={} with group_dim={}'.format(__shape, __group))
        # explicit group dimension: the inputs are expected to be flat sequences
        else:
            assert len(__shape) == 2, (
                'Inputs must be rank 2 (B, T*G), got shape={} with group_dim={}'.format(__shape, __group))
        # lazy init: build sub-layers on first call, placed on input device
        self._build(__shape, device=inputs_arr.device)
        # (B, T, G) => (B, T, H) or (B, T*G) => (B, T, H)
        return self._layers(inputs_arr.long())
