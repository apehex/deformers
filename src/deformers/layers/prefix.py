import os.path

import torch
import torch.nn

import mlable.layers.embedding
import mlable.layers.normalization
import mlable.layers.shaping

# META #########################################################################

SHAPE_MSG = 'Inputs must be rank {}, got shape={} with group_dim={}'

# BYTE #########################################################################

class CompositeBytePrefix(torch.nn.Module):
    """
    Alternative prefix: composite byte embeddings projected to the hidden
    space of the base language model.

    Assumptions:
    - Base model is qwen/qwen3.5-9b with hidden_size=4096.
    - Tokenizer boundaries are identical to the base model.
    - Trunk and lm_head are frozen; only this module is trainable.
    - Byte block size defaults to L_max=32 (see docs/roadmap.md).
    - The byte tokenizer uses pad_id=128 (as in ByteTokenizer).

    Inputs:
    - group_dim <= 0: inputs must be rank-3 (B, T, G) already split into byte
      blocks; CompositeEmbedding is called with group_dim=-1.
    - group_dim > 0: inputs may be rank-2 (B, T*G) flat; CompositeEmbedding
      splits them by group_dim internally.

    Outputs:
    Always (B, T, H) float.
    """

    def __init__(
        self,
        embed_dim: int,
        vocab_dim: int=256,
        latent_dim: int=-1,
        group_dim: int=-1,
        norm_opt: bool=True,
        attn_opt: bool=True,
        **kwargs: dict,
    ) -> None:
        super(CompositeBytePrefix, self).__init__(**kwargs)
        # save for import / export
        self._config = {
            'embed_dim': int(embed_dim),
            'vocab_dim': int(vocab_dim),
            'latent_dim': int(latent_dim),
            'group_dim': int(group_dim),
            'norm_opt': bool(norm_opt),
            'attn_opt': bool(attn_opt),}
        # submodule, initialized on first forward call
        self._layers = None
        self._built = False

    def build(
        self,
        shape: tuple,
        device: object=None,
        dtype: object=None,
    ) -> None:
        if not self._built:
            __layers = []
            # actual group dimension: last dim of input when not configured
            __group_dim = shape[-1] if (self._config['group_dim'] < 1) else self._config['group_dim']
            # merged byte embedding dimension after CompositeEmbedding
            __embed_dim = __group_dim * self._config['embed_dim']
            # projection target dimension: defaults to merged embed dim
            __latent_dim = __embed_dim if (self._config['latent_dim'] < 1) else self._config['latent_dim']
            # divide only if necessary (B, T*G) => (B, T, G) or (B, T, G) => (B, T, G)
            __layers.append(mlable.layers.shaping.Divide(
                axis=-1,
                factor=max(1, self._config['group_dim']),
                insert=bool(self._config['group_dim'] > 1),
                right=bool(self._config['group_dim'] > 1)))
            # (B, T, G) => (B, T, G, E)
            __layers.append(torch.nn.Embedding(
                num_embeddings=self._config['vocab_dim'],
                embedding_dim=self._config['embed_dim']))
            # (B, T, G, E) => (B, T, G, E)
            if self._config['attn_opt']:
                __layers.append(torch.nn.MultiheadAttention(
                    embed_dim=self._config['embed_dim'],
                    num_heads=4,
                    batch_first=True,
                    bias=True))
            # (B, T, G, E) => (B, T, G*E)
            __layers.append(mlable.layers.shaping.Merge(
                axis=-1,
                right=False))
            # (B, T, G*E) => (B, T, G*E)
            if self._config['norm_opt']:
                __layers.append(torch.nn.RMSNorm(
                    normalized_shape=(__embed_dim,),
                    elementwise_affine=True))
            # (B, T, G*E) => (B, T, G*E)
            __layers.append(torch.nn.Linear(
                in_features=__embed_dim,
                out_features=__embed_dim,
                bias=True))
            # (B, T, G*E) => (B, T, G*E)
            __layers.append(torch.nn.SiLU())
            # (B, T, G*E) => (B, T, H)
            __layers.append(torch.nn.Linear(
                in_features=__embed_dim,
                out_features=__latent_dim,
                bias=True))
            # chain together the layers
            self._layers = torch.nn.Sequential(*__layers)
            # move to the target device at build time (no-op if device is None)
            self._layers = self._layers.to(device=device, dtype=dtype)
            # register the build
            self._built = True

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        __shape = tuple(inputs.shape)
        __group = self._config.get('group_dim', -1)
        # unspecified group dimension: the inputs must already be split into blocks
        if __group <= 0:
            assert len(__shape) == 3,SHAPE_MSG.format(3, __shape, __group)
        # explicit group dimension: the inputs are expected to bea batch of flat sequences
        else:
            assert len(__shape) == 2, SHAPE_MSG.format(2, __shape, __group)
        # lazy init: build sub-layers on first call, placed on input device
        self.build(shape=__shape, device=inputs.device, dtype=None)
        # (B, T, G) => (B, T, H) or (B, T*G) => (B, T, H)
        return self._layers(inputs.to(dtype=torch.long))

    def output_shape(self, shape: tuple) -> tuple:
        # shape after embedding (B, T, G*E)
        __shape = self._layers[0].output_shape(shape)
        # default to G*E when H is not specified
        __dim = __shape[-1] if (self._config['latent_dim'] < 1) else self._config['latent_dim']
        # update the last dimension
        return __shape[:-1] + (__dim,)

    def get_config(self) -> dict:
        return dict(self._config)

    def save_checkpoint(self, path: str) -> None:
        torch.save({'config': self._config, 'state': self.state_dict()}, path)

    @classmethod
    def from_config(cls, config: dict, **kwargs: dict) -> torch.nn.Module:
        return cls(**{**config, **kwargs})

    @classmethod
    def load_checkpoint(
        cls,
        path: str,
        shape: tuple,
        device: object=None,
        **kwargs: dict
    ) -> torch.nn.Module:
        # check the disk
        assert os.path.isfile(path), f'model checkpoint not found: {path}'
        # parse the data
        __ckpt = torch.load(path, map_location=device, weights_only=True)
        # instantiate the model
        __prefix = cls.from_config(config=__ckpt['config'], **kwargs)
        # create the layers
        __prefix.build(shape=shape, device=None, dtype=None)
        # load the weights
        __prefix.load_state_dict(__ckpt['state'])
        # alternative transformer prefix
        return __prefix.to(device=device)
