import torch
import torch.nn

import mlable.layers.embedding

# BYTE #########################################################################

class CompositeBytePrefix(torch.nn.Module):
    def __init__(
        self,
        embed_dim: int,
        vocab_dim: int=256,
        latent_dim: int=-1,
        group_dim: int=-1,
        **kwargs: dict,
    ) -> None:
        super(CompositeBytePrefix, self).__init__(**kwargs)
        # save for import / export
        self._config = {
            'embed_dim': embed_dim,
            'vocab_dim': vocab_dim,
            'latent_dim': latent_dim,
            'group_dim': group_dim,}
        # build at runtime
        self._layers = None
        self._built = False

    def _build(
        self,
        shape_arr: tuple
    ) -> None:
        # lazy build at runtime
        if not self._built:
            # the group dimension defaults to the last dimension of the input
            __group_dim = shape_arr[-1] if (self._config['group_dim'] <= 1) else self._config['group_dim']
            # the dimension of the composite embeddings
            __embed_dim = __group_dim * self._config['embed_dim']
            # the output dimension defaults to the dimension of the composite embeddings
            __latent_dim = __embed_dim if (self._config['latent_dim'] < 0) else self._config['latent_dim']
            # chain all the layers
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
                # (B, T, G*E) => (B, T, G*E)
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
        # register
        self._built = True

    def forward(
        self,
        inputs_arr: torch.Tensor
    ) -> torch.Tensor:
        __shape = tuple(inputs_arr.shape)
        __group = self._config.get('group_dim', -1)
        # unspecified group dimension: the inputs must already be split
        if __group <= 1:
            assert len(__shape) == 3, f'Inputs must be rank 3, got shape={__shape} with group={__group}'
        # otherwise, the inputs are supposed to be flattened
        else:
            assert len(__shape) == 2, f'Inputs must be rank 2, got shape={__shape} with group={__group}'
        # initialize all the sub-layers
        self._build(__shape)
        # (B, T, G) => (B, T, H) or (B, T*G) => (B, T, H)
        return self._layers(inputs_arr)
