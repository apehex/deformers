# Ideas

List of experimental directions not yet validated.

## Prefix / Embedding

Composite byte embedding

- encode tokens as fixed-length UTF-8 byte sequences
- embed bytes using small vocabulary (256 entries)
- combine byte embeddings into token embeddings
- optionally apply encoder layer to map to hidden size

Potential advantages:

- remove dependency on large token embedding tables
- exploit internal structure of token strings
- allow deterministic reconstruction of embeddings
- keep the information on the token composition
- compress the embedding layers

Possible variants:

- convolution over byte dimension
- transformer over byte positions
- variable-length byte encodings

## Suffix / Output

Hierarchical softmax head

- replace flat vocabulary projection
- represent tokens as leaves of a binary tree
- predict path decisions instead of full vocabulary logits

Potential benefits:

- reduce output layer parameter count
- reduce compute for token sampling

Tree construction options:

- frequency-based (Huffman)
- embedding clustering
- BPE structure

## Training

Distillation-based patch training

Possible supervision signals:

- embedding regression
- hidden-state matching
- KL divergence on logits
- next-token cross entropy

Possible training stages:

1. local module pretraining
2. frozen-trunk alignment
3. optional partial trunk unfreezing
