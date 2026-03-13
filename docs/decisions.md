# Decisions

## Tokenization

Keep the same partitioning as the tokenizer from the original model.

It guarantees that the input tokens are the same regardless of the model's prefix.

```python
text = 'This avoids the segmentation mismatch with fixed size patching.'
print([text[__s:__e] for (__s, __e) in tokenizer(text, return_offsets_mapping=True)['offset_mapping']])
# ['This', ' avoids', ' the', ' segmentation', ' mismatch', ' with', ' fixed', ' size', ' patch', 'ing', '.']
print([text[__s:__s + 8] for __s in range(0, len(text), 8)])
# ['This avo', 'ids the ', 'segmenta', 'tion mis', 'match wi', 'th fixed', ' size pa', 'tching.']
```

Pad the shorter tokens with null bytes and truncate the longer tokens so that the encoding is fixed to 32 bytes.

This impacts less than 3% of the tokens, which look like data (binary) tokens:

```python
print(len(tokenizer.get_vocab())) # total number of tokens
# 248077
print(len([__t for __t in tokenizer.get_vocab().keys() if len(__t.encode('utf-8')) > 32])) # number of truncated tokens
# 6365
print(100 * 6365 / 248077) # percentage of truncated tokens
# 2.565735638531585
print([__t for __t in tokenizer.get_vocab().keys() if len(__t.encode('utf-8')) > 32][:8])
# ['ĠÐ·Ð°ÑģÑĤÐ°Ð²Ð»ÑıÐµÑĤ',
#  'ĠÑĤÐµÑħÐ½Ð¸ÑĩÐµÑģÐºÐ¸Ñħ',
#  'ĠÐ¿Ð¾ÑģÑĤÐ¾ÑıÐ½Ð½Ð¾Ð³Ð¾',
#  'à¸Ĺà¸µà¹Ģà¸Ķà¸µà¸¢à¸§',
#  'ĠÐĹÐ°ÐºÐ¾Ð½Ð¾Ð´Ð°',
#  'à¸Ļà¸°à¸Ħà¸£à¸±à¸ļ',
#  'à¸£à¸¸à¹Īà¸Ļà¹ĥà¸«à¸¡à¹Ī',
#  'ĠÑģÐµÑĢÑĤÐ¸ÑĦÐ¸ÐºÐ°']
print([__t.encode('utf-8').hex() for __t in TOKENIZER_OBJ.get_vocab().keys() if len(__t.encode('utf-8')) > 32][:8]) # hex encoded, since these are likely data tokens
# ['c4a0c390c2bfc391c4a2c390c2bec391c4a7c390c2bec390c2b4c390c2b8c391c4a4c391c4ae',
#  'c3a0c2b9c4a2c3a0c2b8c4acc3a0c2b8c2b7c3a0c2b9c4aac3a0c2b8c583c3a0c2b8c2a1c3a0c2b9c4a4c3a0c2b8c2a2c3a0c2b8c4a9',
#  'c4a0c391c4a5c391c4a9c390c2b0c391c4a3c391c4a4c390c2bac390c2bec390c2b2',
#  'c3a0c2b8c2a2c3a0c2b8c2b2c3a0c2b8c2a1c3a0c2b8c4a6c3a0c2b9c4aac3a0c2b8c2b3c3a0c2b8c4a6c3a0c2b8c2b7c3a0c2b8c4bb',
#  'c4a0c3a0c2b8c4b7c3a0c2b8c2b2c3a0c2b8c2a1c3a0c2b8c2a3c3a0c2b8c2b0c3a0c2b8c4b6c3a0c2b8c2b1c3a0c2b8c4bc',
#  'c3a0c2b8c2a3c3a0c2b8c2b2c3a0c2b8c2a2c3a0c2b8c4a9c3a0c2b8c2b2c3a0c2b8c4bb',
#  'c3a0c2b8c2b3c3a0c2b8c4bbc3a0c2b8c2a7c3a0c2b8c2a2c3a0c2b8c4a3c3a0c2b8c2b2c3a0c2b8c2a3',
#  'c4a0c390c2b4c390c2b5c390c2b9c391c4a3c391c4a4c390c2b2c390c2bec390c2b2c390c2b0c391c4a4c391c4ae']
```
