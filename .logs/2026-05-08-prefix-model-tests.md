Updated prefix architecture coverage to match current implementation:
- removed legacy `tests/deformers/layers/test_prefix.py` tied to `deformers.layers.prefix.CompositeBytePrefix`
- added model-focused tests in `tests/deformers/models/test_prefix.py` for block stack, byte-axis attention, I/O shapes, padding-mask flow, and token-axis independence
- updated integration smoke tests to import `deformers.models.prefix.CompositeBytePrefix`
- refreshed roadmap architecture status to point to `src/deformers/models/prefix.py`
