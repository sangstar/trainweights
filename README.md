# `trainweights`

Pet project meant to quantize, then serialize PyTorch models in a compact binary format. Ultimately 
meant to be useful for saving model checkpoints in a more compressed format to save heavily on 
disk space.

Still to do:
- General tidiness
- Python-side library to prepare model to call in to library
- `bf16` support
- Smarter quantization (e.g. not simply downcasting to `int8` but 
trying to preserve precision for certain things like biases )