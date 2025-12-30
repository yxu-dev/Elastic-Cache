# Elastic-Cache
[![Project](https://img.shields.io/static/v1?label=Project&message=Github&color=blue&logo=github-pages)](https://vila-lab.github.io/elastic-cache-webpage/)
[![arXiv](https://img.shields.io/badge/Paper-arXiv-red.svg)](https://arxiv.org/abs/2510.14973)

Elastic-Cache is a training-free framework designed to accelerate diffusion language models through efficient KV caching. It features:

- Fast and accurate KV caching for diffusion LLMs, achieving up to 45× speedup over non-accelerated baselines with only a minor drop in accuracy.

- Layer-aware and time-aware KV caching, enabling the model to determine where and when caching is most effective.

- An automatic caching mechanism that analyzes attention drift, removing the need for predefined cache schedules used in prior work.

## Why Elastic-Cache?

- Ready-to-use, training-free component for accelerating diffusion LLMs.

- Provides a controllable trade-off between accuracy and latency.

- Architecture-agnostic, supporting various open-source diffusion LLMs, including LLaDA, Dream, and LLaDA-V.

- Scalable to long sequences, maintaining efficiency as input length grows.

## Next Steps

[✅] Serve diffusion LLMs with Elastic-Cache and batch inference

[✅] Triton implementation

[🚀] Integrate into additional models (e.g., MMaDA)

[🚀] Elastic-Cache v2

## Project Structure

```
.
├── dream/          # Dream model related code
├── llada/          # LLaDA model related code
└── .gitignore      # Git ignore configuration
```


## Installation

1. Clone the repository:
```bash
git clone https://github.com/VILA-Lab/elastic-cache.git
cd elastic-cache
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

Parameter descriptions:
- `--gen_length`: Maximum length of generated text.
- `--window_size`: Sliding window length, less than or equal to gen_length. If less than gen_length, it means using semi_autoregressive remasking.
- `--threshold`: Confidence-aware decoding threshold.
- `--gamma`: Cache update trigger threshold.
- `--track_num`: number of most-attended tokens used for cache update trigger.
- `--block_caching`: block caching far-away [MASK] tokens.


### 1. Using LLaDA Model

```bash
cd llada
bash eval_{task}.sh
```

### 2. Using Dream Model

```bash
cd dream
bash eval_{task}.sh
```

## Acknowledgements

This repository is built upon [LLaDA](https://github.com/ML-GSAI/LLaDA), [Dream](https://github.com/HKUNLP/Dream), [LLaDA-V](https://github.com/ML-GSAI/LLaDA-V), and the [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness).


## License

This project is licensed under the Apache License 2.0. See the [LICENSE](LICENSE) file for details. 


## Citation

```bibtex
@article{nguyen2025attention,
  title={Attention is all you need for kv cache in diffusion llms},
  author={Nguyen-Tri, Quan and Ranjan, Mukul and Shen, Zhiqiang},
  journal={arXiv preprint arXiv:2510.14973},
  year={2025}
}
```