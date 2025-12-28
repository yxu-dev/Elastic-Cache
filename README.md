# Elastic-Cache
[![Project](https://img.shields.io/static/v1?label=Project&message=Github&color=blue&logo=github-pages)](https://vila-lab.github.io/elastic-cache-webpage/)
[![arXiv](https://img.shields.io/badge/Paper-arXiv-red.svg)](https://arxiv.org/abs/2510.14973)



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

## License

This project is licensed under the Apache License 2.0. See the [LICENSE](LICENSE) file for details. 



