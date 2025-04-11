# CAT
[CAT: Circular-Convolutional Attention for Sub-Quadratic Transformers](https://arxiv.org/abs/2504.06704)

# Installation
```
pip install git@github.com:KotaShimomura/ccatten.git
```
or
```
git clone git@github.com:KotaShimomura/ccatten.git

cd ccatten

pip install .
```

# Usage
```python

import torch
import ccatten as cat

batch_size, seq_len, dim = 2, 256, 768
x = torch.randn(batch_size, seq_len, dim)

# basic
cat = CircularAttention(dim=dim, num_heads=8)
output = cat(x)

# cross-attem
tmp = torch.randn(batch_size, seq_len*2, dim)
avg_key = AveragedKeyAttention(dim=dim, num_heads=8)
output = avg_key(x, tmp)

```

# Acknowledgement
```
@misc{yamada2025cat,
  author       = {Yoshihiro Yamada},
  title        = {CAT: Circular-Convolutional Attention for Sub-Quadratic Transformers},
  year         = {2025},
  eprint       = {2504.06704},
  archivePrefix= {arXiv},
  primaryClass = {cs.LG},
  url          = {https://arxiv.org/abs/2504.06704},
  doi          = {10.48550/arXiv.2504.06704}
}
```