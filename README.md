<h1 align='center'>Neural Signature Kernels</h1>
<h2 align='center'>Differentiable computations for the signature-PDE-kernel on CPU and GPU</h2>

This library provides differentiable computation in PyTorch for the [neural-signature-kernels](https://arxiv.org/abs/2303.17671) both on CPU and GPU.

This allows to build state-of-the-art kernel-methods such as Support Vector Machines or Gaussian Processes for high-dimensional, irregularly-sampled, multivariate time series.

---

## How to use the library

```python
import os
os.chdir("..")

import torch

from src.neuralSigKers import NeuralSigKer
from src.utils import ReLU_phi


# Hyperparameters
batch, timesteps, dim = 5, 20, 2
activation = lambda x : x * (x > 0) # ReLU
sigmas={"sigma_0": 1.0, "sigma_A": 1.0, "sigma_b": 2.0}

# Synthetic data
X = torch.rand((batch, timesteps, dim), dtype=torch.float64) # shape (batch, len_x, dim)
Y = torch.rand((batch, timesteps, dim), dtype=torch.float64) # shape (batch, len_y, dim)

# Initialize Kernel
neural_Kernel = NeuralSigKer(V_phi=ReLU_phi, 
                             sigmas=sigmas)

# Compute Gram Matrix
# G: (batch, batch, timesteps, timesteps)
# G[i, j, s, t] = K^{x_i, y_i}(s,t)
G = neural_Kernel.compute_Gram(X, Y, sym=False, max_batch=50)

# Compute Kernel batch-wise
# K: (batch, timesteps, timesteps)
# K[i, s, t] = K^{x_i,y_i}(s,t)
K = neural_Kernel.kernel(X, Y)
```

## Citation

```bibtex
@misc{cirone2023neural,
      title={Neural signature kernels as infinite-width-depth-limits of controlled ResNets}, 
      author={Nicola Muca Cirone and Maud Lemercier and Cristopher Salvi},
      year={2023},
      eprint={2303.17671},
      archivePrefix={arXiv},
      primaryClass={math.DS}
}
```

<!-- 
-->
