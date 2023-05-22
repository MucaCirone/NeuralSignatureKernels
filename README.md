<h1 align='center'>Neural Signature Kernels</h1>
<h2 align='center'>Differentiable computations for the signature-PDE-kernel on CPU and GPU</h2>

This library provides differentiable computation in PyTorch for the [neural-signature-kernels](https://arxiv.org/abs/2303.17671) both on CPU and GPU.

This allows to build state-of-the-art kernel-methods such as Support Vector Machines or Gaussian Processes for high-dimensional, irregularly-sampled, multivariate time series.

---

## How to use the library

```python
import torch
import sigkernel

# Specify the static kernel (for linear kernel use sigkernel.LinearKernel())
static_kernel = sigkernel.RBFKernel(sigma=0.5)

# Specify dyadic order for PDE solver (int > 0, default 0, the higher the more accurate but slower)
dyadic_order = 1

# Specify maximum batch size of computation; if memory is a concern try reducing max_batch, default=100
max_batch = 100

# Initialize the corresponding signature kernel
signature_kernel = sigkernel.SigKernel(static_kernel, dyadic_order)

# Synthetic data
batch, len_x, len_y, dim = 5, 10, 20, 2
X = torch.rand((batch,len_x,dim), dtype=torch.float64, device='cuda') # shape (batch,len_x,dim)
Y = torch.rand((batch,len_y,dim), dtype=torch.float64, device='cuda') # shape (batch,len_y,dim)
Z = torch.rand((batch,len_y,dim), dtype=torch.float64, device='cuda') # shape (batch,len_y,dim)

# Compute signature kernel "batch-wise" (i.e. k(x_1,y_1),...,k(x_batch, y_batch))
K = signature_kernel.compute_kernel(X,Y,max_batch)

# Compute signature kernel Gram matrix (i.e. k(x_i,y_j) for i,j=1,...,batch), also works for different batch_x != batch_y)
G = signature_kernel.compute_Gram(X,Y,sym=False,max_batch)

# Compute MMD distance between samples x ~ X and samples y ~ Y, where X,Y are two distributions on path space...
mmd = signature_kernel.compute_mmd(X,Y,max_batch)
# ... and to backpropagate through the MMD distance simply call .backward(), like any other PyTorch loss function
mmd.backward()

# Compute scoring rule between X and a sample path y, i.e. S_sig(X,y) = E[k(X,X)] - 2E[k(X,y] ...
y = Y[0]
sr = signature_kernel.compute_scoring_rule(X,y,max_batch)

# ... and expected scoring rule between X and Y, i.e. S(X,Y) = E_Y[S_sig(X,y)]
esr = signature_kernel.compute_expected_scoring_rule(X,Y,max_batch)

# Sig CHSIC: XY|Z
sigchsic = signature_kernel.SigCHSIC(X, Y, Z, static_kernel, dyadic_order=1, eps=0.1)
```
