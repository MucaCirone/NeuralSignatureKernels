import torch
import torchcde
import numpy as np
from collections.abc import Callable

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# =============================================================================================
# Samplers
# =============================================================================================

def rademacher_sample(M):
    # MxM Rademacher matrix
    # shape: (M, M)
    return torch.diag(2 * (torch.rand(M) > 0.5).int() - 1).float().to(device)


def matrix_D(M, d):
    # d MxM Rademacher matrices.
    # shape: (d, M, M)
    return torch.stack([rademacher_sample(M) for i in range(d)])


def randomAbeta(d: int, N: int, std_A=1.0, std_b=1.0):
    # Creates d random NxN matrix A and Nx1 vector beta
    # Entries have centered normal distribution

    A = torch.normal(0.0, std_A/np.sqrt(N), size=(d, N, N), requires_grad=True).to(device)
    b = torch.normal(0.0, std_b, size=(d, N, 1), requires_grad=True).to(device)
    Gamma = torch.cat([A, b], dim=-1)

    return [A, b, Gamma]


# =============================================================================================
# Create the random Neural CDE system
# =============================================================================================


class DrivingFields(torch.nn.Module):
    def __init__(self, input_channels, hidden_channels, activation=lambda x: x, std_A=1.0, std_b=1.0):
        ######################
        # input_channels is the number of input channels in the data X. (Determined by the data.)
        # hidden_channels is the number of channels for z_t. (Determined by you!)
        ######################
        super(DrivingFields, self).__init__()

        self.input_channels = input_channels
        self.hidden_channels = hidden_channels

        self.activation = activation

        self.std_A = std_A
        self.std_b = std_b

        _, _, self.Gamma = randomAbeta(input_channels, hidden_channels, std_A, std_b)

    def forward(self, t, z):
        batch = z.shape[0]  # z : (batch, hidden_channels)

        z = self.activation(z)  # z ->  \phi(z)
        z = torch.cat((z, torch.ones([batch, 1]).to(device)), 1)  # z -> [z,1]
        z = z.unsqueeze(-2).unsqueeze(-1)  # z : (batch, n+1) -> z : (batch, 1, n+1, 1)
        G = self.Gamma.expand(1, -1, -1, -1)  # Gamma : (d, n, n+1) ->  G : (1, d, n, n+1)
        res = torch.matmul(G, z)  # res : (batch, d, n, 1)
        res = res.squeeze(-1)  # res : (batch, d, n, 1) ->  res : (batch, d, n)
        res = torch.swapaxes(res, -2, -1)  # res : (batch, d, n) -> res : (batch, n, d)

        return res


# =============================================================================================
# rSig
# =============================================================================================

class rSig(torch.nn.Module):
    def __init__(self,
                 input_channels: int,
                 hidden_channels: int,
                 activation: Callable[[float], float] = lambda x: x,
                 sigmas: dict[str, float] = {"sigma_0": 1.0, "sigma_A": 1.0, "sigma_b": 0.0},
                 cubic: bool = False) -> None:
        """
        Randomized Signatures.
        Implementation of https://arxiv.org/pdf/2303.17671.pdf

        Parameters
        ----------
        input_channels: int
        hidden_channels: int
        activation: function float -> float
            - Must be streamable.
            - Defaults to the identity.
        sigmas: Dict[str, float]
            - Must contain the keys "sigma_0", "sigma_A", "sigma_b" with float values.
            - Defaults to {"sigma_0": 1.0, "sigma_A": 1.0, "sigma_b": 0.0}.
        cubic: bool
            - If True use cubic interpolation
            - If False use linear interpolation.
            - Defaults to False.
        """

        super(rSig, self).__init__()

        self.std_0 = sigmas["sigma_0"]
        self.std_A = sigmas["sigma_A"]
        self.std_b = sigmas["sigma_b"]

        self.input_channels = input_channels
        self.hidden_channels = hidden_channels

        self.cubic = cubic

        self.fields = DrivingFields(input_channels, hidden_channels, activation, self.std_A, self.std_b)
        self.z0 = torch.normal(0.0, self.std_0, size=(self.hidden_channels,)).to(device)

    def forward(self, x: torch.Tensor,
                interval: torch.Tensor = None,
                interval_return: torch.Tensor = None) -> torch.Tensor:
        """
        Computes the Batch Neural Randomized Signature.

        Parameters
        ----------
        x: torch.Tensor (batch, timesteps, input_channels)
        interval: torch.Tensor (timesteps)
            - Timestamps of observations.
            - If None consider it to be linspace(0, 1, timesteps)
            - Defaults to None.
        interval_return: torch.Tensor (timesteps)
            - Timestamps of rSig to return.
            - If None return only last value.
            - Defaults to None.

        Returns
        ----------
        z_T: torch.Tensor (batch, len(interval_return), hidden_channels)
        """

        flag = (interval_return is None)
        batch = x.shape[0]

        if interval is None:
            interval = torch.linspace(0, 1., x.shape[1])

        if self.cubic:
            coeffs = torchcde.natural_cubic_coeffs(x, interval).float()
            X = torchcde.CubicSpline(coeffs, interval)
        else:
            coeffs = torchcde.linear_interpolation_coeffs(x, interval).float()
            X = torchcde.LinearInterpolation(coeffs, interval)

        if flag:
            interval_return = X.interval

        z0 = (self.z0).expand(batch, -1)
        z_T = torchcde.cdeint(X=X, z0=z0,
                              func=self.fields,
                              t=interval_return)

        if flag:
            z_T = z_T[:, 1]

        return z_T


# =============================================================================================
# rSigKer
# =============================================================================================

class rSigKer(torch.nn.Module):
    """
    Discriminator object which uses the rSigKernel kernel on R^d to perform inference.

    Metric is the (unbiased) MMD.
    """
    def __init__(self,
                 hidden_dim=30,
                 MC_iters=1,
                 activation=lambda x: x,
                 sigmas={"sigma_0": 1.0, "sigma_A": 1.0, "sigma_b": 0.0}):

        super(rSigKer, self).__init__()

        self.sigmas = sigmas

        self.activation = activation
        self.MC_iters = MC_iters
        self.hidden_dim = hidden_dim

    def kernel(self, x: torch.float, y: torch.float, interval_return=None):
        """
        rSig kernel on R^d

        Parameters
        ----------
        x : torch.tensor of dim (timesteps_x, d)
        y : torch.tensor of dim (timesteps_y, d)

        Returns
        -------
                    k(x, y) where k(x, y) = neural - rSigKer
        """

        if not (x.dim() == 2) or not (y.dim() == 2):
            raise Exception("Maybe you should use compute_Gram_matrix(X,Y,...)!")

        X = x.clone().detach().unsqueeze(0)  # (1, timesteps_x, d)
        Y = y.clone().detach().unsqueeze(0)  # (1, timesteps_y, d)

        res = self.compute_Gram_matrix(X, Y, interval_return)

        if interval_return is None:
            return res[0, 0]

        return res[:, 0, 0]

    def compute_Gram_matrix(self, X: torch.float, Y: torch.float, interval_return=None):
        """
        Computes the Gram matrix G_ij = k(x_i, y_j), where k is the (sum of) Gaussian kernels
        associated to this instance of the discriminator object

        Parameters
        ----------
        X : torch.tensor of dim (batch_x, timesteps_x, d)
        Y : torch.tensor of dim (batch_y, timesteps_y, d)

        Returns
        -------
                    Gram matrix as specified above
        """

        if not (X.dim() == 3) or not (Y.dim() == 3):
            raise Exception("Either X or Y are not of type (batch, times, dim).")
        if not (X.shape[-1] == Y.shape[-1]):
            raise Exception("dim_X and dim_Y must be the same!")

        batch_x, batch_y, d = X.shape[0], Y.shape[0], X.shape[-1]

        X_equal_Y = (batch_x == batch_y and X.shape[1] == Y.shape[1])
        if X_equal_Y:
            X_equal_Y = (X == Y).all()

        t_x = torch.linspace(0, 1, X.shape[1])
        t_y = torch.linspace(0, 1, Y.shape[1])

        flag_none = (interval_return is None)
        if flag_none:
            interval_return = torch.tensor([0., 1.]).float().to(device)

        times = interval_return.shape[0]

        dotPs = torch.zeros((self.MC_iters, times, batch_x, batch_y), device=X.device)

        # If X == Y compute rSig only once
        if X_equal_Y:
            for iter in range(self.MC_iters):

                model = rSig(input_channels=d,
                             hidden_channels=self.hidden_dim,
                             activation=self.activation,
                             sigmas=self.sigmas)

                S_x = model.forward(X, interval=t_x, interval_return=interval_return)  # (batch_x, times, N)
                S_x = S_x.div(np.sqrt(self.hidden_dim))

                # Compute dotPs[iter][t, i, j] = S_x[i, t, :] \cdot S_x[j, t, :]
                dotPs[iter] = S_x.swapdims(0, 1) @ S_x.swapdims(0, 1).swapdims(1, 2)

        else:
            for iter in range(self.MC_iters):

                model = rSig(input_channels=d,
                             hidden_channels=self.hidden_dim,
                             activation=self.activation,
                             sigmas=self.sigmas)

                S_x = model.forward(X, interval=t_x, interval_return=interval_return)  # (batch_x, times, N)
                S_x = S_x.div(np.sqrt(self.hidden_dim))

                S_y = model.forward(Y, interval=t_y, interval_return=interval_return)  # (batch_y, times, N)
                S_y = S_y.div(np.sqrt(self.hidden_dim))

                # Compute dotPs[iter][t, i, j] = S_x[i, t, :] \cdot S_y[j, t, :]
                dotPs[iter] = S_x.swapdims(0, 1) @ S_y.swapdims(0, 1).swapaxes(1, 2)
                # dotPs[iter] = torch.einsum('itn, jtn -> tij', S_x, S_y)

        if flag_none:
            dotPs = dotPs[:, -1, :, :]

        res = dotPs.mean(dim=0)  # (times, batch_x, batch_y)

        return res

    def metric(self, X: torch.float, Y: torch.float, biased=False):
        """
        Calculates the empirical estimate to the squared Maximum Mean Discrepancy (MMD) between batched samples
        x \in R^d and y \in R^d.

        Parameters
        ----------
        X : torch.tensor of dim (batch_x, timesteps_x, d)
        Y : torch.tensor of dim (batch_y, timesteps_y, d)
        biased      Optional Boolean flag whether to return the biased estimate (diagonals included) or unbiased.

        Returns
        -------
                    Empirical estimate MMD_B(x, y)^2 or MMD_U(x, y)^2
        """

        batch_x = X.shape[0]
        batch_y = Y.shape[0]

        K_XX = self.compute_Gram_matrix(X, X)
        K_XY = self.compute_Gram_matrix(X, Y)
        K_YY = self.compute_Gram_matrix(Y, Y)

        K_XY_m = torch.mean(K_XY)

        if not biased:
            K_XX = K_XX.fill_diagonal_(0.)
            K_YY = K_YY.fill_diagonal_(0.)

            K_XX_m = torch.sum(K_XX)/(batch_x*(batch_x-1))
            K_YY_m = torch.sum(K_YY)/(batch_y*(batch_y-1))
        else:
            K_XX_m = torch.mean(K_XX)
            K_YY_m = torch.mean(K_YY)

        return K_XX_m - 2 * K_XY_m + K_YY_m
