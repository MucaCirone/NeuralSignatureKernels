import torch
import numpy as np

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# =============================================================================================
# Nonlinear Activations
# =============================================================================================


def ReLU_phi(a, b, c):
    # a, b, c = K_{t,t}, K_{t,s}, K_{s,s}
    temp = b / torch.sqrt(a * c)
    temp = temp.to(device)
    temp = torch.minimum(temp, torch.tensor(1))  # Need the min due to numerical errors
    kappa = torch.maximum(temp, -torch.tensor(1))
    return (torch.sqrt(1 - kappa ** 2) + (torch.pi - torch.arccos(kappa) * kappa)) * torch.sqrt(a * c) / (2 * torch.pi)


def ReLU_dot_phi(a, b, c):
    # a, b, c = K_{t,t}, K_{t,s}, K_{s,s}
    temp = (b / torch.sqrt(a * c)).to(device)
    temp = torch.minimum(temp, torch.tensor(1))  # Need the min due to numerical errors
    kappa = torch.maximum(temp, -torch.tensor(1))
    return (torch.pi - torch.arccos(kappa)) / (2 * torch.pi)


def erf_phi(a, b, c):
    # a, b, c = K_{t,t}, K_{t,s}, K_{s,s}
    return (2 / np.pi) * np.arcsin(b / (np.sqrt((a + 0.5) * (c + 0.5))))


def exp_phi(a, b, c, sigma=4.):
    # a, b, c = K_{t,t}, K_{t,s}, K_{s,s}
    return np.exp((a + 2 * b + c) / (2 * sigma ** 2))


def exp_dot_phi(a, b, c, sigma=4.):
    # a, b, c = K_{t,t}, K_{t,s}, K_{s,s}
    return np.exp((a + 2 * b + c) / (2 * sigma ** 2))


def id_phi(a, b, c):
    # a, b, c = K_{t,t}, K_{t,s}, K_{s,s}
    return b


def id_dot_phi(a, b, c):
    # a, b, c = K_{t,t}, K_{t,s}, K_{s,s}
    return 0.0


# =============================================================================================
# Other Utils
# =============================================================================================


def augment_with_time(x: torch.Tensor,
                      grid: torch.Tensor = None) -> torch.Tensor:
    """
    Returns the time augmented (in dimension 0) paths i.e

        X_t --> (t,X_t)

    Parameters
        ----------
        x: (batch, timesteps, d)

        grid: (timesteps)
            if grid is None then the grid is supposed to be linspace(0, 1, (x.shape[1]))

    Returns
        -------
        x_augmented : (batch, timesteps, 1+d)
    """

    if grid is None:
        grid = torch.linspace(0, 1, (x.shape[1]))

    grid = grid.to(x.device.type)
    x_augmented = torch.cat((grid.expand(x.shape[0], -1).unsqueeze(-1), x),
                            dim=-1)

    return x_augmented


class Gram_Batcher:
    def __init__(self, max_batch: int = 50) -> None:
        """
        This class is a wrapper which allows to easily batch some function.

        Parameters
        ----------
        max_batch: int
            - The maximum batch number on which to use the function
        """
        self.max_batch = max_batch

    def forward(self,
                X: torch.Tensor, Y: torch.Tensor,
                sym: bool = False,
                **kwargs) -> None:
        """
        The function to batch.

        !!! TO BE IMPLEMENTED !!!

        Raises:
            NotImplementedError: _description_
        """
        raise NotImplementedError

    def forward_batched(self,
                        X: torch.Tensor, Y: torch.Tensor,
                        sym: bool = False,
                        **kwargs) -> torch.Tensor:
        """
        Computes F batch-wise.

        Parameters
        ------------
        X: torch.Tensor (batch_X, ...)
        Y: torch.Tensor (batch_Y,  ...)
        sym: bool
            - if True then X and Y are treated as being equal .
            - This is done WITHOUT CHECKS!!

        Returns
        -------
        Same output as F.
        """

        batch_X, batch_Y = X.shape[0], Y.shape[0]

        if batch_X <= self.max_batch and batch_Y <= self.max_batch:
            return self.forward(X, Y, sym, **kwargs)

        elif batch_X <= self.max_batch and batch_Y > self.max_batch:
            cutoff = int(batch_Y/2)
            Y1, Y2 = Y[:cutoff], Y[cutoff:]
            K1 = self.forward_batched(X, Y1, sym=False, **kwargs)
            K2 = self.forward_batched(X, Y2, sym=False, **kwargs)
            return torch.cat((K1, K2), dim=1)

        elif batch_X > self.max_batch and batch_Y <= self.max_batch:
            cutoff = int(batch_X/2)
            X1, X2 = X[:cutoff], X[cutoff:]
            K1 = self.forward_batched(X1, Y, sym=False, **kwargs)
            K2 = self.forward_batched(X2, Y, sym=False, **kwargs)
            return torch.cat((K1, K2), dim=0)

        cutoff_X, cutoff_Y = int(batch_X/2), int(batch_Y/2)
        X1, X2 = X[:cutoff_X], X[cutoff_X:]
        Y1, Y2 = Y[:cutoff_Y], Y[cutoff_Y:]

        K11 = self.forward_batched(X1, Y1, sym=sym, **kwargs)
        K22 = self.forward_batched(X2, Y2, sym=sym, **kwargs)

        K12 = self.forward_batched(X1, Y2, sym=False, **kwargs)
        # If X==Y then K21 is just the "transpose" of K12
        if sym:
            K21 = self.symmetric_treatment(K12)
        else:
            K21 = self.forward_batched(X2, Y1, sym=False, **kwargs)

        K_top = torch.cat((K11, K12), dim=1)
        K_bottom = torch.cat((K21, K22), dim=1)
        return torch.cat((K_top, K_bottom), dim=0)

    def symmetric_treatment(K12: torch.Tensor) -> torch.Tensor:
        """
        Specify how to deal with the symmetric case.

        Parameters
        ------------
        K12: torch.Tensor

        Returns:
        ------------
        torch.Tensor
        """

        K21 = K12.T
        return K21
