import torch
from src.utils import id_phi
from collections.abc import Callable

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# =============================================================================================
# Kernels
# =============================================================================================


class NeuralSigKer(torch.nn.Module):
    def __init__(self,
                 V_phi: Callable[[tuple[float, float, float], float]] = id_phi,
                 sigmas: dict[str, float] = {"sigma_0": 1.0, "sigma_A": 1.0, "sigma_b": 0.0}):
        '''
        Neural Signature Kernel
        Implementation of https://arxiv.org/pdf/2303.17671.pdf

        Parameters
        ----------
        V_phi : function tuple[float, float, float] -> float
            - (a,b,c) -> V_phi(\Sigma) with \Sigma := [[a,b],[b,c]]
            - Must be strameable.
            - Defaults to (a,b,c) -> b
        sigmas: Dict[str, float]
            - Must contain the keys "sigma_0", "sigma_A", "sigma_b" with float values.
            - Defaults to {"sigma_0": 1.0, "sigma_A": 1.0, "sigma_b": 0.0}.
        '''

        super(NeuralSigKer, self).__init__()

        self.V_phi = V_phi

        self.sigmas = sigmas

        self.std_0 = sigmas["sigma_0"]
        self.std_A = sigmas["sigma_A"]
        self.std_b = sigmas["sigma_b"]

    def _kernel_same(self,
                     X: torch.Tensor) -> torch.Tensor:
        """
        Computes Kxx.

        Parameters
        ----------
        X: torch.Tensor (batch_x, timesteps_x, d)

        Returns
        -------
        Kxx: torch.Tensor (batch_x, timesteps_x, timesteps_x)
        """

        batch_x, timesteps_x = X.shape[0], X.shape[1]
        var_0, var_A, var_b = self.std_0**2, self.std_A**2, self.std_b**2

        # dX2: (batch_x, timesteps_x-1, timesteps_x-1)
        # dX2[i, s, t] = < dx_i(s), dx_i(t) > = \sum_{k=1}^d dX[i, s, *, k] * dX[i, *, t, k]
        dX2 = (torch.diff(X, dim=1).unsqueeze(2)*torch.diff(X, dim=1).unsqueeze(1)).sum(dim=-1)

        # Initialize Kernel
        # K: (batch_x, batch_y, timesteps_x, timesteps_y)
        Kxx = torch.zeros((batch_x, timesteps_x, timesteps_x)).to(X.device.type)
        Kxx[..., 0, :] = var_0
        Kxx[..., :, 0] = var_0

        def compute_next(s, t):
            past = Kxx[..., s+1, t] + Kxx[..., s, t+1] - Kxx[..., s, t]
            innovation = (var_A * self.V_phi(Kxx[..., s, s], Kxx[..., s, t], Kxx[..., t, t]) + var_b) * dX2[..., s, t]
            return past + innovation

        for s in range(timesteps_x-1):
            # Compute K[:, :, s+1, t+1] and K[:, :, t+1, s+1] when t < s
            for t in range(s):
                Kxx[..., s+1, t+1] = compute_next(s, t)
                Kxx[..., t+1, s+1] = compute_next(t, s)
            # Compute K[:, :, s+1, t+1] when t==s
            Kxx[..., s+1, s+1] = compute_next(s, s)

        return Kxx

    def _kernel(self,
                X: torch.Tensor, Y: torch.Tensor,
                Kxx: torch.Tensor,
                Kyy: torch.Tensor) -> torch.Tensor:
        """
        Computes the Neural Signature Kernel on R^d paths.
        Recall

            K^{x,y}(s+1,t+1) = K^{x,y}(s+1,t) + K^{x,y}(s,t+1) - K^{x,y}(s,t)
                                + (sigma_A^2*V_phi(K^{x,y}(s,s), K^{x,y}(s,t), K^{x,y}(t,t)) + sigma_b^2) <dx_s, dy_t>

        !! Make sure elements in X and Y all sample on the same grid !!

        Parameters
        ----------
        X: torch.Tensor (batch_x, timesteps_x, d)
        Y: torch.Tensor (batch_y, timesteps_y, d)
        Kxx: torch.Tensor (batch_x, timesteps_x, timesteps_x, d)
        Kyy: torch.Tensor (batch_y, timesteps_y, timesteps_y, d)

        Returns
        -------
        K: torch.Tensor (batch_x, batch_y, timesteps_x, timesteps_y)
        """

        batch_x, batch_y = X.shape[0], Y.shape[0]
        timesteps_x, timesteps_y = X.shape[1], Y.shape[1]

        Kxx_expand = Kxx.unsqueeze(1).expand(-1, batch_y, -1, -1)
        Kyy_expand = Kyy.unsqueeze(0).expand(batch_x, -1, -1, -1)

        var_0, var_A, var_b = self.std_0**2, self.std_A**2, self.std_b**2

        # dXdY: (batch_x, batch_y, timesteps_x-1, timesteps_y-1)
        # dXdY[i, j, s, t] = < dx_i(s), dy_j(t) > = \sum_{k=1}^d dX[i, *, s, *, k] * dY[*, j, *, t]
        dXdY = (torch.diff(X, dim=1).unsqueeze(1).unsqueeze(3)*torch.diff(Y, dim=1).unsqueeze(0).unsqueeze(2)).sum(dim=-1)

        # Initialize Kernel
        # K: (batch_x, batch_y, timesteps_x, timesteps_y)
        K = torch.zeros((batch_x, batch_y, timesteps_x, timesteps_y)).to(X.device.type)
        K[..., 0, :] = var_0
        K[..., :, 0] = var_0

        # Helper Function
        def compute_next(s, t):
            past = K[..., s+1, t] + K[..., s, t+1] - K[..., s, t]
            # innovation: (batch_x, batch_y)
            innovation = (var_A * self.V_phi(Kxx_expand[..., s, s], K[..., s, t], Kyy_expand[..., t, t]) + var_b) * dXdY[..., s, t]
            return past + innovation

        for s in range(timesteps_x-1):
            # Compute K[:, :, s+1, t+1] and K[:, :, t+1, s+1] when t < s
            for t in range(s):
                K[..., s+1, t+1] = compute_next(s, t)
                K[..., t+1, s+1] = compute_next(t, s)
            # Compute K[:, :, s+1, t+1] when t==s
            K[..., s+1, s+1] = compute_next(s, s)

        return K

    def kernel(self,
               X: torch.Tensor, Y: torch.Tensor,
               Kxx: torch.Tensor = None,
               Kyy: torch.Tensor = None,
               sym: bool = False,
               max_batch: int = 50) -> torch.Tensor:
        """
        Computes the batched Neural Signature Kernel on R^d paths.
        !! Make sure elements in X and Y all sample on the same grid !!

        Parameters
        ----------
        X: torch.Tensor (batch_x, timesteps_x, d)
        Y: torch.Tensor (batch_y, timesteps_y, d)
        Kxx: torch.Tensor (batch_x, timesteps_x, timesteps_x, d)
            - if None compute from scratch, else use the given data.
            - !! Must have all times !!
        Kyy: torch.Tensor(batch_y, timesteps_y, timesteps_y, d)
            - if None compute from scratch, else use the given data.
            - !! Must have all times !!
        max_batch: int
            - the maximum batch size for the computation.
        sym: bool
            - if True then X and Y are treated as being equal .
            - This is done WITHOUT CHECKS!!
        all_times: bool
            - if True then return the kernel for all time pairs (s,t).
            - if False only return the last value.

        Returns
        -------
        K: torch.Tensor
            - (batch_x, batch_y, timesteps_x, timesteps_y) if all_times
            - (batch_x, batch_y) if not all_times
            - K[i, j, s, t] = K_{phi}^{x_i,y_j}(s,t)
        """

        self._compatibility_checks(X, Y)

        # Check if Kxx and Kyy have already been supplied, if not compute them
        if Kxx is None:
            Kxx = self._kernel_same(X)
            # print('Kxx computed!')
        if Kyy is None:
            Kyy = self._kernel_same(Y)
            # print('Kyy computed!')

        # Compute the Gram matrix
        batch_X, batch_Y = X.shape[0], Y.shape[0]

        if batch_X <= max_batch and batch_Y <= max_batch:
            return self._kernel(X, Y, Kxx, Kyy)

        elif batch_X <= max_batch and batch_Y > max_batch:
            cutoff = int(batch_Y/2)
            Y1, Y2 = Y[:cutoff], Y[cutoff:]
            Kyy1, Kyy2 = Kyy[:cutoff], Kyy[cutoff:]
            K1 = self.kernel(X, Y1, Kxx, Kyy1, sym=False, max_batch=max_batch)
            K2 = self.kernel(X, Y2, Kxx, Kyy2, sym=False, max_batch=max_batch)
            return torch.cat((K1, K2), dim=1)

        elif batch_X > max_batch and batch_Y <= max_batch:
            cutoff = int(batch_X/2)
            X1, X2 = X[:cutoff], X[cutoff:]
            Kxx1, Kxx2 = Kxx[:cutoff], Kxx[cutoff:]
            K1 = self.kernel(X1, Y, Kxx1, Kyy, sym=False, max_batch=max_batch)
            K2 = self.kernel(X2, Y, Kxx2, Kyy, sym=False, max_batch=max_batch)
            return torch.cat((K1, K2), dim=0)

        cutoff_X, cutoff_Y = int(batch_X/2), int(batch_Y/2)
        X1, X2 = X[:cutoff_X], X[cutoff_X:]
        Kxx1, Kxx2 = Kxx[:cutoff_X], Kxx[cutoff_X:]
        Y1, Y2 = Y[:cutoff_Y], Y[cutoff_Y:]
        Kyy1, Kyy2 = Kyy[:cutoff_Y], Kyy[cutoff_Y:]

        K11 = self.kernel(X1, Y1, Kxx1, Kyy1, sym=sym, max_batch=max_batch)

        K12 = self.kernel(X1, Y2, Kxx1, Kyy2, sym=False, max_batch=max_batch)
        if sym:
            K21 = K12.swapaxes(0, 1).swapaxes(-2, -1)
        else:
            K21 = self.kernel(X2, Y1, Kxx2, Kyy1, sym=False, max_batch=max_batch)

        K22 = self.kernel(X2, Y2, Kxx2, Kyy2, sym=sym, max_batch=max_batch)

        K_top = torch.cat((K11, K12), dim=1)
        K_bottom = torch.cat((K21, K22), dim=1)
        return torch.cat((K_top, K_bottom), dim=0)

    def _compatibility_checks(self, X: torch.Tensor, Y: torch.Tensor) -> None:
        """
        Makes some needed compatibility checks.

        Parameters
        ----------
        X: torch.Tensor (batch_x, timesteps_x, d)
        Y: torch.Tensor (batch_y, timesteps_y, d)
        """

        if not (X.dim() == 3) or not (Y.dim() == 3):
            raise Exception("Either X or Y are not of type (batch, times, dim).")

        if not (X.shape[-1] == Y.shape[-1]):
            raise Exception("dim_X and dim_Y must be the same!")

        if not (X.shape[1] == Y.shape[1]):
            raise NotImplementedError("Different Grid case not yet implemented. \n"
                                      "Use torchcde package to interpolate the data")
