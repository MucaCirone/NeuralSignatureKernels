import torch
from src.utils import id_phi, id_dot_phi
from collections.abc import Callable

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# =============================================================================================
# Base Kernels
# =============================================================================================


class NeuralSigKer(torch.nn.Module):
    """
        # Neural Signature Kernel
        Implementation of https://arxiv.org/pdf/2303.17671.pdf

        d^2 K_phi^{x,y}(s,t) =  [\sigma_A^2 V_phi(Sigma_phi^{x,y}(s,t)) + \sigma_b^2] <dx_s, dy_t>
    """

    def __init__(self,
                 V_phi: Callable[[tuple[float, float, float], float]] = id_phi,
                 sigmas: dict[str, float] = {"sigma_0": 1.0, "sigma_A": 1.0, "sigma_b": 0.0}) -> None:
        """
        Parameters
        ----------
        V_phi : function tuple[float, float, float] -> float
            - (a,b,c) -> V_phi(\Sigma) with \Sigma := [[a,b],[b,c]]
            - Must be streamable.
            - Defaults to (a,b,c) -> b
        sigmas: Dict[str, float]
            - Must contain the keys "sigma_0", "sigma_A", "sigma_b" with float values.
            - Defaults to {"sigma_0": 1.0, "sigma_A": 1.0, "sigma_b": 0.0}.
        """

        super(NeuralSigKer, self).__init__()

        self.V_phi = V_phi

        self.sigmas = sigmas

        self.std_0 = sigmas["sigma_0"]
        self.std_A = sigmas["sigma_A"]
        self.std_b = sigmas["sigma_b"]

    def kernel(self,
               X: torch.Tensor, Y: torch.Tensor,
               Kxx: torch.Tensor = None,
               Kyy: torch.Tensor = None,
               sym: bool = False) -> torch.Tensor:
        """
        Computes the batched Neural Signature Kernel on R^d paths.
        !! Make sure elements in X and Y all sample on the same grid !!
        !! Make batches of X and Y are equal !!

        Parameters
        ----------
        X: torch.Tensor (batch, timesteps_x, d)
        Y: torch.Tensor (batch, timesteps_y, d)
        Kxx: torch.Tensor (batch, timesteps_x, timesteps_x)
            - The batched NSK on X: Kxx[i,s,t] = K_phi^{x_i,x_i}(s,t)
            - if None then compute from scratch, else use the given data.
            - !! Must have all times !!
        Kyy: torch.Tensor(batch, timesteps_y, timesteps_y)
            - The batched NSK on Y: Kyy[j,s,t] = K_phi^{y_j,y_j}(s,t)
            - if None then compute from scratch, else use the given data.
            - !! Must have all times !!
        sym: bool
            - if True then X and Y are treated as being equal .
            - This is done WITHOUT CHECKS!!

        Returns
        -------
        K: torch.Tensor (batch, timesteps_x, timesteps_y)
            - K[i, s, t] = K_{phi}^{x_i,y_i}(s,t)
        """

        # Some preliminary checks
        if not (X.shape == Y.shape):
            raise Exception("X and Y must have the same shape!")

        if sym and not (Kxx is None):
            return Kxx
        if sym and not (Kyy is None):
            return Kyy

        if Kxx is None:
            Kxx = self._kernel_same(X)
            if sym:
                return Kxx
            # print('Kxx computed!')
        if Kyy is None:
            Kyy = self._kernel_same(Y)
            # print('Kyy computed!')

        batch, timesteps = X.shape[0], X.shape[1]
        var_0, var_A, var_b = self.std_0**2, self.std_A**2, self.std_b**2

        # dXdY: (batch, timesteps-1, timesteps-1)
        # dXdY[i, s, t] = < dx_i(s), dy_i(t) > = \sum_{k=1}^d dX[i, s, *, k] * dY[i, *, t, k]
        dXdY = (torch.diff(X, dim=1).unsqueeze(2)*torch.diff(Y, dim=1).unsqueeze(1)).sum(dim=-1)

        # Initialize Kernel
        # K: (batch, timesteps, timesteps)
        Kxy = torch.zeros((batch, timesteps, timesteps)).to(X.device.type)
        Kxy[..., 0, :] = var_0
        Kxy[..., :, 0] = var_0

        def compute_next(s, t):
            past = Kxy[..., s+1, t] + Kxy[..., s, t+1] - Kxy[..., s, t]
            innovation = (var_A * self.V_phi(Kxx[..., s, s], Kxy[..., s, t], Kyy[..., t, t]) + var_b) * dXdY[..., s, t]
            return past + innovation

        for s in range(timesteps-1):
            # Compute K[:, :, s+1, t+1] and K[:, :, t+1, s+1] when t < s
            for t in range(s):
                Kxy[..., s+1, t+1] = compute_next(s, t)
                Kxy[..., t+1, s+1] = compute_next(t, s)
            # Compute K[:, :, s+1, t+1] when t==s
            Kxy[..., s+1, s+1] = compute_next(s, s)

        return Kxy

    def compute_Gram(self,
                     X: torch.Tensor, Y: torch.Tensor,
                     Kxx: torch.Tensor = None,
                     Kyy: torch.Tensor = None,
                     sym: bool = False,
                     max_batch: int = 50) -> torch.Tensor:
        """
        Computes the batched Neural Signature Kernel Gram matrix on R^d paths.
        !! Make sure elements in X and Y all sample on the same grid !!

        Parameters
        ----------
        X: torch.Tensor (batch_x, timesteps_x, d)
        Y: torch.Tensor (batch_y, timesteps_y, d)
        Kxx: torch.Tensor (batch_x, timesteps_x, timesteps_x)
            - if None compute from scratch, else use the given data.
            - !! Must have all times !!
        Kyy: torch.Tensor(batch_y, timesteps_y, timesteps_y)
            - if None compute from scratch, else use the given data.
            - !! Must have all times !!
        max_batch: int
            - the maximum batch size for the computation.
        sym: bool
            - if True then X and Y are treated as being equal .
            - This is done WITHOUT CHECKS!!

        Returns
        -------
        K: torch.Tensor (batch_x, batch_y, timesteps_x, timesteps_y)
            - K[i, j, s, t] = K_{phi}^{x_i,y_j}(s,t)
        """

        self._compatibility_checks(X, Y)

        # Check if Kxx and Kyy have already been supplied, if not compute them
        if Kxx is None:
            Kxx = self._kernel_same(X)
        if Kyy is None:
            if sym:
                Kyy = Kxx
            else:
                Kyy = self._kernel_same(Y)

        # Compute the Gram matrix
        batch_X, batch_Y = X.shape[0], Y.shape[0]

        if batch_X <= max_batch and batch_Y <= max_batch:
            return self._Gram(X, Y, Kxx, Kyy)

        elif batch_X <= max_batch and batch_Y > max_batch:
            cutoff = int(batch_Y/2)
            Y1, Y2 = Y[:cutoff], Y[cutoff:]
            Kyy1, Kyy2 = Kyy[:cutoff], Kyy[cutoff:]
            K1 = self.compute_Gram(X, Y1, Kxx, Kyy1, sym=False, max_batch=max_batch)
            K2 = self.compute_Gram(X, Y2, Kxx, Kyy2, sym=False, max_batch=max_batch)
            return torch.cat((K1, K2), dim=1)

        elif batch_X > max_batch and batch_Y <= max_batch:
            cutoff = int(batch_X/2)
            X1, X2 = X[:cutoff], X[cutoff:]
            Kxx1, Kxx2 = Kxx[:cutoff], Kxx[cutoff:]
            K1 = self.compute_Gram(X1, Y, Kxx1, Kyy, sym=False, max_batch=max_batch)
            K2 = self.compute_Gram(X2, Y, Kxx2, Kyy, sym=False, max_batch=max_batch)
            return torch.cat((K1, K2), dim=0)

        cutoff_X, cutoff_Y = int(batch_X/2), int(batch_Y/2)
        X1, X2 = X[:cutoff_X], X[cutoff_X:]
        Kxx1, Kxx2 = Kxx[:cutoff_X], Kxx[cutoff_X:]
        Y1, Y2 = Y[:cutoff_Y], Y[cutoff_Y:]
        Kyy1, Kyy2 = Kyy[:cutoff_Y], Kyy[cutoff_Y:]

        K11 = self.compute_Gram(X1, Y1, Kxx1, Kyy1, sym=sym, max_batch=max_batch)

        K12 = self.compute_Gram(X1, Y2, Kxx1, Kyy2, sym=False, max_batch=max_batch)
        if sym:
            K21 = K12.swapaxes(0, 1).swapaxes(-2, -1)
        else:
            K21 = self.compute_Gram(X2, Y1, Kxx2, Kyy1, sym=False, max_batch=max_batch)

        K22 = self.compute_Gram(X2, Y2, Kxx2, Kyy2, sym=sym, max_batch=max_batch)

        K_top = torch.cat((K11, K12), dim=1)
        K_bottom = torch.cat((K21, K22), dim=1)
        return torch.cat((K_top, K_bottom), dim=0)

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

    def _Gram(self,
              X: torch.Tensor, Y: torch.Tensor,
              Kxx: torch.Tensor,
              Kyy: torch.Tensor) -> torch.Tensor:
        """
        Computes the Neural Signature Kernel Gram matrix.
        Recall

            K^{x,y}(s+1,t+1) = K^{x,y}(s+1,t) + K^{x,y}(s,t+1) - K^{x,y}(s,t)
                                + (sigma_A^2*V_phi(K^{x,x}(s,s), K^{x,y}(s,t), K^{y,y}(t,t)) + sigma_b^2) <dx_s, dy_t>

        !! Make sure elements in X and Y all sample on the same grid !!

        Parameters
        ----------
        X: torch.Tensor (batch_x, timesteps_x, d)
        Y: torch.Tensor (batch_y, timesteps_y, d)
        Kxx: torch.Tensor (batch_x, timesteps_x, timesteps_x)
        Kyy: torch.Tensor (batch_y, timesteps_y, timesteps_y)

        Returns
        -------
        K: torch.Tensor (batch_x, batch_y, timesteps_x, timesteps_y)
        """

        batch_x, batch_y = X.shape[0], Y.shape[0]
        timesteps_x, timesteps_y = X.shape[1], Y.shape[1]

        var_0, var_A, var_b = self.std_0**2, self.std_A**2, self.std_b**2

        # dXdY: (batch_x, batch_y, timesteps_x-1, timesteps_y-1)
        # dXdY[i, j, s, t] = < dx_i(s), dy_j(t) > = \sum_{k=1}^d dX[i, *, s, *, k] * dY[*, j, *, t, k]
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
            innovation = (var_A * self.V_phi(Kxx.unsqueeze(1)[..., s, s], K[..., s, t], Kyy.unsqueeze(0)[..., t, t]) + var_b) * dXdY[..., s, t]
            return past + innovation

        for s in range(timesteps_x-1):
            # Compute K[:, :, s+1, t+1] and K[:, :, t+1, s+1] when t < s
            for t in range(s):
                K[..., s+1, t+1] = compute_next(s, t)
                K[..., t+1, s+1] = compute_next(t, s)
            # Compute K[:, :, s+1, t+1] when t==s
            K[..., s+1, s+1] = compute_next(s, s)

        return K

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
            raise Exception("d_X and d_Y must be the same!")

        if not (X.shape[1] == Y.shape[1]):
            raise NotImplementedError("Different Grid case not yet implemented. \n"
                                      "Use torchcde package to interpolate the data")


class inhomNeuralSigKer(torch.nn.Module):
    """
        # Time InHomogeneous Neural Signature Kernel
        Implementation of https://arxiv.org/pdf/2303.17671.pdf

         d\kappa_phi^{x,y}(t) =  [\sigma_A^2 V_phi(Sigma_phi^{x,y}(t)) + \sigma_b^2] <\dot x_t, \dot y_t> dt
    """

    def __init__(self,
                 V_phi: Callable[[tuple[float, float, float], float]] = id_phi,
                 sigmas: dict[str, float] = {"sigma_0": 1.0, "sigma_A": 1.0, "sigma_b": 0.0}) -> None:
        """
        Parameters
        ----------
        V_phi : function tuple[float, float, float] -> float
            - (a,b,c) -> V_phi(\Sigma) with \Sigma := [[a,b],[b,c]]
            - Must be strameable.
            - Defaults to (a,b,c) -> b
        sigmas: Dict[str, float]
            - Must contain the keys "sigma_0", "sigma_A", "sigma_b" with float values.
            - Defaults to {"sigma_0": 1.0, "sigma_A": 1.0, "sigma_b": 0.0}.
        """

        super(inhomNeuralSigKer, self).__init__()

        self.V_phi = V_phi

        self.sigmas = sigmas

        self.std_0 = sigmas["sigma_0"]
        self.std_A = sigmas["sigma_A"]
        self.std_b = sigmas["sigma_b"]

    def kernel(self,
               X: torch.Tensor, Y: torch.Tensor,
               Kxx: torch.Tensor = None,
               Kyy: torch.Tensor = None,
               interval: torch.Tensor = None,
               sym: bool = False) -> torch.Tensor:
        """
        Computes the batched Inhomogeneous Neural Signature Kernel on R^d paths.
        !! Make sure elements in X and Y all sample on the same grid !!
        !! Make batches of X and Y are equal !!

        Parameters
        ----------
        X: torch.Tensor (batch, timesteps, d)
        Y: torch.Tensor (batch, timesteps, d)
        Kxx: torch.Tensor (batch, timesteps)
            - if None compute from scratch, else use the given data.
            - !! Must have all times !!
        Kyy: torch.Tensor(batch, timesteps)
            - if None compute from scratch, else use the given data.
            - !! Must have all times !!
        interval: torch.Tensor (timesteps)
            - The interval over which elements of X and Y are sampled
            - If None then interval = torch.linspace(0, 1, timesteps_x)
            - Defaults to None
        sym: bool
            - if True then X and Y are treated as being equal .
            - This is done WITHOUT CHECKS!!

        Returns
        -------
        K: torch.Tensor (batch, timesteps)
            - K[i, t] = \kappa_{phi}^{x_i,y_i}(t)
        """

        # Some preliminary checks
        if not (X.shape == Y.shape):
            raise Exception("X and Y must have the same shape!")

        if sym and not (Kxx is None):
            return Kxx
        if sym and not (Kyy is None):
            return Kyy

        if Kxx is None:
            Kxx = self._kernel_same(X)
            if sym:
                return Kxx
        if Kyy is None:
            Kyy = self._kernel_same(Y)

        batch, timesteps = X.shape[0], X.shape[1]
        var_0, var_A, var_b = self.std_0**2, self.std_A**2, self.std_b**2

        if interval is None:
            interval = torch.linspace(0, 1, timesteps)
        dt = interval.diff(dim=0)

        # dXdY: (batch, timesteps-1)
        # dXdY[i, t] = < dx_i(t), dy_i(t) > = \sum_{k=1}^d dX[i, t, k] * dY[i, t, k]
        dXdY = (torch.diff(X, dim=1)*torch.diff(Y, dim=1)).sum(dim=-1)

        # Initialize Kernel
        # K: (batch, timesteps)
        Kxy = torch.zeros((batch, timesteps)).to(X.device.type)
        Kxy[..., 0] = var_0

        def compute_next(t):
            past = Kxy[..., t]
            innovation = (var_A * self.V_phi(Kxx[..., t], Kxy[..., t], Kyy[..., t]) + var_b) * dXdY[..., t]/dt[t]
            return past + innovation

        for t in range(timesteps-1):
            # Compute Kxy[:, t+1]
            Kxy[..., t+1] = compute_next(t)

        return Kxy

    def compute_Gram(self,
                     X: torch.Tensor, Y: torch.Tensor,
                     Kxx: torch.Tensor = None,
                     Kyy: torch.Tensor = None,
                     interval: torch.Tensor = None,
                     sym: bool = False,
                     max_batch: int = 50) -> torch.Tensor:
        """
        Computes the batched Inhomogeneous Neural Signature Kernel Gram matrix on R^d paths.
        !! Make sure elements in X and Y all sample on the same grid !!

        Parameters
        ----------
        X: torch.Tensor (batch_x, timesteps, d)
        Y: torch.Tensor (batch_y, timesteps, d)
        Kxx: torch.Tensor (batch_x, timesteps)
            - if None compute from scratch, else use the given data.
            - !! Must have all times !!
        Kyy: torch.Tensor(batch_y, timesteps)
            - if None compute from scratch, else use the given data.
            - !! Must have all times !!
        interval: torch.Tensor (timesteps)
            - The interval over which elements of X and Y are sampled
            - If None then interval = torch.linspace(0, 1, timesteps_x)
            - Defaults to None
        max_batch: int
            - the maximum batch size for the computation.
        sym: bool
            - if True then X and Y are treated as being equal.
            - This is done WITHOUT CHECKS!!

        Returns
        -------
        K: torch.Tensor (batch_x, batch_y, timesteps)
            - K[i, j, t] = K_{phi}^{x_i,y_j}(t)
        """

        self._compatibility_checks(X, Y)

        # Check if Kxx and Kyy have already been supplied, if not compute them
        if Kxx is None:
            Kxx = self._kernel_same(X)
        if Kyy is None:
            if sym:
                Kyy = Kxx
            else:
                Kyy = self._kernel_same(Y)

        # Compute the Gram matrix
        batch_X, batch_Y = X.shape[0], Y.shape[0]

        if batch_X <= max_batch and batch_Y <= max_batch:
            return self._Gram(X, Y, Kxx, Kyy, interval)

        elif batch_X <= max_batch and batch_Y > max_batch:
            cutoff = int(batch_Y/2)
            Y1, Y2 = Y[:cutoff], Y[cutoff:]
            Kyy1, Kyy2 = Kyy[:cutoff], Kyy[cutoff:]
            K1 = self.compute_Gram(X, Y1, Kxx, Kyy1, interval, sym=False, max_batch=max_batch)
            K2 = self.compute_Gram(X, Y2, Kxx, Kyy2, interval, sym=False, max_batch=max_batch)
            return torch.cat((K1, K2), dim=1)

        elif batch_X > max_batch and batch_Y <= max_batch:
            cutoff = int(batch_X/2)
            X1, X2 = X[:cutoff], X[cutoff:]
            Kxx1, Kxx2 = Kxx[:cutoff], Kxx[cutoff:]
            K1 = self.compute_Gram(X1, Y, Kxx1, Kyy, interval, sym=False, max_batch=max_batch)
            K2 = self.compute_Gram(X2, Y, Kxx2, Kyy, interval, sym=False, max_batch=max_batch)
            return torch.cat((K1, K2), dim=0)

        cutoff_X, cutoff_Y = int(batch_X/2), int(batch_Y/2)
        X1, X2 = X[:cutoff_X], X[cutoff_X:]
        Kxx1, Kxx2 = Kxx[:cutoff_X], Kxx[cutoff_X:]
        Y1, Y2 = Y[:cutoff_Y], Y[cutoff_Y:]
        Kyy1, Kyy2 = Kyy[:cutoff_Y], Kyy[cutoff_Y:]

        K11 = self.compute_Gram(X1, Y1, Kxx1, Kyy1, interval, sym=sym, max_batch=max_batch)

        K12 = self.compute_Gram(X1, Y2, Kxx1, Kyy2, interval, sym=False, max_batch=max_batch)
        if sym:
            K21 = K12.swapaxes(0, 1)
        else:
            K21 = self.compute_Gram(X2, Y1, Kxx2, Kyy1, interval, sym=False, max_batch=max_batch)

        K22 = self.compute_Gram(X2, Y2, Kxx2, Kyy2, interval, sym=sym, max_batch=max_batch)

        K_top = torch.cat((K11, K12), dim=1)
        K_bottom = torch.cat((K21, K22), dim=1)
        return torch.cat((K_top, K_bottom), dim=0)

    def _kernel_same(self, X: torch.Tensor,
                     interval: torch.Tensor = None) -> torch.Tensor:
        """
        Computes \kappa_xx.

        Parameters
        ----------
        X: torch.Tensor (batch_x, timesteps_x, d)
        interval: torch.Tensor (timesteps_x)
            - The interval over which elements of X are sampled
            - If None then interval = torch.linspace(0, 1, timesteps_x)
            - Defaults to None

        Returns
        -------
        Kxx: torch.Tensor (batch_x, timesteps_x)
        """

        batch_x, timesteps_x = X.shape[0], X.shape[1]
        var_0, var_A, var_b = self.std_0**2, self.std_A**2, self.std_b**2

        if interval is None:
            interval = torch.linspace(0, 1, timesteps_x)
        dt = interval.diff(dim=0)

        # dXdX: (batch_x, timesteps_x-1)
        # dXdX[i, t] = < dx_i(t), dx_i(t) > = \sum_{k=1}^d dX[i, t, k] * dX[i, t, k]
        dXdX = (torch.diff(X, dim=1)*torch.diff(X, dim=1)).sum(dim=-1)

        # Initialize Kernel
        # K: (batch_x, batch_y, timesteps_x, timesteps_y)
        Kxx = torch.zeros((batch_x, timesteps_x)).to(X.device.type)
        Kxx[..., 0] = var_0

        def compute_next(t):
            past = Kxx[..., t]
            innovation = (var_A * self.V_phi(Kxx[..., t], Kxx[..., t], Kxx[..., t]) + var_b) * dXdX[..., t]/dt[t]
            return past + innovation

        for t in range(timesteps_x-1):
            # Compute Kxx[:, t+1]
            Kxx[..., t+1] = compute_next(t)

        return Kxx

    def _Gram(self,
              X: torch.Tensor, Y: torch.Tensor,
              Kxx: torch.Tensor,
              Kyy: torch.Tensor,
              interval: torch.Tensor = None) -> torch.Tensor:
        """
        Computes the Neural Signature Kernel Gram matrix.
        Recall

            \kappa^{x,y}(t+1) = \kappa^{x,y}(t)
                                + (sigma_A^2*V_phi(K^{x,x}(t), K^{x,y}(t), K^{y,y}(t)) + sigma_b^2) <dx_t, dy_t>/dt

        !! Make sure elements in X and Y all sample on the same grid !!

        Parameters
        ----------
        X: torch.Tensor (batch_x, timesteps, d)
        Y: torch.Tensor (batch_y, timesteps, d)
        Kxx: torch.Tensor (batch_x, timesteps)
        Kyy: torch.Tensor (batch_y, timesteps)
        interval: torch.Tensor (timesteps)
            - The interval over which elements of X and Y are sampled
            - If None then interval = torch.linspace(0, 1, timesteps_x)
            - Defaults to None

        Returns
        -------
        K: torch.Tensor (batch_x, batch_y, timesteps)
        """

        batch_x, batch_y = X.shape[0], Y.shape[0]
        timesteps = X.shape[1]

        var_0, var_A, var_b = self.std_0**2, self.std_A**2, self.std_b**2

        if interval is None:
            interval = torch.linspace(0, 1, timesteps)
        dt = interval.diff(dim=0)

        # dXdY: (batch_x, batch_y, timesteps-1)
        # dXdY[i, j, t] = < dx_i(t), dy_j(t) > = \sum_{k=1}^d dX[i, *, t, k] * dY[*, j, t, k]
        dXdY = (torch.diff(X, dim=1).unsqueeze(1)*torch.diff(Y, dim=1).unsqueeze(0)).sum(dim=-1)

        # Initialize Kernel
        # K: (batch_x, batch_y, timesteps)
        K = torch.zeros((batch_x, batch_y, timesteps)).to(X.device.type)
        K[..., 0] = var_0

        # Helper Function
        def compute_next(t):
            past = K[..., t]
            innovation = (var_A * self.V_phi(Kxx.unsqueeze(0)[..., t], K[..., t], Kyy.unsqueeze(1)[..., t]) + var_b) * dXdY[..., t]/dt[t]
            return past + innovation

        for t in range(timesteps-1):
            # Compute K[:, :, t+1]
            K[..., t+1] = compute_next(t)

        return K

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
            raise Exception("d_X and d_Y must be the same!")

        if not (X.shape[1] == Y.shape[1]):
            raise NotImplementedError("Different Grid case not yet implemented. \n"
                                      "Use torchcde package to interpolate the data")


# =============================================================================================
# Neural Tangent Kernels
# =============================================================================================


class OmegaKer(torch.nn.Module):
    """
        # Neural Tangent Neural Signature Kernel - Omega Kernel

        d^2 Omega_phi^{x,y}(s,t) = [ {\dot V}_phi^{x,y}(s,t) * Omega_phi^{x,y}(s,t) ] <dx_s, dy_t>
    """

    def __init__(self,
                 V_phi: Callable[[tuple[float, float, float], float]] = id_phi,
                 V_dot_phi: Callable[[tuple[float, float, float], float]] = id_dot_phi,
                 sigmas: dict[str, float] = {"sigma_0": 1.0, "sigma_A": 1.0, "sigma_b": 0.0}) -> None:
        """
        Parameters
        ----------
        V_phi : function tuple[float, float, float] -> float
            - (a,b,c) -> V_phi(\Sigma) with \Sigma := [[a,b],[b,c]]
            - Must be streamable.
            - Defaults to (a,b,c) -> b
         V_phi : function tuple[float, float, float] -> float
            - (a,b,c) -> {\dot V}_phi(\Sigma) with \Sigma := [[a,b],[b,c]]
            - Must be streamable.
            - Defaults to (a,b,c) -> 0.0
        sigmas: Dict[str, float]
            - Must contain the keys "sigma_0", "sigma_A", "sigma_b" with float values.
            - Defaults to {"sigma_0": 1.0, "sigma_A": 1.0, "sigma_b": 0.0}.
        """

        super(OmegaKer, self).__init__()

        self.V_phi = V_phi
        self.V_dot_phi = V_dot_phi

        self.sigmas = sigmas

        self.std_0 = sigmas["sigma_0"]
        self.std_A = sigmas["sigma_A"]
        self.std_b = sigmas["sigma_b"]

        # The base Neural Signature Kernel
        self.phiSK = NeuralSigKer(self.V_phi, self.sigmas)

    def kernel(self, X: torch.Tensor, Y: torch.Tensor,
               Kxx: torch.Tensor = None,
               Kyy: torch.Tensor = None,
               Kxy: torch.Tensor = None,
               sym: bool = False) -> torch.Tensor:
        """
        Computes the batched Omega NTK on R^d-valued paths.
        !! Make sure elements in X and Y all sample on the same grid !!
        !! Make batches of X and Y are equal !!

        Parameters
        ----------
        X: torch.Tensor (batch, timesteps_x, d)
        Y: torch.Tensor (batch, timesteps_y, d)
        Kxx: torch.Tensor (batch, timesteps_x, timesteps_x)
            - The batched NSK on X: Kxx[i,s,t] = K_phi^{x_i,x_i}(s,t)
            - if None then compute from scratch, else use the given data.
            - !! Must have all times !!
        Kyy: torch.Tensor(batch, timesteps_y, timesteps_y)
            - The batched NSK on Y: Kyy[i,s,t] = K_phi^{y_i,y_i}(s,t)
            - if None then compute from scratch, else use the given data.
            - !! Must have all times !!
        Kxy: torch.Tensor(batch, timesteps_x, timesteps_y)
            - The batched NSK on (X,Y): Kyy[i,s,t] = K_phi^{x_i,y_i}(s,t)
            - if None then compute from scratch, else use the given data.
            - !! Must have all times !!
        sym: bool
            - if True then X and Y are treated as being equal .
            - !! This is done WITHOUT CHECKS!!

        Returns
        -------
        Omega: torch.Tensor (batch, timesteps_x, timesteps_y)
            - Omega[i, s, t] = Omega_{phi}^{x_i,y_i}(s,t)
        """

        # Some preliminary checks

        ## Correct Shape
        if not (X.shape == Y.shape):
            raise Exception("X and Y must have the same shape!")
        ## Check if Kxx and Kyy have already been supplied, if not compute them
        if Kxx is None:
            Kxx = self.phiSK._kernel_same(X)
        if Kyy is None:
            Kyy = self.phiSK._kernel_same(Y)
        if Kxy is None:
            Kxy = self.phiSK.kernel(X, Y, Kxx, Kyy, sym=sym)

        # The actual computation

        batch, timesteps = X.shape[0], X.shape[1]
        var_0, var_A = self.std_0**2, self.std_A**2

        ## dXdY: (batch, timesteps-1, timesteps-1)
        ## dXdY[i, s, t] = < dx_i(s), dy_i(t) > = \sum_{k=1}^d dX[i, s, *, k] * dY[i, *, t, k]
        dXdY = (torch.diff(X, dim=1).unsqueeze(2)*torch.diff(Y, dim=1).unsqueeze(1)).sum(dim=-1)

        ## Initialize Kernel
        ## Omega: (batch, timesteps, timesteps)
        Omega = torch.zeros((batch, timesteps, timesteps)).to(X.device.type)
        Omega[:, 0, :] = var_0
        Omega[:, :, 0] = var_0

        def compute_next(s, t):
            V_dot_now = var_A * self.V_dot_phi(Kxx[..., s, s], Kxy[..., s, t], Kyy[..., t, t])
            past = Omega[..., s+1, t] + Omega[..., s, t+1] - Omega[..., s, t]
            innovation = V_dot_now * Omega[..., s, t] * dXdY[..., s, t]
            return past + innovation

        for s in range(timesteps-1):
            ## Compute Omega[:, :, s+1, t+1] and omega[:, :, t+1, s+1] when t < s
            for t in range(s):
                Omega[..., s+1, t+1] = compute_next(s, t)
                Omega[..., t+1, s+1] = compute_next(t, s)
            ## Compute K[:, :, s+1, t+1] when t==s
            Omega[..., s+1, s+1] = compute_next(s, s)

        return Omega

    def compute_Gram(self,
                     X: torch.Tensor, Y: torch.Tensor,
                     Kxx: torch.Tensor = None,
                     Kyy: torch.Tensor = None,
                     Kxy: torch.Tensor = None,
                     sym: bool = False,
                     max_batch: int = 50) -> torch.Tensor:
        """
        Computes the batched Omega NTK Gram matrix on R^d paths.
        !! Make sure elements in X and Y all sample on the same grid !!

        Parameters
        ----------
        X: torch.Tensor (batch_x, timesteps_x, d)
        Y: torch.Tensor (batch_y, timesteps_y, d)
        Kxx: torch.Tensor (batch_x, timesteps_x, timesteps_x)
            - The batched NSK on X: Kxx[i,s,t] = K_phi^{x_i,x_i}(s,t)
            - if None then compute from scratch, else use the given data.
            - !! Must have all times !!
        Kyy: torch.Tensor(batch_y, timesteps_y, timesteps_y)
            - The batched NSK on Y: Kyy[j,s,t] = K_phi^{y_j,y_j}(s,t)
            - if None then compute from scratch, else use the given data.
            - !! Must have all times !!
        Kxy: torch.Tensor(batch_x, batch_y, timesteps_x, timesteps_y)
            - The batched NSK on (X,Y): Kyy[i,s,t] = K_phi^{x_i,y_i}(s,t)
            - if None then compute from scratch, else use the given data.
            - !! Must have all times !!
        max_batch: int
            - the maximum batch size for the computation.
        sym: bool
            - if True then X and Y are treated as being equal .
            - This is done WITHOUT CHECKS!!

        Returns
        -------
        Omega: torch.Tensor (batch_x, batch_y, timesteps_x, timesteps_y)
            - Omega[i, j, s, t] = Omega_{phi}^{x_i,y_j}(s,t)
        """

        self._compatibility_checks(X, Y)

        # Check if Kxx and Kyy have already been supplied, if not compute them
        if Kxx is None:
            Kxx = self.phiSK._kernel_same(X)
        if Kyy is None:
            if sym:
                Kyy = Kxx
            else:
                Kyy = self.phiSK._kernel_same(Y)
        if Kxy is None:
            Kxy = self.phiSK.compute_Gram(X, Y, Kxx, Kyy, sym=sym, max_batch=max_batch)

        # Compute the Gram matrix
        batch_X, batch_Y = X.shape[0], Y.shape[0]

        if batch_X <= max_batch and batch_Y <= max_batch:
            return self._Gram(X, Y, Kxx, Kyy, Kxy)

        elif batch_X <= max_batch and batch_Y > max_batch:
            cutoff = int(batch_Y/2)
            Y1, Y2 = Y[:cutoff], Y[cutoff:]
            Kyy1, Kyy2 = Kyy[:cutoff], Kyy[cutoff:]
            Kxy1, Kxy2 = Kxy[:, :cutoff], Kxy[:, cutoff:]
            K1 = self.compute_Gram(X, Y1, Kxx, Kyy1, Kxy1, sym=False, max_batch=max_batch)
            K2 = self.compute_Gram(X, Y2, Kxx, Kyy2, Kxy2, sym=False, max_batch=max_batch)
            return torch.cat((K1, K2), dim=1)

        elif batch_X > max_batch and batch_Y <= max_batch:
            cutoff = int(batch_X/2)
            X1, X2 = X[:cutoff], X[cutoff:]
            Kxx1, Kxx2 = Kxx[:cutoff], Kxx[cutoff:]
            Kxy1, Kxy2 = Kxy[:cutoff, :], Kxy[cutoff:, :]
            K1 = self.compute_Gram(X1, Y, Kxx1, Kyy, Kxy1, sym=False, max_batch=max_batch)
            K2 = self.compute_Gram(X2, Y, Kxx2, Kyy, Kxy2, sym=False, max_batch=max_batch)
            return torch.cat((K1, K2), dim=0)

        cutoff_X, cutoff_Y = int(batch_X/2), int(batch_Y/2)
        X1, X2 = X[:cutoff_X], X[cutoff_X:]
        Kxx1, Kxx2 = Kxx[:cutoff_X], Kxx[cutoff_X:]
        Y1, Y2 = Y[:cutoff_Y], Y[cutoff_Y:]
        Kyy1, Kyy2 = Kyy[:cutoff_Y], Kyy[cutoff_Y:]

        Kxy11, Kxy12 = Kxy[:cutoff_X, :cutoff_Y], Kxy[:cutoff_X, cutoff_Y:]
        Kxy21, Kxy22 = Kxy[cutoff_X:, :cutoff_Y], Kxy[cutoff_X:, cutoff_Y:]

        K11 = self.compute_Gram(X1, Y1, Kxx1, Kyy1, Kxy11, sym=sym, max_batch=max_batch)

        K12 = self.compute_Gram(X1, Y2, Kxx1, Kyy2, Kxy12, sym=False, max_batch=max_batch)
        if sym:
            K21 = K12.swapaxes(0, 1).swapaxes(-2, -1)
        else:
            K21 = self.compute_Gram(X2, Y1, Kxx2, Kyy1, Kxy21, sym=False, max_batch=max_batch)

        K22 = self.compute_Gram(X2, Y2, Kxx2, Kyy2, Kxy22, sym=sym, max_batch=max_batch)

        K_top = torch.cat((K11, K12), dim=1)
        K_bottom = torch.cat((K21, K22), dim=1)
        return torch.cat((K_top, K_bottom), dim=0)

    def _Gram(self,
              X: torch.Tensor, Y: torch.Tensor,
              Kxx: torch.Tensor,
              Kyy: torch.Tensor,
              Kxy: torch.Tensor) -> torch.Tensor:
        """
        Computes the Omega NTK Gram matrix.

        !! Make sure elements in X and Y all sample on the same grid !!

        Parameters
        ----------
        X: torch.Tensor (batch_x, timesteps_x, d)
        Y: torch.Tensor (batch_y, timesteps_y, d)
        Kxx: torch.Tensor (batch_x, timesteps_x, timesteps_x)
        Kyy: torch.Tensor (batch_y, timesteps_y, timesteps_y)
        Kxy: torch.Tensor(batch_x, batch_y, timesteps_x, timesteps_y)

        Returns
        -------
        Omega: torch.Tensor (batch_x, batch_y, timesteps_x, timesteps_y)
            - Omega[i,j,s,t] = Omega_phi^{x_i,y_j}(s,t)
        """

        batch_x, batch_y = X.shape[0], Y.shape[0]
        timesteps_x, timesteps_y = X.shape[1], Y.shape[1]

        var_0, var_A = self.std_0**2, self.std_A**2

        # dXdY: (batch_x, batch_y, timesteps_x-1, timesteps_y-1)
        # dXdY[i, j, s, t] = < dx_i(s), dy_j(t) > = \sum_{k=1}^d dX[i, *, s, *, k] * dY[*, j, *, t, k]
        dXdY = (torch.diff(X, dim=1).unsqueeze(1).unsqueeze(3)*torch.diff(Y, dim=1).unsqueeze(0).unsqueeze(2)).sum(dim=-1)

        # Initialize Kernel
        # omega: (batch_x, batch_y, timesteps_x, timesteps_y)
        Omega = torch.zeros((batch_x, batch_y, timesteps_x, timesteps_y)).to(X.device.type)
        Omega[..., 0, :] = var_0
        Omega[..., :, 0] = var_0

        # Helper Function
        def compute_next(s, t):
            V_dot_now = var_A * self.V_dot_phi(Kxx.unsqueeze(1)[..., s, s], Kxy[..., s, t], Kyy.unsqueeze(0)[..., t, t])
            past = Omega[..., s+1, t] + Omega[..., s, t+1] - Omega[..., s, t]
            innovation = V_dot_now * Omega[..., s, t] * dXdY[..., s, t]
            return past + innovation

        for s in range(timesteps_x-1):
            ## Compute Omega[:, :, s+1, t+1] and omega[:, :, t+1, s+1] when t < s
            for t in range(s):
                Omega[..., s+1, t+1] = compute_next(s, t)
                Omega[..., t+1, s+1] = compute_next(t, s)
            ## Compute K[:, :, s+1, t+1] when t==s
            Omega[..., s+1, s+1] = compute_next(s, s)

        return Omega

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
            raise Exception("d_X and d_Y must be the same!")

        if not (X.shape[1] == Y.shape[1]):
            raise NotImplementedError("Different Grid case not yet implemented. \n"
                                      "Use torchcde package to interpolate the data")


class XiKer(torch.nn.Module):
    """
        # Neural Tangent Neural Signature Kernel - Xi Kernel

        d^2 Xi_phi^{x,y}(s,t) = [ V_phi^{x,y}(s,t) + {\dot V}_phi^{x,y}(s,t) * Xi_phi^{x,y}(s,t) ] <dx_s, dy_t>
    """

    def __init__(self,
                 V_phi: Callable[[tuple[float, float, float], float]] = id_phi,
                 V_dot_phi: Callable[[tuple[float, float, float], float]] = id_dot_phi,
                 sigmas: dict[str, float] = {"sigma_0": 1.0, "sigma_A": 1.0, "sigma_b": 0.0}) -> None:
        """
        Parameters
        ----------
        V_phi : function tuple[float, float, float] -> float
            - (a,b,c) -> V_phi(\Sigma) with \Sigma := [[a,b],[b,c]]
            - Must be streamable.
            - Defaults to (a,b,c) -> b
         V_phi : function tuple[float, float, float] -> float
            - (a,b,c) -> {\dot V}_phi(\Sigma) with \Sigma := [[a,b],[b,c]]
            - Must be streamable.
            - Defaults to (a,b,c) -> 0.0
        sigmas: Dict[str, float]
            - Must contain the keys "sigma_0", "sigma_A", "sigma_b" with float values.
            - Defaults to {"sigma_0": 1.0, "sigma_A": 1.0, "sigma_b": 0.0}.
        """

        super(XiKer, self).__init__()

        self.V_phi = V_phi
        self.V_dot_phi = V_dot_phi

        self.sigmas = sigmas

        self.std_0 = sigmas["sigma_0"]
        self.std_A = sigmas["sigma_A"]
        self.std_b = sigmas["sigma_b"]

        # The base Neural Signature Kernel
        self.phiSK = NeuralSigKer(self.V_phi, self.sigmas)

    def kernel(self, X: torch.Tensor, Y: torch.Tensor,
               Kxx: torch.Tensor = None,
               Kyy: torch.Tensor = None,
               Kxy: torch.Tensor = None,
               sym: bool = False) -> torch.Tensor:
        """
        Computes the batched Xi NTK on R^d-valued paths.
        !! Make sure elements in X and Y all sample on the same grid !!
        !! Make batches of X and Y are equal !!

        Parameters
        ----------
        X: torch.Tensor (batch, timesteps_x, d)
        Y: torch.Tensor (batch, timesteps_y, d)
        Kxx: torch.Tensor (batch, timesteps_x, timesteps_x)
            - The batched NSK on X: Kxx[i,s,t] = K_phi^{x_i,x_i}(s,t)
            - if None then compute from scratch, else use the given data.
            - !! Must have all times !!
        Kyy: torch.Tensor(batch, timesteps_y, timesteps_y)
            - The batched NSK on Y: Kyy[i,s,t] = K_phi^{y_i,y_i}(s,t)
            - if None then compute from scratch, else use the given data.
            - !! Must have all times !!
        Kxy: torch.Tensor(batch, timesteps_x, timesteps_y)
            - The batched NSK on (X,Y): Kyy[i,s,t] = K_phi^{x_i,y_i}(s,t)
            - if None then compute from scratch, else use the given data.
            - !! Must have all times !!
        sym: bool
            - if True then X and Y are treated as being equal .
            - !! This is done WITHOUT CHECKS!!

        Returns
        -------
        Xi: torch.Tensor (batch, timesteps_x, timesteps_y)
            - Xi[i, s, t] = Xi_{phi}^{x_i,y_i}(s,t)
        """

        # Some preliminary checks

        ## Correct Shape
        if not (X.shape == Y.shape):
            raise Exception("X and Y must have the same shape!")
        ## Check if Kxx and Kyy have already been supplied, if not compute them
        if Kxx is None:
            Kxx = self.phiSK._kernel_same(X)
        if Kyy is None:
            Kyy = self.phiSK._kernel_same(Y)
        if Kxy is None:
            Kxy = self.phiSK.kernel(X, Y, Kxx, Kyy, sym=sym)

        # The actual computation

        batch, timesteps = X.shape[0], X.shape[1]
        var_A, var_b = self.std_A**2, self.std_b**2

        ## dXdY: (batch, timesteps-1, timesteps-1)
        ## dXdY[i, s, t] = < dx_i(s), dy_i(t) > = \sum_{k=1}^d dX[i, s, *, k] * dY[i, *, t, k]
        dXdY = (torch.diff(X, dim=1).unsqueeze(2)*torch.diff(Y, dim=1).unsqueeze(1)).sum(dim=-1)

        ## Initialize Kernel
        ## Xi: (batch, timesteps, timesteps)
        Xi = torch.zeros((batch, timesteps, timesteps)).to(X.device.type)

        def compute_next(s, t):
            V_now = var_A * self.V_phi(Kxx[..., s, s], Kxy[..., s, t], Kyy[..., t, t]) + var_b
            V_dot_now = var_A * self.V_dot_phi(Kxx[..., s, s], Kxy[..., s, t], Kyy[..., t, t])
            past = Xi[..., s+1, t] + Xi[..., s, t+1] - Xi[..., s, t]
            innovation = (V_now + V_dot_now * Xi[..., s, t]) * dXdY[..., s, t]
            return past + innovation

        for s in range(timesteps-1):
            ## Compute Omega[:, :, s+1, t+1] and omega[:, :, t+1, s+1] when t < s
            for t in range(s):
                Xi[..., s+1, t+1] = compute_next(s, t)
                Xi[..., t+1, s+1] = compute_next(t, s)
            ## Compute K[:, :, s+1, t+1] when t==s
            Xi[..., s+1, s+1] = compute_next(s, s)

        return Xi

    def compute_Gram(self,
                     X: torch.Tensor, Y: torch.Tensor,
                     Kxx: torch.Tensor = None,
                     Kyy: torch.Tensor = None,
                     Kxy: torch.Tensor = None,
                     sym: bool = False,
                     max_batch: int = 50) -> torch.Tensor:
        """
        Computes the batched Xi NTK Gram matrix on R^d paths.
        !! Make sure elements in X and Y all sample on the same grid !!

        Parameters
        ----------
        X: torch.Tensor (batch_x, timesteps_x, d)
        Y: torch.Tensor (batch_y, timesteps_y, d)
        Kxx: torch.Tensor (batch_x, timesteps_x, timesteps_x)
            - The batched NSK on X: Kxx[i,s,t] = K_phi^{x_i,x_i}(s,t)
            - if None then compute from scratch, else use the given data.
            - !! Must have all times !!
        Kyy: torch.Tensor(batch_y, timesteps_y, timesteps_y)
            - The batched NSK on Y: Kyy[j,s,t] = K_phi^{y_j,y_j}(s,t)
            - if None then compute from scratch, else use the given data.
            - !! Must have all times !!
        Kxy: torch.Tensor(batch_x, batch_y, timesteps_x, timesteps_y)
            - The batched NSK on (X,Y): Kyy[i,s,t] = K_phi^{x_i,y_i}(s,t)
            - if None then compute from scratch, else use the given data.
            - !! Must have all times !!
        max_batch: int
            - the maximum batch size for the computation.
        sym: bool
            - if True then X and Y are treated as being equal .
            - This is done WITHOUT CHECKS!!

        Returns
        -------
        Xi: torch.Tensor (batch_x, batch_y, timesteps_x, timesteps_y)
            - Xi[i, j, s, t] = Xi_{phi}^{x_i,y_j}(s,t)
        """

        self._compatibility_checks(X, Y)

        # Check if Kxx and Kyy have already been supplied, if not compute them
        if Kxx is None:
            Kxx = self.phiSK._kernel_same(X)
        if Kyy is None:
            if sym:
                Kyy = Kxx
            else:
                Kyy = self.phiSK._kernel_same(Y)
        if Kxy is None:
            Kxy = self.phiSK.compute_Gram(X, Y, Kxx, Kyy, sym=sym, max_batch=max_batch)

        # Compute the Gram matrix
        batch_X, batch_Y = X.shape[0], Y.shape[0]

        if batch_X <= max_batch and batch_Y <= max_batch:
            return self._Gram(X, Y, Kxx, Kyy, Kxy)

        elif batch_X <= max_batch and batch_Y > max_batch:
            cutoff = int(batch_Y/2)
            Y1, Y2 = Y[:cutoff], Y[cutoff:]
            Kyy1, Kyy2 = Kyy[:cutoff], Kyy[cutoff:]
            Kxy1, Kxy2 = Kxy[:, :cutoff], Kxy[:, cutoff:]
            K1 = self.compute_Gram(X, Y1, Kxx, Kyy1, Kxy1, sym=False, max_batch=max_batch)
            K2 = self.compute_Gram(X, Y2, Kxx, Kyy2, Kxy2, sym=False, max_batch=max_batch)
            return torch.cat((K1, K2), dim=1)

        elif batch_X > max_batch and batch_Y <= max_batch:
            cutoff = int(batch_X/2)
            X1, X2 = X[:cutoff], X[cutoff:]
            Kxx1, Kxx2 = Kxx[:cutoff], Kxx[cutoff:]
            Kxy1, Kxy2 = Kxy[:cutoff, :], Kxy[cutoff:, :]
            K1 = self.compute_Gram(X1, Y, Kxx1, Kyy, Kxy1, sym=False, max_batch=max_batch)
            K2 = self.compute_Gram(X2, Y, Kxx2, Kyy, Kxy2, sym=False, max_batch=max_batch)
            return torch.cat((K1, K2), dim=0)

        cutoff_X, cutoff_Y = int(batch_X/2), int(batch_Y/2)
        X1, X2 = X[:cutoff_X], X[cutoff_X:]
        Kxx1, Kxx2 = Kxx[:cutoff_X], Kxx[cutoff_X:]
        Y1, Y2 = Y[:cutoff_Y], Y[cutoff_Y:]
        Kyy1, Kyy2 = Kyy[:cutoff_Y], Kyy[cutoff_Y:]

        Kxy11, Kxy12 = Kxy[:cutoff_X, :cutoff_Y], Kxy[:cutoff_X, cutoff_Y:]
        Kxy21, Kxy22 = Kxy[cutoff_X:, :cutoff_Y], Kxy[cutoff_X:, cutoff_Y:]

        K11 = self.compute_Gram(X1, Y1, Kxx1, Kyy1, Kxy11, sym=sym, max_batch=max_batch)

        K12 = self.compute_Gram(X1, Y2, Kxx1, Kyy2, Kxy12, sym=False, max_batch=max_batch)
        if sym:
            K21 = K12.swapaxes(0, 1).swapaxes(-2, -1)
        else:
            K21 = self.compute_Gram(X2, Y1, Kxx2, Kyy1, Kxy21, sym=False, max_batch=max_batch)

        K22 = self.compute_Gram(X2, Y2, Kxx2, Kyy2, Kxy22, sym=sym, max_batch=max_batch)

        K_top = torch.cat((K11, K12), dim=1)
        K_bottom = torch.cat((K21, K22), dim=1)
        return torch.cat((K_top, K_bottom), dim=0)

    def _Gram(self,
              X: torch.Tensor, Y: torch.Tensor,
              Kxx: torch.Tensor,
              Kyy: torch.Tensor,
              Kxy: torch.Tensor) -> torch.Tensor:
        """
        Computes the Xi NTK Gram matrix.

        !! Make sure elements in X and Y all sample on the same grid !!

        Parameters
        ----------
        X: torch.Tensor (batch_x, timesteps_x, d)
        Y: torch.Tensor (batch_y, timesteps_y, d)
        Kxx: torch.Tensor (batch_x, timesteps_x, timesteps_x)
        Kyy: torch.Tensor (batch_y, timesteps_y, timesteps_y)
        Kxy: torch.Tensor(batch_x, batch_y, timesteps_x, timesteps_y)

        Returns
        -------
        Omega: torch.Tensor (batch_x, batch_y, timesteps_x, timesteps_y)
            - Omega[i,j,s,t] = Omega_phi^{x_i,y_j}(s,t)
        """

        batch_x, batch_y = X.shape[0], Y.shape[0]
        timesteps_x, timesteps_y = X.shape[1], Y.shape[1]

        var_A, var_b = self.std_A**2, self.std_b**2

        # dXdY: (batch_x, batch_y, timesteps_x-1, timesteps_y-1)
        # dXdY[i, j, s, t] = < dx_i(s), dy_j(t) > = \sum_{k=1}^d dX[i, *, s, *, k] * dY[*, j, *, t, k]
        dXdY = (torch.diff(X, dim=1).unsqueeze(1).unsqueeze(3)*torch.diff(Y, dim=1).unsqueeze(0).unsqueeze(2)).sum(dim=-1)

        # Initialize Kernel
        # Xi: (batch_x, batch_y, timesteps_x, timesteps_y)
        Xi = torch.zeros((batch_x, batch_y, timesteps_x, timesteps_y)).to(X.device.type)

        # Helper Function
        def compute_next(s, t):
            V_now = var_A * self.V_phi(Kxx.unsqueeze(1)[..., s, s], Kxy[..., s, t], Kyy.unsqueeze(1)[..., t, t]) + var_b
            V_dot_now = var_A * self.V_dot_phi(Kxx.unsqueeze(1)[..., s, s], Kxy[..., s, t], Kyy.unsqueeze(0)[..., t, t])
            past = Xi[..., s+1, t] + Xi[..., s, t+1] - Xi[..., s, t]
            innovation = (V_now + V_dot_now * Xi[..., s, t]) * dXdY[..., s, t]
            return past + innovation

        for s in range(timesteps_x-1):
            ## Compute Omega[:, :, s+1, t+1] and omega[:, :, t+1, s+1] when t < s
            for t in range(s):
                Xi[..., s+1, t+1] = compute_next(s, t)
                Xi[..., t+1, s+1] = compute_next(t, s)
            ## Compute K[:, :, s+1, t+1] when t==s
            Xi[..., s+1, s+1] = compute_next(s, s)

        return Xi

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
            raise Exception("d_X and d_Y must be the same!")

        if not (X.shape[1] == Y.shape[1]):
            raise NotImplementedError("Different Grid case not yet implemented. \n"
                                      "Use torchcde package to interpolate the data")


class NTKer(torch.nn.Module):
    """
        # Neural Tangent Neural Signature Kernel - NTK Kernel

        Theta_phi^{x,y}(s,t) = K_phi^{x,y}(s,t) + Omega_phi^{x,y}(s,t) + Xi_phi^{x,y}(s,t)
    """

    def __init__(self,
                 V_phi: Callable[[tuple[float, float, float], float]] = id_phi,
                 V_dot_phi: Callable[[tuple[float, float, float], float]] = id_dot_phi,
                 sigmas: dict[str, float] = {"sigma_0": 1.0, "sigma_A": 1.0, "sigma_b": 0.0}) -> None:
        """
        Parameters
        ----------
        V_phi : function tuple[float, float, float] -> float
            - (a,b,c) -> V_phi(\Sigma) with \Sigma := [[a,b],[b,c]]
            - Must be streamable.
            - Defaults to (a,b,c) -> b
         V_phi : function tuple[float, float, float] -> float
            - (a,b,c) -> {\dot V}_phi(\Sigma) with \Sigma := [[a,b],[b,c]]
            - Must be streamable.
            - Defaults to (a,b,c) -> 0.0
        sigmas: Dict[str, float]
            - Must contain the keys "sigma_0", "sigma_A", "sigma_b" with float values.
            - Defaults to {"sigma_0": 1.0, "sigma_A": 1.0, "sigma_b": 0.0}.
        """

        super(NTKer, self).__init__()

        self.V_phi = V_phi
        self.V_dot_phi = V_dot_phi

        self.sigmas = sigmas

        self.std_0 = sigmas["sigma_0"]
        self.std_A = sigmas["sigma_A"]
        self.std_b = sigmas["sigma_b"]

        # The needed Kernels
        self.phiSK = NeuralSigKer(self.V_phi, self.sigmas)
        self.OmegaK = OmegaKer(self.V_phi, self.V_dot_phi, self.sigmas)
        self.XiK = XiKer(self.V_phi, self.V_dot_phi, self.sigmas)

        self.K = None
        self.Gram = None

    def kernel(self, X: torch.Tensor, Y: torch.Tensor,
               Kxx: torch.Tensor = None,
               Kyy: torch.Tensor = None,
               Kxy: torch.Tensor = None,
               sym: bool = False) -> torch.Tensor:
        """
        Computes the batched Full NTK on R^d-valued paths in a neat package.
        !! Make sure elements in X and Y all sample on the same grid !!
        !! Make batches of X and Y are equal !!

        Parameters
        ----------
        X: torch.Tensor (batch, timesteps_x, d)
        Y: torch.Tensor (batch, timesteps_y, d)
        Kxx: torch.Tensor (batch, timesteps_x, timesteps_x)
            - The batched NSK on X: Kxx[i,s,t] = K_phi^{x_i,x_i}(s,t)
            - if None then compute from scratch, else use the given data.
            - !! Must have all times !!
        Kyy: torch.Tensor(batch, timesteps_y, timesteps_y)
            - The batched NSK on Y: Kyy[i,s,t] = K_phi^{y_i,y_i}(s,t)
            - if None then compute from scratch, else use the given data.
            - !! Must have all times !!
        Kxy: torch.Tensor(batch, timesteps_x, timesteps_y)
            - The batched NSK on (X,Y): Kyy[i,s,t] = K_phi^{x_i,y_i}(s,t)
            - if None then compute from scratch, else use the given data.
            - !! Must have all times !!
        sym: bool
            - if True then X and Y are treated as being equal .
            - !! This is done WITHOUT CHECKS!!

        Returns
        -------
        NTK: torch.Tensor (batch, timesteps_x, timesteps_y)
            - NTK[i, s, t] = Theta_{phi}^{x_i,y_i}(s,t)
        """

        # Some preliminary checks

        ## Correct Shape
        if not (X.shape == Y.shape):
            raise Exception("X and Y must have the same shape!")
        ## Check if Kxx and Kyy have already been supplied, if not compute them
        if Kxx is None:
            Kxx = self.phiSK._kernel_same(X)
        if Kyy is None:
            Kyy = self.phiSK._kernel_same(Y)
        if Kxy is None:
            Kxy = self.phiSK.kernel(X, Y, Kxx, Kyy, sym=sym)

        OmegaK_kernel = self.OmegaK.kernel(X, Y, Kxx, Kyy, Kxy, sym=sym)
        XiK_kernel = self.XiK.kernel(X, Y, Kxx, Kyy, Kxy, sym=sym)

        Theta = Kxy + OmegaK_kernel + XiK_kernel
        self.K = Theta

        return Theta

    def compute_Gram(self,
                     X: torch.Tensor, Y: torch.Tensor,
                     Kxx: torch.Tensor = None,
                     Kyy: torch.Tensor = None,
                     Kxy: torch.Tensor = None,
                     sym: bool = False,
                     max_batch: int = 50) -> torch.Tensor:
        """
        Computes the batched Full NTK Gram matrix on R^d paths in a neat package.
        !! Make sure elements in X and Y all sample on the same grid !!

        Parameters
        ----------
        X: torch.Tensor (batch_x, timesteps_x, d)
        Y: torch.Tensor (batch_y, timesteps_y, d)
        Kxx: torch.Tensor (batch_x, timesteps_x, timesteps_x)
            - The batched NSK on X: Kxx[i,s,t] = K_phi^{x_i,x_i}(s,t)
            - if None then compute from scratch, else use the given data.
            - !! Must have all times !!
        Kyy: torch.Tensor(batch_y, timesteps_y, timesteps_y)
            - The batched NSK on Y: Kyy[j,s,t] = K_phi^{y_j,y_j}(s,t)
            - if None then compute from scratch, else use the given data.
            - !! Must have all times !!
        Kxy: torch.Tensor(batch_x, batch_y, timesteps_x, timesteps_y)
            - The batched NSK on (X,Y): Kyy[i,s,t] = K_phi^{x_i,y_i}(s,t)
            - if None then compute from scratch, else use the given data.
            - !! Must have all times !!
        max_batch: int
            - the maximum batch size for the computation.
        sym: bool
            - if True then X and Y are treated as being equal .
            - This is done WITHOUT CHECKS!!

        Returns
        -------
        NTK: torch.Tensor (batch_x, batch_y, timesteps_x, timesteps_y)
            - NTK[i, j, s, t] = \Theta_{phi}^{x_i,y_j}(s,t)
        """

        self._compatibility_checks(X, Y)

        # Check if Kxx and Kyy have already been supplied, if not compute them
        if Kxx is None:
            Kxx = self.phiSK._kernel_same(X)
        if Kyy is None:
            if sym:
                Kyy = Kxx
            else:
                Kyy = self.phiSK._kernel_same(Y)
        if Kxy is None:
            Kxy = self.phiSK.compute_Gram(X, Y, Kxx, Kyy, sym=sym, max_batch=max_batch)

        Omega_Gram = self.OmegaK.compute_Gram(X, Y, Kxx, Kyy, Kxy, sym=sym, max_batch=max_batch)
        Xi_Gram = self.XiK.compute_Gram(X, Y, Kxx, Kyy, Kxy, sym=sym, max_batch=max_batch)

        Theta = Kxy + Omega_Gram + Xi_Gram
        self.Gram = Theta

        return Theta

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
            raise Exception("d_X and d_Y must be the same!")

        if not (X.shape[1] == Y.shape[1]):
            raise NotImplementedError("Different Grid case not yet implemented. \n"
                                      "Use torchcde package to interpolate the data")
