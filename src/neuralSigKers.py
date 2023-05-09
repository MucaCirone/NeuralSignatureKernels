import torch
from src.utils import id_phi

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# =============================================================================================
# Kernels
# =============================================================================================


class NeuralSigKer(torch.nn.Module):
    def __init__(self,
                 V_phi=id_phi,
                 sigmas=torch.tensor([1.0, 1.0, 1.0])):

        super(NeuralSigKer, self).__init__()

        self.V_phi = V_phi

        self.sigmas = sigmas

        self.var_0 = sigmas[0]**2
        self.var_A = sigmas[1]**2
        self.var_b = sigmas[2]**2

    def K_same(self, x):
        """
        phi-SigKernel on R^d

        Parameters
        ----------
        x : torch.Tensor of dim (timesteps_x, d)

        Returns
        -------
                    phi-SK(x, x) : torch.Tensor  (timesteps_x, timesteps_x)
        """

        var_0, var_A, var_b = self.var_0, self.var_A, self.var_b

        Delta_x = torch.diff(x, dim=0)
        T = Delta_x.shape[0]

        K = var_0 * torch.ones((T + 1, T + 1)).to(device)

        for t in range(T):
            for s in range(t):
                K[t + 1, s + 1] = K[t + 1, s] + K[t, s + 1] - K[t, s] + \
                                  (var_A * self.V_phi(K[t, t], K[t, s], K[s, s]) + var_b) * (Delta_x[t] @ Delta_x[s])

                # Use simmetricity
                K[s + 1, t + 1] = K[t + 1, s + 1]

            K[t + 1, t + 1] = 2 * K[t + 1, t] - K[t, t] + (var_A * self.V_phi(K[t, t], K[t, t], K[t, t]) + var_b) * (Delta_x[t] @ Delta_x[t])

        return K

    def kernel(self, x, y, Kxx=None, Kyy=None):
        """
        phi-SigKernel on R^d

        Parameters
        ----------
        x : torch.Tensor of dim (timesteps_x, d)
        y : torch.Tensor of dim (timesteps_y, d)

        Returns
        -------
                    phi-SK(x, y) : torch.Tensor  (timesteps_x, timesteps_y)
        """

        if not (x.dim() == 2) or not (y.dim() == 2):
            raise Exception("Either X or Y are not of type (times, dim).")
        if not (x.shape[-1] == y.shape[-1]):
            raise Exception("dim_X and dim_Y must be the same!")

        # If x == y use previous method
        X_equal_Y = (x.shape[0] == y.shape[0])
        if X_equal_Y:
            X_equal_Y = (x == y).all()

        if X_equal_Y:
            if not (Kxx is None):
                return Kxx
            if not (Kyy is None):
                return Kyy
            return self.K_same(x)

        # Check if Kxx and Kyy have already been supplied, if not compute them
        if Kxx is None:
            Kxx = self.K_same(x)
        if Kyy is None:
            Kyy = self.K_same(y)

        var_0, var_A, var_b = self.var_0, self.var_A, self.var_b

        Delta_x = torch.diff(x, dim=0)
        Delta_y = torch.diff(y, dim=0)

        T_x = Delta_x.shape[0]
        T_y = Delta_y.shape[0]

        if not (T_x == T_y):
            raise NotImplementedError("Different Grid case not yet implemented. \n"
                                      "Use torchcde package to interpolate the data")

        K = var_0 * torch.ones((T_x + 1, T_y + 1))

        for t in range(T_x):

            for s in range(t):
                K[t + 1, s + 1] = K[t + 1, s] + K[t, s + 1] - K[t, s]\
                                  + (var_A * self.V_phi(Kxx[t, t], K[t, s], Kyy[s, s]) + var_b) * (Delta_x[t] @ Delta_y[s])
            for s in range(t + 1):
                K[s + 1, t + 1] = K[s + 1, t] + K[s, t + 1] - K[s, t] \
                                  + (var_A * self.V_phi(Kxx[s, s], K[s, t], Kyy[t, t]) + var_b) * (Delta_x[s] @ Delta_y[t])

        return K
