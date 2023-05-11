import torch
import numpy as np


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class cResNet(torch.nn.Module):
    def __init__(self,
                 input_channels,
                 hidden_channels,
                 phi=lambda x: x,
                 sigmas=torch.tensor([1.0, 1.0, 1.0]),
                 partition=torch.linspace(0, 1, 51)):

        super(cResNet, self).__init__()

        # Initialize class parameters
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        N, d = self.hidden_channels, self.input_channels

        self.phi = phi
        self.std_0, self.std_A, self.std_b = sigmas[0], sigmas[1], sigmas[2]

        self.partition = partition
        self.M = self.partition.size()[0]

        # Sample parameters initial value (z0), matrices (Aks), biases (bks) and projector (v).

        self.v_param = torch.nn.Parameter(torch.normal(0.0, 1, size=(1, N)).to(device))
        self.z0_param = torch.nn.Parameter(torch.normal(0.0, 1, size=(N, 1)).to(device))
        self.Aks_param = torch.nn.Parameter(torch.normal(0.0, 1, size=(d, N, N)).to(device))
        self.bks_param = torch.nn.Parameter(torch.normal(0.0, 1, size=(d, N, 1)).to(device))

    def forward(self, X):
        """
                Controlled Res Net forward pass

                Parameters
                ----------
                X : torch.tensor of dim (batch, timesteps_x, d)

                Returns
                -------
                { < v^N, Y_1^{N,D}(x) > }_{x in X} : torch.tensor of dim (batch, 1)
        """

        batch, times, d = X.shape
        assert d == self.input_channels
        assert times == self.M

        v = 1 / np.sqrt(self.hidden_channels) * self.v_param
        z0 = self.std_0 * self.z0_param
        Aks = self.std_A / np.sqrt(self.hidden_channels) * self.Aks_param
        bks = self.std_b * self.bks_param
        fields = torch.cat([Aks, bks], dim=-1)  # Concatenation of the Aks and bks

        diff_X = X.diff(dim=1)  # Batched increments : (batch, timesteps_x - 1, d)

        Y = z0.unsqueeze(0).repeat(batch, 1, 1)  # Initial value : (batch, N, 1)

        for t in range(diff_X.shape[1]):

            dX = diff_X[:, t, :].unsqueeze(-1)  # Increment : (batch, d, 1)

            # Compute \sum_{k=1}^d (A_k \phi(Y) + b_k) dx^k_t
            z = self.phi(Y)  # Y ->  \phi(Y)
            z = torch.cat((z, torch.ones([batch, 1, 1]).to(device)), 1)  # z -> [z,1]
            z = z.unsqueeze(1)  # z : (batch, N+1, 1) -> z : (batch, 1, N+1, 1)
            G = fields.expand(1, -1, -1, -1)  # fields : (d, N, N+1) ->  G : (1, d, N, N+1)
            res = torch.matmul(G, z)  # res : (batch, d, N, 1)
            res = res.squeeze(-1)  # res : (batch, d, N, 1) ->  res : (batch, d, N)
            res = torch.swapaxes(res, -2, -1)  # res : (batch, d, N) -> res : (batch, N, d)
            res = torch.matmul(res, dX)  # res : (batch, N, 1)

            Y = Y + res  # Y : (batch, N, 1)

        projector = v.unsqueeze(0)  # projector : (1, 1, N)
        out = torch.matmul(projector, Y)  # out : (batch, 1, 1)
        out = out.squeeze(-1).squeeze(-1)  # out : (batch, )

        return out
