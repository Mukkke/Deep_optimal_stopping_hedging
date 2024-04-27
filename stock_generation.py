import numpy as np
import torch


class Stock:
    def __init__(self, T, K, sigma, delta, S0, r, N, M, d, e, K_path):
        """
        T: timestep
        K: strike
        sigma:
        delta
        S0:
        r:
        N: the number of timesteps with large gird
        M: small grid within large grid
        d:  the number of financial securities.
        e:  the total number of instruments in the portfolio
        K_path: the number of paths
        """

        self.T = T
        self.K = K
        self.sigma = sigma * np.ones(d)
        self.delta = delta
        self.S0 = S0 * np.ones(d)
        self.r = r
        self.N = N
        self.M = M
        self.d = d
        self.e = e
        self.W = np.zeros((N * M + 1, K_path, d))
        self.K_path = K_path

    def GBM(self):
        """
        output:
        s: stock price (NM+1, K_path, d)
        update W   (NM+1, K_path, d)  W[0,:,:] = zeros # attention here

        """
        dt = self.T / self.N
        N = self.N
        sigma = self.sigma
        K_path = self.K_path
        d = self.d
        r = self.r

        delta = self.delta
        S0 = self.S0

        So_vec = S0 * np.ones((1, K_path, d))
        Z = np.random.standard_normal((N, K_path, d))
        s = S0 * np.exp(np.cumsum((r - delta - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * Z, axis=0))
        W = np.concatenate((np.zeros((1, K_path, d)), np.sqrt(dt) * Z), axis=0)  # check dimsensions
        s = np.append(So_vec, s, axis=0)
        self.W = W
        return s

    def g(self, n, k_path, X):
        max1 = torch.max(X[int(n), k_path, :].float() - self.K)
        return np.exp(-self.r * (self.T / self.N) * n) * torch.max(max1, torch.tensor([0.0]))

    def map_to_financial_instruments(self):
        """
        W:Brownian motion that represents the random fluctuations associated with financial securities
        W.shape = (N, K_path, d)

        output:
        p_function: R^d -> R^e, here P represents mapped results
        P.shape = (NM + 1, K_path, e)
        """
        W = self.W
        S_0 = self.S0
        sigma = self.sigma
        N = self.N
        K_path = self.K_path
        e = self.e
        T = NM

        p = np.zeros((N + 1, K_path, e))
        for m in range(N + 1):
            for k in range(K_path):
                p[m, k, :] = S_0 * np.exp(sigma * W[m, k, :] - sigma ** 2 * (m * T / N) / 2)

        return p

        """
        discounted_value = self.calculate_discounted_value()

        pca = PCA(n_components=self.e)
        mapped_values = np.zeros((self.N*self.M + 1, self.K_path, self.e))
        for i in range(self.N*self.M + 1):
            pca.fit(discounted_value[i])
            mapped_values[i] = pca.transform(discounted_value[i])

        return mapped_values

    def calculate_discounted_value(self):
        dt = self.T / self.N
        discount_factor = np.exp(-self.r * dt)
        discounted_value = np.zeros((self.N* *self.M + 1, self.K_path, self.d))
        s = self.GBM()
        for i in range(self.N*self.M + 1):
            discounted_value[i] = s[i] * discount_factor ** i
        return discounted_value
        """
