import torch
from torch import nn

class ClusteringLayer(nn.Module):

    def __init__(self, n_clusters, embedding_dim, alpha):
        super(ClusteringLayer, self).__init__()
        self.alpha = alpha
        self.centroids = nn.Parameter(torch.empty(
            n_clusters, embedding_dim), requires_grad=True)  # placeholder
        self.initiated = False

    def pair_dist_mat(self, x, y):

        # input x (M, L) y (N, L) output (M, N)

        x = x.unsqueeze(1)
        z = torch.pow(x - y, 2).sum(-1)
        return z

    def soft_dist(self, z, c, alpha):

        z_pair_dist = self.pair_dist_mat(z, c)
        p = torch.pow(1. + z_pair_dist / alpha, -(alpha + 1.) / 2.)
        p = p / torch.sum(p, dim=1, keepdim=True)
        return p

    def target_dist(self, p):

        # p (M,N)

        q = torch.pow(p, 2.) / torch.sum(p, dim=0, keepdim=True)
        q = p / torch.sum(p, dim=1, keepdim=True)
        return q

    def forward(self, z):

        p = self.soft_dist(z, self.centroids, self.alpha)
        q = self.target_dist(p)

        p = torch.clamp(p, min=1e-10)
        q = torch.clamp(q, min=1e-10)

        return p, q
