import torch
from torch import nn

class RepresentationLayer(nn.Module):

  def __init__(self, alpha, embedding_dim):

    super(RepresentationLayer, self).__init__()

    self.alpha
    self.embedding_dim = embedding_dim
    self.initiated = False

    self.deep_repr = nn.Sequential(nn.Linear(embedding_dim, 250),
                                   nn.ReLU(),
                                   nn.Linear(250, 500),
                                   nn.ReLU(),
                                   nn.Linear(500, 250),
                                   nn.ReLU(),
                                   nn.Linear(250, embedding_dim))

  def pair_dist_mat(self, x, y):

        # input x (M, L) y (N, L) output (M, N)

        x = x.unsqueeze(1)
        z = torch.pow(x - y, 2).sum(-1)
        return z

    def soft_dist(self, z, c, alpha = self.alpha):

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

    z_ = self.deep_repr(z)

    p = self.soft_dist(z,z)
    q = self.soft_dist(z_, z_)

    return p, q
