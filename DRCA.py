import torch
from torch import nn
from LinearAE import *
from clusteringLayer import *
from deepReprLayer import *
import numpy as np
import math
import os
from sklearn.cluster import KMeans

class DRCA(nn.Module):

    # Deep Representation Clutsering Accelerameter

    def __init__(self,
                 input_dim: "input dim equals sequence length",
                 embedding_dim: "latent space dim" =10,
                 n_clusters: "number of initiate clusters" = 10,
                 gamma: "latent loss weights" = .01,
                 alpha: "defined variance on calculating clustering metric default 1" = 1.,
                 pretrained_model_state=None,
                 clustering_train_data=None,
                 clustering_layer="DEC"):

        super(DRCA, self).__init__()

        self.alpha = alpha
        self.input_dim = input_dim
        self.embedding_dim = embedding_dim
        self.gamma = gamma
        self.n_clusters = n_clusters
        self.clustering_train = clustering_train_data
        self.pretrained_model = pretrained_model_state
        self.clustering_layer = clustering_layer
        # self.save_dist = []

        # **************** build model structure ******** #

        RL = RepresentationLayer(self.alpha, self.embedding_dim)
        CL = ClusteringLayer(self.n_clusters, self.embedding_dim, self.alpha)
        AE = LinearAE(self.input_dim, self.embedding_dim)
        if (self.clustering_layer == "DEC"):
            self.model = nn.ModuleDict({"encoder": AE.encoder,
                                        "clustering": CL,
                                        "decoder": AE.decoder})
        else if (self.clustering_layer == "DeepRepr"):
            self.model = nn.ModuleDict({"encoder": AE.encoder,
                                        "clustering": RL,
                                        "decoder": AE.decoder})
        if self.pretrained():
            self.load_model()

    def pretrained(self):
        return self.pretrained_model is not None

    def load_model(self):

        print("=" * 89)
        print(f"loading pretrianed model from {self.pretrained_model}")

        with open(self.pretrained_model, "rb") as f:
            self.model.load_state_dict(torch.load(self.pretrained_model))

        if (self.clustering_layer == "DEC"):
            self.init_centroids()

    def init_centroids(self):

        assert not self.model["clustering"].initiated

        print("initiate centroids with KMeans...")

        km = KMeans(n_clusters=self.n_clusters, n_init=20)

        z = self.model["encoder"](self.clustering_train)
        z = z.detach().numpy()

        km.fit(z)

        centroids = torch.from_numpy(km.cluster_centers_)

        self.model["clustering"].centroids.data = centroids

        self.model["clustering"].initiated = True

        print("done!")
        print("=" * 89)

    def pair_dist_mat(self, x, y):
        x = x.unsqueeze(1)
        z = torch.pow(x - y, 2).sum(-1)
        return z

    def forward(self, input):

        rec_crit = torch.nn.MSELoss()
        kld_crit = torch.nn.KLDivLoss()

        if not self.pretrained():

            x = self.model["encoder"](input)
            x_hat = self.model["decoder"](x)
            loss = rec_crit(x_hat, input)

            return (loss,)

        else:

            if not (self.model["clustering"].initiated):
                self.load_model()

            z = self.model["encoder"](input)
            p, q = self.model["clustering"](z)
            x_hat = self.model["decoder"](z)

            lr = rec_crit(input, x_hat)

            if (self.clustering_layer == "DEC"):
                lc = kld_crit(input=torch.log(p), target=q)
            else if (self.clustering_layer == "DeepRepr"):
                lc = kld_crit(input=torch.log(q), target=p) - \
                    torch.sum(p, torch.log(p))
            loss = lr + self.gamma * lc

            return (loss, lr, lc)

            # class CrossEntropyLoss(nn.Module):

            #     def __init__(self):
            #         super(CrossEntropyLoss, self).__init__()

            #     def forward(self, p, q):
            #         return torch.sum(-1. * torch.mul(p, torch.log(q)))

            # class EntropyLoss(nn.Module):
            #     """docstring for EntropyLoss"""

            #     def __init__(self):
            #         super(EntropyLoss, self).__init__()

            #     def forward(self, p):
            #         return torch.sum(torch.mul(p, torch.log(p)))
