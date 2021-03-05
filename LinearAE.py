import torch
from torch import nn
import torch.nn.functional as F


class Encoder(nn.Module):

    def __init__(self, input_dim, embedding_dim):

        super(Encoder, self).__init__()

        self.INPUT_DIM = input_dim
        self.EMBED_DIM = embedding_dim

        self.encode = nn.Sequential(
            nn.Linear(input_dim, 250),
            nn.ReLU(),
            nn.Linear(250, 500),
            nn.ReLU(),
            nn.Linear(500, 500),
            nn.ReLU(),
            nn.Linear(500, 250),
            nn.ReLU(),
            nn.Linear(250, embedding_dim)
        )

    def forward(self, input):

        # input shape: (batch, seq_len)

        out = self.encode(input)

        return out

class Decoder(nn.Module):

    def __init__(self, output_dim, embedding_dim):

        super(Decoder, self).__init__()

        self.OUTPUT_DIM = output_dim
        self.EMBED_DIM = embedding_dim

        self.decode = nn.Sequential(
            nn.Linear(embedding_dim, 250),
            nn.ReLU(),
            nn.Linear(250, 500),
            nn.ReLU(),
            nn.Linear(500, 500),
            nn.ReLU(),
            nn.Linear(500, 250),
            nn.ReLU(),
            nn.Linear(250, output_dim)
        )

    def forward(self, embedding):

        out = self.decode(embedding)
        return out

class LinearAE(nn.Module):

    def __init__(self, input_dim, embedding_dim):
        super(LinearAE, self).__init__()

        self.INPUT_DIM = input_dim
        self.EMBED_DIM = embedding_dim

        self.encoder = Encoder(input_dim, embedding_dim)
        self.decoder = Decoder(input_dim, embedding_dim)

    def forward(self, input):

        embedding = self.encoder(input)
        output = self.decoder(embedding)

        return embedding, output
