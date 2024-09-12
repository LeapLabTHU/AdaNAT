import torch
import torch.nn as nn
import math

from torch.nn import functional as F


#################################################################################
#               Embedding Layers for Timesteps and Class Labels                 #
#################################################################################

class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """

    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


class AdaLN(nn.Module):
    def __init__(self, x_dim, c_dim):
        super().__init__()
        self.norm = nn.LayerNorm(x_dim, elementwise_affine=False, eps=1e-6)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(c_dim, 2 * x_dim, bias=True)
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = self.norm(x)
        if x.ndim == 3:  # B, T, D
            scale, shift = scale.unsqueeze(1), shift.unsqueeze(1)
        x = x * (1 + scale) + shift
        return x


class PolicyNet(nn.Module):
    def __init__(self, in_channels=512, out_channels=3, hidden_size=1152):
        super().__init__()

        if in_channels != hidden_size:
            self.x_embedder = nn.Linear(in_channels, hidden_size)  # for feat input
        else:
            self.x_embedder = None
        self.t_embedder = TimestepEmbedder(hidden_size)

        self.adaLN1 = AdaLN(x_dim=hidden_size, c_dim=hidden_size)
        self.dw_conv = nn.Conv2d(hidden_size, hidden_size, kernel_size=3, stride=2, padding=1, groups=hidden_size)
        self.pw_conv = nn.Conv2d(hidden_size, 128, kernel_size=1, stride=1, padding=0)

        self.linear1 = nn.Linear(64*128, 1024)
        self.adaLN2 = AdaLN(x_dim=1024, c_dim=hidden_size)

        self.linear2 = nn.Linear(1024, 512)
        self.adaLN3 = AdaLN(x_dim=512, c_dim=hidden_size)

        self.linear3 = nn.Linear(512, out_channels)
        self.initialize_weights()

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # # Initialize timestep embedding MLP:
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        def _adaLN_init(module):
            if isinstance(module, AdaLN):
                nn.init.constant_(module.adaLN_modulation[-1].weight, 0)
                nn.init.constant_(module.adaLN_modulation[-1].bias, 0)
        self.apply(_adaLN_init)

        nn.init.constant_(self.linear3.weight, 0)
        nn.init.constant_(self.linear3.bias, 0)

    def forward(self, state):

        x, t = state['feat'], state['timestep']
        if self.x_embedder is not None:
            x = self.x_embedder(x)

        c = self.t_embedder(t) if t is not None else 0

        x = self.adaLN1(x, c)
        x = x.permute(0, 2, 1).reshape(x.size(0), -1, 16, 16).contiguous()

        x = self.dw_conv(x)
        x = F.gelu(x)

        x = self.pw_conv(x)
        x = F.gelu(x)

        x = x.view(x.size(0), -1)

        x = self.linear1(x)
        x = self.adaLN2(x, c)
        x = F.gelu(x)

        x = self.linear2(x)
        x = self.adaLN3(x, c)
        x = F.gelu(x)

        x = self.linear3(x)
        return x


