import torch
import torch.nn as nn
import torch.nn.functional as F

class TDNN(nn.Module):
    def __init__(self, in_dim, out_dim, k, d=1):
        super().__init__()
        self.conv = nn.Conv1d(in_dim, out_dim, kernel_size=k, dilation=d)
        self.bn = nn.BatchNorm1d(out_dim)

    def forward(self, x):
        return torch.relu(self.bn(self.conv(x)))


class StatsPool(nn.Module):
    def forward(self, x):
        mean = x.mean(dim=-1)
        std = x.std(dim=-1)
        return torch.cat([mean, std], dim=-1)


class XVector(nn.Module):
    def __init__(self, device, spectrogram_layer, emb_dim=512, n_speakers=100):
        super().__init__()
        self.device = device
        self.spectrogram = spectrogram_layer
        n_mels = spectrogram_layer.mel_bands
        
        self.frame = nn.Sequential(
            TDNN(n_mels, 512, k=5, d=1),
            TDNN(512, 512, k=3, d=2),
            TDNN(512, 512, k=3, d=3),
            TDNN(512, 512, k=1, d=1),
            TDNN(512, 1500, k=1, d=1),
        )
        self.pool = StatsPool()
        self.seg1 = nn.Linear(3000, emb_dim)
        self.seg_bn = nn.BatchNorm1d(emb_dim)
        self.seg2 = nn.Linear(emb_dim, emb_dim)
        self.head = nn.Linear(emb_dim, n_speakers)

    def forward(self, wav, return_embedding=False):
        feats = self.spectrogram(wav).transpose(1, 2)
        h = self.frame(feats)
        h = self.pool(h)
        emb = self.seg1(h)
        if return_embedding:
            return F.normalize(emb, dim=1)
        h = self.seg2(self.seg_bn(torch.relu(emb)))
        return self.head(h)