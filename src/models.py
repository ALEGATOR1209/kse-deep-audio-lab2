import torch
import torchaudio.transforms as T
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import warnings

warnings.filterwarnings("ignore", message="An output with one or more elements was resized")

class MelSpectrogramExtractor(nn.Module):
  def __init__(
      self,
      mel_bands: int = 32,
      sample_rate: int = 16000,
      hop_ms: float = 10.0,
      win_ms: float = 25.0,
      n_fft: int = 512,
      eps: float = 1e-6,
  ):
    super().__init__()
    self.mel_bands = mel_bands
    self.eps = eps
    self.hop_ms = hop_ms
    self.melspec = nn.Sequential(
      T.MelSpectrogram(
        sample_rate=sample_rate,
        n_fft=n_fft,
        win_length=int(sample_rate * win_ms / 1000),
        hop_length=int(sample_rate * hop_ms / 1000),
        n_mels=mel_bands,
        normalized=True,
      )
    )

  def forward(self, X):
    mel = self.melspec(X)
    mel = torch.log(mel + self.eps)
    mel = mel.transpose(1, 2)
    return mel

class VoiceActivityDetector(nn.Module):
  def __init__(
    self,
    device: str,
    spectrogram_layer: MelSpectrogramExtractor,
    subsample_size: int = 320,
  ):
    super().__init__()
    self.device = device
    self.spectrogram = spectrogram_layer
    self.subsample_size = subsample_size

    self.cnn = nn.Sequential(
      nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),
      nn.ReLU(),
      nn.MaxPool2d(2),
      nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
      nn.ReLU(),
      nn.MaxPool2d(2),
      nn.Flatten(),
      nn.Linear(2048, 64),
    )

    self.lstm = nn.LSTM(
      input_size=64,
      hidden_size=128,
      num_layers=2,
      dropout=0.5,
      bidirectional=True,
      batch_first=True,
    )

    self.head = nn.Linear(2 * 128, 1)

  def _chunk(self, mel, labels, mask):
    n_frames = min(mel.shape[1], labels.shape[1])
    mel, labels, mask = mel[:, :n_frames], labels[:, :n_frames], mask[:, :n_frames]

    chunk_size = int(self.subsample_size / self.spectrogram.hop_ms)
    padding = (chunk_size - n_frames % chunk_size) % chunk_size

    if padding:
      mel = F.pad(mel, (0, 0, 0, padding))
      labels = F.pad(labels, (0, padding))
      mask = F.pad(mask, (0, padding), value=False)

    n_chunks = mel.shape[1] // chunk_size
    mel = mel.reshape(mel.shape[0], n_chunks, chunk_size, mel.shape[2])

    labels = labels.reshape(labels.shape[0], n_chunks, chunk_size)
    labels = labels.sum(-1)
    labels = labels > chunk_size // 2

    mask = mask.reshape(mask.shape[0], n_chunks, chunk_size)
    mask = mask.any(-1)

    return mel, labels, mask

  def forward(self, X):
    (waveform, labels, mask) = X

    mel = self.spectrogram(waveform)

    (mel, labels, mask) = self._chunk(mel, labels, mask)

    B, n_chunks, H, W = mel.shape

    Y = mel.reshape(B * n_chunks, 1, H, W)
    Y = self.cnn(Y)
    Y = Y.reshape(B, n_chunks, -1)

    lengths = mask.sum(dim=1).cpu()
    packed = pack_padded_sequence(Y, lengths, batch_first=True, enforce_sorted=False)
    Y, _ = self.lstm(packed)
    Y, _ = pad_packed_sequence(Y, batch_first=True, total_length=n_chunks)
    Y = self.head(Y).squeeze(-1)

    return Y, labels, mask

class Segmentator(nn.Module):
  def __init__(
    self,
    device: str,
    spectrogram_layer: MelSpectrogramExtractor,
    window_size: int = 5000,
    window_step: int = 500,
    kmax: int = 3,
  ):
    super().__init__()
    self.device = device
    self.spectrogram = spectrogram_layer
    self.window_size = window_size
    self.window_step = window_step
    self.kmax = kmax

    self.cnn = nn.Sequential(
      nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),
      nn.ReLU(),
      nn.MaxPool2d((1, 2)),
      nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
      nn.ReLU(),
      nn.MaxPool2d((1, 2)),
    )

    self.lstm = nn.LSTM(
      input_size=256,
      hidden_size=512,
      num_layers=2,
      dropout=0.5,
      bidirectional=True,
      batch_first=True,
    )

    self.head = nn.Linear(2 * 512, kmax)

  # This method is AI generated
  def _chunk(self, mel, labels, mask):
    # mel: (B, T, M)   labels: (B, K, T)   mask: (B, T)   — all at hop_ms (10ms) resolution
    n_frames = min(mel.shape[1], labels.shape[-1])
    mel, labels, mask = mel[:, :n_frames], labels[:, :, :n_frames], mask[:, :n_frames]

    win = int(self.window_size / self.spectrogram.hop_ms)   # 5000/10 -> 500 frames (5s)
    hop = int(self.window_step / self.spectrogram.hop_ms)   #  500/10 ->  50 frames (500ms)

    # pad so a `win`-wide window stepped by `hop` covers every frame
    if n_frames <= win:
        n_windows, pad = 1, win - n_frames
    else:
        n_windows = (n_frames - win + hop - 1) // hop + 1
        pad = (n_windows - 1) * hop + win - n_frames

    if pad:
        mel    = F.pad(mel,    (0, 0, 0, pad))
        labels = F.pad(labels, (0, pad))
        mask   = F.pad(mask,   (0, pad), value=False)

    # overlapping windows via unfold (views, no copy)
    mel    = mel.unfold(1, win, hop).permute(0, 1, 3, 2)     # (B, W, win, M)
    labels = labels.unfold(2, win, hop).permute(0, 2, 3, 1)  # (B, W, K, win)
    mask   = mask.unfold(1, win, hop)                        # (B, W, win)

    labels = self._pack(labels, self.kmax)                   # (B, W, kmax, win)
    return mel, labels, mask

  # This method is AI generated
  def _pack(self, labels, kmax):
    B, W, T, K = labels.shape
    if K < kmax:
        labels = torch.cat([labels, labels.new_zeros(B, W, T, kmax - K)], dim=3)
        K = kmax
    present = labels.sum(dim=2).int()
    _, order = torch.sort(present, dim=-1, descending=True, stable=True)
    idx = order[:, :, None, :kmax].expand(-1, -1, T, -1)
    return torch.gather(labels, 3, idx)

  def forward(self, X):
    waveform, labels, mask = X

    # (Batch, Time, Mel)
    mel = self.spectrogram(waveform)

    mel, labels, mask = self._chunk(mel, labels, mask)
    B, W, T, M = mel.shape

    # # (Batch, Window, Time, Mel)
    # print("Mel =", mel.shape)

    # # (Batch, Window, Speakers, Time)
    # print("Labels = ", labels.shape)

    # # (Batch, Window, Mask)
    # print("Mask = ", mask.shape)

    Y = mel.reshape(B * W, 1, T, M)
    Y = self.cnn(Y)
    Y = Y.permute(0, 2, 1, 3)
    Y = Y.reshape(B * W, T, -1)

    lengths = mask.reshape(B * W, T).sum(dim=1).clamp(min=1).cpu()
    Y = pack_padded_sequence(Y, lengths, batch_first=True, enforce_sorted=False)
    Y, _ = self.lstm(Y)
    Y, _ = pad_packed_sequence(Y, batch_first=True, total_length=T)

    Y = self.head(Y)
    Y = Y.reshape(B, W, T, self.kmax)

    return Y, labels, mask
