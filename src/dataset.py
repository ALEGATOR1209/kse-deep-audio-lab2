from torch.utils.data import Dataset, Sampler
from torch.nn.utils.rnn import pad_sequence
from pathlib import Path
from math import floor, ceil
import pandas as pd
import librosa
import torch
import random

RTTM_COLUMNS = [
    "type", "file_id", "channel", "start", "duration",
    "ortho", "speaker_type", "speaker", "conf", "lookahead",
]

class VoxConverseDataset(Dataset):
  def __init__(
      self,
      annotations_dir: Path,
      audio_dir: Path,
      sample_rate: int,
      normalize_audio=True,
      audio_transforms=None,
):
    self.annotations_dir = annotations_dir
    self.audio_dir = audio_dir
    self.sr = sample_rate

    self.annotation_files = [f for f in sorted(annotations_dir.iterdir())]
    self.audio_files = [f for f in sorted(audio_dir.iterdir())]

    self.normalize_audio = normalize_audio
    self.audio_transforms = audio_transforms

    self._match_files()

  def __len__(self):
    return len(self.annotation_files)

  def __getitem__(self, index: int):
    assert index in range(0, len(self))

    annfile = self.annotation_files[index]
    df = self._load_annotations(annfile)
    waveform = self._load_waveform(self.annotation_to_audio[annfile])

    return df, waveform

  def _load_annotations(self, file: Path):
    df = pd.read_csv(
      file,
      sep=r'\s+',
      header=None,
      names=RTTM_COLUMNS,
    )

    return list(df[["speaker", "start", "duration"]].itertuples(index=False, name="Turn"))

  def _load_waveform(self, file: Path):
    y, sr = librosa.load(file, sr=self.sr)
    assert sr == self.sr
    assert len(y.shape) == 1

    if self.audio_transforms is not None:
      y = self.audio_transforms(samples=y, sample_rate=sr)

    if self.normalize_audio:
      y = librosa.util.normalize(y)

    return y

  def _match_files(self):
    annotation_names = set([f.stem for f in self.annotation_files])
    audio_names = set([f.stem for f in self.audio_files])

    assert annotation_names == audio_names

    audio_by_stem = {f.stem: f for f in self.audio_files}

    self.annotation_to_audio = {
      ann: audio_by_stem[ann.stem]
      for ann in self.annotation_files
    }

def make_collate_fn(hop: int, sample_rate: int):
  hop_samples = int(hop * sample_rate / 1000)
  hop_seconds = hop / 1000

  def collate_fn(batch):
    waveforms = [torch.as_tensor(w, dtype=torch.float32) for _, w in batch]
    wav_lengths = [len(w) for w in waveforms]
    waveforms = pad_sequence(waveforms, batch_first=True)

    max_size = waveforms.shape[-1]
    n_frames = ceil(max_size / hop_samples)

    labels = torch.zeros(len(batch), n_frames)
    mask = torch.zeros(len(batch), n_frames, dtype=torch.bool)

    for i, ((turns, _), wl) in enumerate(zip(batch, wav_lengths)):
      # calculate number of unpadded samples
      n_valid = ceil(wl / hop_samples)

      # set unpadded samples as true, paddings are false
      mask[i, :n_valid] = True

      for t in turns:
        # calculate start sample of the speech turn
        start = floor(t.start / hop_seconds)
        start = max(0, start)

        # calculate end sample of the speech turn
        end = t.start + t.duration
        end = floor(end / hop_seconds)
        end = min(n_valid, end)

        # label samples between start and end as speech.
        # non-speech samples would remain 0
        labels[i, start:end] = 1.0

    return waveforms, labels, mask

  return collate_fn

class AudioSampler(Sampler):
    def __init__(self, lengths, batch_size, shuffle=True):
        self.lengths = lengths
        self.batch_size = batch_size
        self.shuffle = shuffle

    def __iter__(self):
        idx = sorted(range(len(self.lengths)), key=lambda i: self.lengths[i])
        batches = [idx[i:i + self.batch_size] for i in range(0, len(idx), self.batch_size)]

        if self.shuffle:
            random.shuffle(batches)

        yield from batches

    def __len__(self):
        return (len(self.lengths) + self.batch_size - 1) // self.batch_size