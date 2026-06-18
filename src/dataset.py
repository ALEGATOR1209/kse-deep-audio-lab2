from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from pathlib import Path
import pandas as pd
import librosa
import torch

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
