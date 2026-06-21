# Option 2. Speaker diarization

## Pyannote-Based

env file - `pyannote-environment.yml`  
notebook - `lab2-pyannote.ipynb`

most of the workflow is based on the [tutorial](https://github.com/pyannote/pyannote-audio/blob/develop/tutorials/training_a_model.ipynb)

### Problems
tutorial's DDEG was using float16 data type that caused overflow on the inference stage, custom impl was suggested by claude to tackle this problem

### Results

| Model                 | DER    |
| --------------------- | ------ |
| pretrained            | 11.1%  |
| finetuned (1 epoch)   | 10.8%  |

