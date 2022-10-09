## MT3: Multi-Task Multitrack Music Transcription - Pytorch

This is an unofficial implementation of MT3: Multi-Task Multitrack Music Transcription in pytorch.

## Usage

### Run MT3 in Docker

Build docker image:
```bash
bash build_mt3_docker.sh
```

Run MT3 transcription:
```bash
docker run --rm -it --gpus all -v "${PWD}/input:/input" -v "${PWD}/output:/output" mt3
```

### Run MT3 in Python

```python
from inference import InferenceHandler

handler = InferenceHandler('./pretrained')
handler.inference('music.mp3')
```

```python
# training not done yet
```

## Citations

```bibtex
@article{gardner2021mt3,
  title={MT3: Multi-Task Multitrack Music Transcription},
  author={Gardner, Josh and Simon, Ian and Manilow, Ethan and Hawthorne, Curtis and Engel, Jesse},
  journal={arXiv preprint arXiv:2111.03017},
  year={2021}
}
```