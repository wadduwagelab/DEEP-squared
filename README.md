# DEEP-squared: Deep Learning Powered De-scattering with Excitation Patterning

## Citation
If you find our work or this repository useful, please consider giving a star ⭐ and a citation.

```
@misc{wijethilake2022deep2,
      title={DEEP-squared: Deep Learning Powered De-scattering with Excitation Patterning}, 
      author={Navodini Wijethilake and Mithunjha Anandakumar and Cheng Zheng and Peter T. C. So and Murat Yildirim and Dushan N. Wadduwage},
      year={2022},
      eprint={2210.10892},
      archivePrefix={arXiv},
      primaryClass={eess.IV}
}
```

This repository contains the implementation of the physics-informed forward model, which generates simulated DEEP image stacks, and the DEEP-squared inverse model, which reconstructs de-scattered images from 32 patterned DEEP-TFM measurements.

![Figure1_method](https://github.com/Mithunjha/DEEP2/assets/67052077/1bf527a2-c3fa-4fb9-97d5-a901eb3d1427)

## Getting Started
### Installation
The algorithms were developed in Pytorch Environment : [https://pytorch.org/](https://pytorch.org/)

```python
pip install torch==1.10.0+cu113 torchvision==0.11.1+cu113 torchaudio===0.10.0+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html
```

Run the below code to install all other dependencies.

```python
pip install -r requirements.txt
```

### Training

```python
python3 run.py <CASE> <DATA_PATH> <>
```

### Evaluation

```python
python3 validation.py --case <CASE> --model_path <MODEL_PATH> --

