# DEEP-squared: Deep Learning Powered De-scattering with Excitation Patterning

Official implementation of "DEEP-squared: deep learning powered De-scattering with Excitation Patterning" (Light: Science & Applications, 2023).

## Citation
If you find our work or this repository useful, please consider giving a star ⭐ and a citation.

```
@article{deepsquared2023,
  title={DEEP-squared: deep learning powered De-scattering with Excitation Patterning},
  author={Navodini Wijethilake and Mithunjha Anandakumar and Cheng Zheng and Peter T. C. So and Murat Yildirim and Dushan N. Wadduwage},
  journal={Light: Science & Applications},
  volume={12},
  issue={1},
  pages={228},
  year={2023},
  doi = {https://doi.org/10.1038/s41377-023-01248-6}
}
```

This repository contains the implementation of the physics-informed forward model, which generates simulated DEEP image stacks, and the DEEP-squared inverse model, which reconstructs de-scattered images from 32 patterned DEEP-TFM measurements.

![Figure1_method](https://github.com/Mithunjha/DEEP-squared/assets/67052077/d0deba89-53f6-48b5-b766-ec86b3867423)

## Dataset
Find the dataset used in our work at : [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.8161051.svg)](https://doi.org/10.5281/zenodo.8161051)


## Getting Started
### Installation
The deep learning algorithms were developed in Pytorch Environment : [https://pytorch.org/](https://pytorch.org/) and the forward model was implemented in MatLab.

```python
pip install torch==1.10.0+cu113 torchvision==0.11.1+cu113 torchaudio===0.10.0+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html
```

Run the below code to install all other dependencies.

```python
pip install -r requirements.txt
```

### Training

Use the following code to train the model for a particular dataset case and loss function.

```python
python3 run.py --case <CASE> --save_model_path <PATH_TO_A_FOLDER> --lossfunc <LOSS_FUNCTION> --experiment_name <EXPERIMENT_NAME> --epochs <#EPOCHS>
```

### Evaluation

Use the following code to evaluate the performance of the pre-trained model for any dataset case.

```python
python3 validation.py --case <CASE> --model_path <MODEL_PATH> --output_path <OUTPUT_PATH>
```
### Sample output
The quantitative and qualitative output for the **Beads 4 Scattering Lengths** dataset case follows.

*Evaluation metric : mean value for the entire test dataset (standard deviation)* 

**MSE error** : 0.00013 (5.8144e-05)

**SSIM** : 0.7874(0.1295)

**PSNR** : 39.1606 (1.8238)

![20_prediction](https://github.com/Mithunjha/DEEP-squared/assets/67052077/df73007b-56d1-45c5-9285-46fad918781d)


