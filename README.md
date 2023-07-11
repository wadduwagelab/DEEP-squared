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

Use the following code to train the model for a particular dataset case and loss function.

```python
python3 run.py --case <CASE> --lossfunc <LOSS_FUNCTION> --experiment_name <EXPERIMENT_NAME> --epochs <#EPOCHS>
```

### Evaluation

Use the following code to evaluate the performance of the pre-trained model for any dataset case.

```python
python3 validation.py --case <CASE> --model_path <MODEL_PATH> --output_path <OUTPUT_PATH>
```
### Sample output
The quantitative and qualitative output for the **Beads 4SLS** dataset case follows.

*Evaluation metric : mean value for the entire test dataset (standard deviation)* 

**MSE error** : 0.00013 (5.8144e-05)

**SSIM** : 0.7874(0.1295)

**PSNR** : 39.1606 (1.8238)

![20_prediction](https://github.com/Mithunjha/DEEP2/assets/67052077/caccf1b1-277f-4943-b22b-7ec424c20993)



