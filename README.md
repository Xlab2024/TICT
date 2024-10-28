
<!-- 关于本项目 -->
# TICT
<img src="images/TICT.png" alt="MvDeFormer-logo" width="833" height="475">

## Environment
### Setup
Clone the repository and navigate into the directory:
```
$ git clone https://github.com/Xlab2024/TICT.git
$ cd TICT
```

### Install requirements
Create a conda environment, activate it, and install the required packages:
```
$ conda env create -f environment.yaml
```
## Download
### Dataset
- We are currently utilizing the <a href="https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3041807/">LIDC-IDRI</a> dataset. You can download the original dataset <a href="https://wiki.cancerimagingarchive.net/pages/viewpage.action?pageId=1966254">here</a>. 
Follow the instructions in [prepare_datasets.md](data_preprocessing/prepare_datasets.md) to preprocess the dataset.

### Pretrained Model
- We've provided the pretrained weights used for our paper's experiments. You can download these weights <a href="https://drive.google.com/drive/folders/14_V8E0XklPRax4S4nQfeUnHS2gml-_PO">here</a>.  <br /><br />

## Training
To train PerX2CT, run `main.py` with the hyper-parameters provided below:
```
python main.py --train True --gpus <gpu_ids> --name <exp_name> --base <path_to_base_config>
```
**Note**: The hyperparameters used in our paper's experiments are set as default. <br />
**Note**: The configuration for training the PerX2CT model without zoom-in is `configs/PerX2CT.yaml`, while the configuration for training the PerX2CT<sub>global</sub> model with zoom-in is `configs/PerX2CT_global_w_zoomin.yaml`. <br /><br />

## Citation
If you use this code for your research, please cite our paper.

```bibtex
@ARTICLE{10533689,
  author={Jin, Can and Meng, Xiangzhu and Li, Xuanheng and Wang, Jie and Pan, Miao and Fang, Yuguang},
  journal={IEEE Transactions on Mobile Computing}, 
  title={Rodar: Robust Gesture Recognition Based on mmWave Radar Under Human Activity Interference}, 
  year={2024},
  volume={},
  number={},
  pages={1-14},
  keywords={Interference;Gesture recognition;Feature extraction;Radar;Sensors;Doppler effect;Data mining;Wireless Sensing;Millimeter Wave Radar;Gesture Recognition;Human Activity Interference;Deep Learning},
  doi={10.1109/TMC.2024.3402356}}

<p align="right">(<a href="#top">Reture to top</a>)</p>
