
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
Create a conda environment with the file provided:
```
$ conda env create -f environment.yaml
```
## Download
### Dataset
- Convert the CT slices in your own dataset into 256x256 png images and store them in “./data/CT/<dateset>”, then write the file names into train.txt and val.txt.

### Pretrained Model
- For the pretrained weights VQ-F4 using in autoencoder. You can download from <a href="https://ommer-lab.com/files/latent-diffusion/vq-f4.zip">here</a>.
- For the pretrained weights LDM-VQ-4 using in noise estimation network. You can download from <a href="https://ommer-lab.com/files/latent-diffusion/sr_bsr.zip">here</a>.
## Training
To train TICT, run `main.py` with the hyper-parameters provided below:
```
python main.py --base configs/latent-diffusion/config.yaml --gpus 1
```
During the training process, the generated results on the test set will be automatically evaluated and saved at regular intervals.

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
