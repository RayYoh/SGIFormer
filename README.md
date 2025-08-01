<div align="center">

# SGIFormer: Semantic-guided and Geometric-enhanced Interleaving Transformer for 3D Instance Segmentation

[![python](https://img.shields.io/badge/-Python_3.8-blue?logo=python&logoColor=white)](https://github.com/pre-commit/pre-commit)
[![license](https://img.shields.io/badge/License-MIT-green.svg?labelColor=gray)](LICENSE)

[**Paper**](https://arxiv.org/abs/2407.11564) | [**Project Page**](https://rayyoh.github.io/SGIFormer/) | [**Weights**](https://huggingface.co/RayYoh/SGIFormer)

If you have any questions, please consider open an issue. I will help you in any way I can.

</div>

<p align="center">
    <img src="docs/overview.png" alt="overview" width="90%" />
</p>

This is the official implementation of paper "SGIFormer: Semantic-guided and Geometric-enhanced Interleaving Transformer for 3D Instance Segmentation".

<b>Update (27/05/2025):</b> Our new method, [LaSSM](https://github.com/RayYoh/LaSSM), achieves better performance on ScanNet++ V2, and SGIFormer won second place at the [CVPR 2025 ScanNet++ challenge](https://kaldir.vc.in.tum.de/scannetpp/cvpr2025).

<details>
<summary><b>Update: SGIFormer achieves the SOTA performance on ScanNet++ V2 test set for 3D instance segmentation. Test scores accessed on 22 Jan 2025. <a href="https://kaldir.vc.in.tum.de/scannetpp/benchmark/insseg" target="_blank">ScanNet++ V2 test set</a></b> (The results are provided by official based on this repo)</summary>

![image](./docs/leadboard_v2.png)
</details>

<details>
<summary><b> SGIFormer achieves the SOTA performance on ScanNet++ test set for 3D instance segmentation benchmark. Test scores accessed on 24 June 2024. </b></summary>

![image](./docs/leadboard.png)
</details>

## :memo: To-Do List
- [x] Release trained weights and experiment record.
- [x] Installation instructions.
- [x] Processing datasets.
- [x] Release training configs.
- [x] Release training code.



## :floppy_disk: Trained Results
| Model | Benchmark | Num GPUs | mAP | AP50 | Config | Tensorboard | Exp Record | Model |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| SGIFormer | ScanNet++ V2 Test | - | 29.9 | 45.7 | - | - | - | - |
| SGIFormer | ScanNet++ Val | 4 | 23.9 | 37.5 | [Link](https://github.com/RayYoh/SGIFormer/blob/main/configs/scannetpp/insseg-sgiformer-spunet-v2.py) | [Link](https://huggingface.co/RayYoh/SGIFormer/tensorboard) | [Link](https://huggingface.co/RayYoh/SGIFormer/raw/main/insseg-scannetpp-sgiformer-spunet/train.log)  | [Link](https://huggingface.co/RayYoh/SGIFormer/blob/main/insseg-scannetpp-sgiformer-spunet/model/model_best.pth)  |
| SGIFormer | ScanNet Val | 4 | 58.9 | 78.4  | [Link](https://github.com/RayYoh/SGIFormer/blob/main/configs/scannet/insseg-sgiformer-spunet-v2.py) | - | [Link](https://huggingface.co/RayYoh/SGIFormer/raw/main/insseg-scannet-sgiformer-spunet/train.log) | [Link](https://huggingface.co/RayYoh/SGIFormer/blob/main/insseg-scannet-sgiformer-spunet/model/model_best.pth) |
| SGIFormer-L | ScanNet Val | 4 | 61.0 | 81.2 | [Link](https://github.com/RayYoh/SGIFormer/blob/main/configs/scannet/insseg-sgiformer-l-spunet-v2.py) | - | [Link](https://huggingface.co/RayYoh/SGIFormer/raw/main/insseg-scannet-sgiformer-l-spunet/train.log) | [Link](https://huggingface.co/RayYoh/SGIFormer/blob/main/insseg-scannet-sgiformer-l-spunet/model/model_best.pth) |
| SGIFormer | ScanNet200 Val | 4 | 28.9 | 38.6  | [Link](https://github.com/RayYoh/SGIFormer/blob/main/configs/scannet200/insseg-sgiformer-spunet-v2.py) | - | [Link](https://huggingface.co/RayYoh/SGIFormer/raw/main/insseg-scannet200-sgiformer-spunet/train.log) | [Link](https://huggingface.co/RayYoh/SGIFormer/blob/main/insseg-scannet200-sgiformer-spunet/model/model_best.pth) |
| SGIFormer-L | ScanNet200 Val | 4 | 29.2 | 39.4 | [Link](https://github.com/RayYoh/SGIFormer/blob/main/configs/scannet200/insseg-sgiformer-l-minkunet.py)  | - | [Link](https://huggingface.co/RayYoh/SGIFormer/raw/main/insseg-scannet-sgiformer-l-spunet/train.log) | [Link](https://huggingface.co/RayYoh/SGIFormer/blob/main/insseg-scannet-sgiformer-l-spunet/model/model_best.pth) |


## :hammer: Installation
Our model is built on Pointcept toolkit, you can follow its [official instructions](https://github.com/Pointcept/Pointcept?tab=readme-ov-file#installation) to install the packages:

```bash
conda create -n 3dseg python=3.8 -y
conda activate 3dseg

pip install ninja==1.11.1.1

# Choose version you want here: https://pytorch.org/get-started/previous-versions/
# CUDA 11.7
pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117

pip install h5py pyyaml
pip install sharedarray tensorboard tensorboardx yapf addict einops scipy plyfile termcolor timm

# Note: (recommended) if you encounter problem in this step, try to download packages on official web and install locally
conda install pytorch-cluster pytorch-scatter pytorch-sparse -c pyg -y
pip install torch-geometric

# spconv (SparseUNet)
# refer https://github.com/traveller59/spconv
pip install spconv-cu117
```
Note that they also provide scripts to build correponding docker image: [build_image.sh](https://github.com/Pointcept/Pointcept/blob/main/scripts/build_image.sh)


## :mag: Data Preprocessing 
**ScanNet V2 & ScanNet200**
- Download the [ScanNet V2](http://www.scan-net.org/) dataset.
- Run preprocessing code for raw ScanNet as follows:

```bash
# RAW_SCANNET_DIR: the directory of downloaded ScanNet raw dataset.
# PROCESSED_SCANNET_DIR: the directory of the processed ScanNet dataset (output dir).
# Comparing with Pointcept, we just add superpoint extraction
python pointcept/datasets/preprocessing/scannet/preprocess_scannet.py --dataset_root ${RAW_SCANNET_DIR} --output_root ${PROCESSED_SCANNET_DIR}
```
- Link processed dataset to codebase:
```bash
# PROCESSED_SCANNET_DIR: the directory of the processed ScanNet dataset.
mkdir data
ln -s ${PROCESSED_SCANNET_DIR} ${CODEBASE_DIR}/data/scannet
```

**ScanNet++**
- Download the [ScanNet++](https://kaldir.vc.in.tum.de/scannetpp/) dataset.
- Run preprocessing code for raw ScanNet++ as follows:
```bash
# RAW_SCANNETPP_DIR: the directory of downloaded ScanNet++ raw dataset.
# PROCESSED_SCANNETPP_DIR: the directory of the processed ScanNet++ dataset (output dir).
# NUM_WORKERS: the number of workers for parallel preprocessing.
python pointcept/datasets/preprocessing/scannetpp/preprocess_scannetpp.py --dataset_root ${RAW_SCANNETPP_DIR} --output_root ${PROCESSED_SCANNETPP_DIR} --num_workers ${NUM_WORKERS}
```
- **Note:** We find that the preprocessing code of Pointcept yields higher scores compared to the official code of ScanNet++. However, when testing the results using the official code, the performance drops. The reason behind this is the difference in the processed validation data between Pointcept and the official code as discussed in [#279](https://github.com/Pointcept/Pointcept/issues/279). Despite the performance difference, it has been observed that Pointcept's processed data allows for faster training compared to the official code. Therefore, a suggested solution is to use Pointcept's data for training and the [official data](https://github.com/scannetpp/scannetpp) (specifically vtx_ instead of sampled_) for validation. After preprocessing, you can transfer the data into Pointcept format as follows:
```bash
python pointcept/custom/transfer_scannetpp_vtx_to_pcept.py
```
- Sampling and chunking large point cloud data in train/val split as follows (only used for training):
```bash
# PROCESSED_SCANNETPP_DIR: the directory of the processed ScanNet++ dataset (output dir).
# NUM_WORKERS: the number of workers for parallel preprocessing.
python pointcept/datasets/preprocessing/sampling_chunking_data.py --dataset_root ${PROCESSED_SCANNETPP_DIR} --grid_size 0.01 --chunk_range 6 6 --chunk_stride 3 3 --split train --num_workers ${NUM_WORKERS}
python pointcept/datasets/preprocessing/sampling_chunking_data.py --dataset_root ${PROCESSED_SCANNETPP_DIR} --grid_size 0.01 --chunk_range 6 6 --chunk_stride 3 3 --split val --num_workers ${NUM_WORKERS}
```
- Link processed dataset to codebase:
```bash
# PROCESSED_SCANNETPP_DIR: the directory of the processed ScanNet dataset.
mkdir data
ln -s ${PROCESSED_SCANNETPP_DIR} ${CODEBASE_DIR}/data/scannetpp
```

## 🚀 Training
Same to [Pointcept](https://github.com/Pointcept/Pointcept), the training process is based on configs in `configs` folder. The training scripts will create an experiment folder in `exp` and backup essential code in the experiment folder. Training config, log file, tensorboard, and checkpoints will also be saved during the training process.

**Attention:** Note that a cricital difference from Pointcept is that most of data augmentation operations are conducted on GPU in this [file](/pointcept/custom/transform_tensor.py). Make sure `ToTensor` is before the augmentation operations.

**ScanNet V2, SGIFormer**
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 sh scripts/train.sh -g 4 -d scannet -c insseg-spformer-pt-spunet-v2 -n insseg-spformer-pt-spunet-v2
CUDA_VISIBLE_DEVICES=0,1,2,3 sh scripts/train.sh -g 4 -d scannet -c insseg-sgiformer-spunet-v2 -n insseg-sgiformer-spunet-v2
```
You can also optionally skip the first step and use our provided pre-trained weight [scannetv2_pt](https://huggingface.co/RayYoh/SGIFormer/blob/main/scannetv2_pt.pth).

**ScanNet V2, SGIFormer-L**

First download the pre-trained backbone from [Spherical Mask](https://github.com/yunshin/SphericalMask) [Weight](https://drive.google.com/file/d/1OeHRgkEkxvPkUOrFacmNUevrrAjHW6DA/view).
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 sh scripts/train.sh -g 4 -d scannet -c insseg-sgiformer-l-spunet-v2 -n insseg-sgiformer-l-spunet-v2
```

**ScanNet200, SGIFormer**

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 sh scripts/train.sh -g 4 -d scannet200 -c insseg-spformer-pt-spunet-v2 -n insseg-spformer-pt-spunet-v2
CUDA_VISIBLE_DEVICES=0,1,2,3 sh scripts/train.sh -g 4 -d scannet200 -c insseg-sgiformer-spunet-v2 -n insseg-sgiformer-spunet-v2
```
You can also optionally skip the first step and use our provided pre-trained weight [scannet200_pt](https://huggingface.co/RayYoh/SGIFormer/blob/main/scannet200_pt.pth).

**ScanNet200, SGIFormer-L**
First download the pre-trained backbone from [Mask3D](https://github.com/JonasSchult/Mask3D) [Weight](https://github.com/oneformer3d/oneformer3d/releases/download/v1.0/mask3d_scannet200.pth), you can also use our provided weight [mask3d_scannet200](https://huggingface.co/RayYoh/SGIFormer/blob/main/mask3d_scannet200.pth).
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 sh scripts/train.sh -g 4 -d scannet200 -c insseg-sgiformer-l-minkunet -n insseg-sgiformer-l-minkunet
```

**ScanNet++, SGIFormer**
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 sh scripts/train.sh -g 4 -d scannetpp -c insseg-sgiformer-spunet-v2 -n insseg-sgiformer-spunet-v2
```
**Note**: we load the model pre-trained on ScanNet V2, you need to train ScanNet V2 first or use our provided weight [insseg-scannet-sgiformer-spunet](https://huggingface.co/RayYoh/SGIFormer/tree/main/insseg-scannet-sgiformer-spunet).


## :books: License

This repository is released under the [MIT license](LICENSE).

## :clap: Acknowledgement

Our code is primarily built upon [Pointcept](https://github.com/Pointcept/Pointcept), [OneFormer3D](https://github.com/oneformer3d/oneformer3d), [Mask3D](https://github.com/JonasSchult/Mask3D), [SPFormer](https://github.com/sunjiahao1999/SPFormer), [Spherical Mask](https://github.com/yunshin/SphericalMask). We also thank [Pytorch Lightning](https://github.com/Lightning-AI/pytorch-lightning), [Point Cloud Matters](https://github.com/HaoyiZhu/PointCloudMatters?tab=readme-ov-file), and [Mask3D](https://github.com/JonasSchult/Mask3D) for their excellent templates.

## :pencil: Citation

```bib
@article{yao2024sgiformer,
  title={SGIFormer: Semantic-guided and Geometric-enhanced Interleaving Transformer for 3D Instance Segmentation},
  author={Yao, Lei and Wang, Yi and Liu, Moyun and Chau, Lap-Pui},
  journal={IEEE Transactions on Circuits and Systems for Video Technology},
  year={2024},
  publisher={IEEE}
}
```