# WSI_FiVE
Paper: Generalizable Whole Slide Image Classification with Fine-Grained Visual-Semantic Interaction
(CVPR 2024) https://arxiv.org/abs/2402.19326.

## Data Preparation

### Pathology Reports
The original TCGA pathology report data comes from https://github.com/tatonetti-lab/tcga-path-reports.

The GPT preprocessing code and data are provided in `gpt_preprocess`.

### Pathology Image Dataset
You can download and process the image dataset follow [DSMIL](https://github.com/binli123/dsmil-wsi).

Or you can directly download the precomputed features here: 
[Camelyon16](https://uwmadison.box.com/shared/static/l9ou15iwup73ivdjq0bc61wcg5ae8dwe.zip),
[TCGA](https://uwmadison.box.com/shared/static/tze4yqclajbdzjwxyb8b1umfwk9vcdwq.zip), 
which are also provided by [DSMIL](https://github.com/binli123/dsmil-wsi). 

Or download by code.
```angular2html
python download.py --dataset=tcga
python download.py --dataset=c16
```
This dataset requires 30GB of free disk space.

## Environment Setup
To set up the environment, you can easily run the following command:
```
conda create -n wsifv python=3.8.16
conda activate wsifv
pip install -r requirements.txt
```

Install Apex as follows
```
git clone https://github.com/NVIDIA/apex
cd apex
pip install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
```
## Model
The default training model is trained with fixed pth. To train the model end-to-end, change the parameter `IS_IMG_PTH` to `False` in the `configs`.
### Train
The config files lie in `configs`.
```
CUDA_VISIBLE_DEVICES=0,1 \
python -m torch.distributed.launch \
--nproc_per_node=2 \
--master_port=20138 \
main.py \
-cfg configs/wsi/fix_pth.yaml \
--output workdirs/tmp_cp
```

### Test
```
CUDA_VISIBLE_DEVICES=1 \
python -m torch.distributed.launch \
--nproc_per_node=1 \
--master_port=24528 \
main.py \
-cfg configs/wsi/fix_pth.yaml \
--output workdirs/tmp \
--only_test \
--pretrained \
workdirs/tmp_cp/ckpt_epoch_27.pth
```

## Log
- 20240405 fix pth version upload fv_2.0.0
- 20240406 add end to end training model fv_2.0.1

## TODO
- optimize code
- add some annotation

## Bibtex
If this project is useful for you, please consider citing our paper.

## Acknowledgements
Parts of the codes are borrowed from [X-CLIP](https://github.com/microsoft/VideoX/tree/master/X-CLIP), [MedCLIP](https://github.com/RyanWangZf/MedCLIP). 
Sincere thanks to their wonderful works.