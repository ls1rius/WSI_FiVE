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
#### For End-to-end Training:
- End-to-end training will train the backbone of the model at the same time, and its performance limit will be higher, but it will also require greater GPU usage. It is recommended that the parameter NUM_FRAMES can be reduced appropriately, around 2048.
- At the same time, due to the differences in the original WSI clipping method and data storage method, the data reading method may need to be adjusted appropriately. You can modify line 1874 in the file "datasets/pipeline.py" to suit your local training data.


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
workdirs/five_fix_pth_95.4.pth
```
PTH can be found [here](https://drive.google.com/file/d/1Z1MO-IYuosW2kAw04GHMUguAc345jnf0/view?usp=sharing)

## Log
- 20240405 fix pth version upload fv_2.0.0
- 20240406 add end to end training model fv_2.0.1
- 20240427 add pth
- 20240622 delete part redundant code

## TODO
- optimize code
- add some annotation

## Bibtex
If this project is useful for you, please consider citing our paper.
```
@inproceedings{li2024generalizable,
  title={Generalizable Whole Slide Image Classification with Fine-Grained Visual-Semantic Interaction},
  author={Li, Hao and Chen, Ying and Chen, Yifei and Yu, Rongshan and Yang, Wenxian and Wang, Liansheng and Ding, Bowen and Han, Yuchen},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={11398--11407},
  year={2024}
}
```

## Acknowledgements
Parts of the codes are borrowed from [X-CLIP](https://github.com/microsoft/VideoX/tree/master/X-CLIP), [MedCLIP](https://github.com/RyanWangZf/MedCLIP). 
Sincere thanks to their wonderful works.