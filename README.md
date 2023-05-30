# Learning Common Rationale to Improve Self-Supervised Representation for Fine-Grained Visual Recognition Problems 
<img src="examples/framework.png"> 
This project contains the implementation of learning common rationale to improve self-supervised representation for fine-grained visual recognition, as presented in our paper

> Learning Common Rationale to Improve Self-Supervised Representation for Fine-Grained Visual Recognition Problems,   
> Yangyang Shu, Anton van den Hengel and Lingqiao Liu*  
> *To be appeared in CVPR 2023*

## Datasets
| Dataset | Download Link |
| -- | -- |
| CUB-200-2011 | https://paperswithcode.com/dataset/cub-200-2011 |
| Stanford Cars | http://ai.stanford.edu/~jkrause/cars/car_dataset.html |
| FGVC Aircraft | http://www.robots.ox.ac.uk/~vgg/data/fgvc-aircraft/ |


Please download and organize the datasets in this structure:
```
LCR
├── CUB200/
│   ├── train/ 
    ├── test/
    ├── image_list/
├── StanfordCars/
│   ├── train/ 
    ├── test/
    ├── image_list/
├── Aircraft/
│   ├── train/ 
    ├── test/
    ├── image_list/
```

# For byol
Install the required packages:
```
pip install -r requirements.txt
```

## RUN
- The running commands for several datasets are shown below. You can also refer to ``run.sh``.
```
python main.py --data_dir ./CUB200 --log_dir ./logs/ -c configs/byol_cub200.yaml --ckpt_dir ./.cache/ --hide_progress
python main.py --data_dir ./StanfordCars --log_dir ./logs/ -c configs/byol_stanfordcars.yaml --ckpt_dir ./.cache/ --hide_progress
python main.py --data_dir ./Aircraft --log_dir ./logs/ -c configs/byol_aircrafts.yaml --ckpt_dir ./.cache/ --hide_progress

```

# For moco v2

Code will come soon.


