#export PATH=/home/ubuntu/miniconda3/envs/dl/bin:$PATH
#

python main.py --data_dir ./CUB200 --log_dir ./logs/ -c configs/byol_cub200.yaml --ckpt_dir ./.cache/ --hide_progress
python main.py --data_dir ./StanfordCars --log_dir ./logs/ -c configs/byol_stanfordcars.yaml --ckpt_dir ./.cache/ --hide_progress
python main.py --data_dir ./Aircraft --log_dir ./logs/ -c configs/byol_aircrafts.yaml --ckpt_dir ./.cache/ --hide_progress
