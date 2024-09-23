export CUDA_VISIBLE_DEVICES=2

accelerate launch finetune_view_ip2p.py --config='configs/farm.yaml'