export CUDA_VISIBLE_DEVICES=0,
export PYTHONPATH=$PYTHONPATH:./

python stream_train.py --config configs/n3dv/train_coffee_50_2.txt \
    --datadir ./data/neural_3d/coffee_martini \
    --basedir ./log/neural_3d \
    --render_test 1 --render_path 1