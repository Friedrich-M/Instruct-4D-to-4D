export PYTHONPATH=$PYTHONPATH:./
export CUDA_VISIBLE_DEVICES=0

data_dir=./data/dycheck/mochi-high-five
output_dir=./log/dycheck
max_num_iterations=30000

ns-train nerfplayer-nerfacto \
    --data ${data_dir} \
    --max-num-iterations ${max_num_iterations} \
    --output-dir ${output_dir}