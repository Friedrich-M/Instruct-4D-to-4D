export PYTHONPATH=.
export CUDA_VISIBLE_DEVICES=0

data_dir=data/dycheck/mochi-high-five
load_dir=logs/mochi-high-five/edit-nerfacto/2024-09-22_154420/nerfstudio_models

output_dir=./

ns-train edit-nerfacto \
    --data ${data_dir} --load-dir ${load_dir} \
    --render-mode True --output-dir ${output_dir}