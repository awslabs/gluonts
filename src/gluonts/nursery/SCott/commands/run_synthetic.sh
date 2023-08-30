base_dir=`pwd`

# The optimal learning rate is specified in the supplementary material
lr=1e-3
# SCSG, SCott, Adam, Adagrad, SAdam, SAdagrad
optim=Adam
# mlp, lstm, nbeats
model=linear
data=synthetic
# gamma used for SCott version optimizers, will have no effect on other optimizers like plain Adam
# the optimal gamma is specified in the supplementary material
gamma=0.25

job_name=${model}_${data}/${optim}

run_cmd="python3 run.py --optimizer=${optim} \
        --lr=${lr} \
        --model=${model} \
        --task_name=job_name \
        --prediction_length=24 \
        --context_length=72 \
        --num_batches_per_epoch=50 \
        --data_file=${data} \
        --num_strata=16 \
        --epochs=100 \
        --gamma=${gamma} \
        --anchor_freq=20 \
        --tensorboard_path=${base_dir}/runs/${job_name} \
        &> ${model}_${data}_${optim}.log"

echo ${run_cmd}
eval ${run_cmd}