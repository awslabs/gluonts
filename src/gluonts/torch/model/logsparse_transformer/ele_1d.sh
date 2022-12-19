#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=3
now=$(date +"%Y%m%d_%H%M%S")
dataset='ele'
batch_size=64
epoch=20
instances=500000
lr=0.001
overlap=True
pred_days=7
enc_len=168
dec_len=24
embedded_dim=20
scale_att=True
n_head=8
q_len='1'
num_layers=3
print_freq=100
input_size=5
v_partition=0.1
early_stop_ep=20
path='data/electricity.txt'
for seed in "0" "1" "2" "3" "4"
do
  info='Fixed+Adam'+${seed}+${q_len}+${enc_len}+${dec_len}+${now}
  outdir='ele/'${info}
  if [ ! -d "${outdir}" ]; then
      mkdir -p ${outdir}
  fi
  echo ${outdir}
  nohup python -u main.py --embedded_dim ${embedded_dim} \
  --early_stop_ep ${early_stop_ep} \
  --v_partition ${v_partition} \
  --seed ${seed} \
  --print-freq ${print_freq} \
  --input-size ${input_size} \
  --dataset ${dataset} \
  --q_len ${q_len} \
  --dec_len ${dec_len} \
  --pred_days ${pred_days} \
  --enc_len ${enc_len} \
  --train-ins-num ${instances} \
  --n_head ${n_head} \
  --lr ${lr} \
  --path ${path} \
  --num-layers ${num_layers} \
  --outdir ${outdir} \
  --batch-size ${batch_size} \
  --overlap \
  --scale_att \
  --epoch ${epoch} > ${outdir}'/train.log' 2>&1 &
done
