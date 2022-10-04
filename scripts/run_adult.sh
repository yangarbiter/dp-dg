#!/usr/bin/env bash

ROOTDIR="data"

BATCHSIZE="256"
DATASET="adult"
EPOCHS="20"
CLIPNORM="0.1"


mkdir -p ./logs/${DATASET}

#############################
# ERM + DPSGDf
#############################
MODEL="logistic_regression"
LR="0.1"
SAMPLERATE="0.005"
wd=0.01

for CLIPNORM in 0.5
do
  for SIGMA in 1.0
  do
    SIGMA2=$(echo "$SIGMA 10.0" | awk '{print $1 * $2}')
    for SEED in 1 2 3 4 5
    do
      # Running fair DP-SGD (DP-SGD-f)
      PYTHONPATH=. python examples/run_expt.py \
        --dataset $DATASET --model $MODEL --n_epochs $EPOCHS --batch_size $BATCHSIZE --root_dir $ROOTDIR \
        --optimizer SGD --delta 1e-5 --sigma ${SIGMA} --max_per_sample_grad_norm $CLIPNORM --enable_fair_privacy \
        --uniform_iid --sample_rate $SAMPLERATE --weight_decay 0.01 --lr ${LR} --seed $SEED \
        --C0 ${CLIPNORM} --sigma2 ${SIGMA2}\
        --log_dir ./logs/${DATASET}/dpsgdferm-${MODEL}-lr${LR}-dpsgdf_1e-5_${SIGMA}_${CLIPNORM}_${SAMPLERATE}_${SEED} \
        --algorithm ERMDPSGDf --download

      # Running standard DP-SGD
      PYTHONPATH=. python examples/run_expt.py \
        --dataset $DATASET --model $MODEL --n_epochs $EPOCHS --batch_size $BATCHSIZE --root_dir $ROOTDIR \
        --optimizer SGD --delta 1e-5 --sigma ${SIGMA} --max_per_sample_grad_norm $CLIPNORM --enable_privacy \
        --uniform_iid --sample_rate $SAMPLERATE --weight_decay ${wd} --lr ${LR} --seed $SEED \
        --log_dir ./logs/${DATASET}/erm-${MODEL}-lr${LR}-dpsgd_1e-5_${SIGMA}_${CLIPNORM}_${SAMPLERATE}_wd${wd}_${SEED} \
        --algorithm ERM --download
    done
  done
done

for CLIPNORM in 0.5
do
  for SIGMA in 5.0
  do
    SIGMA2=$(echo "$SIGMA 10.0" | awk '{print $1 * $2}')
    for SEED in 1 2 3 4 5
    do
      # Running DP-IS-SGD
      PYTHONPATH=. python examples/run_expt.py \
        --dataset $DATASET --model $MODEL --n_epochs $EPOCHS --batch_size $BATCHSIZE --root_dir $ROOTDIR \
        --optimizer SGD --delta 1e-5 --sigma ${SIGMA} --max_per_sample_grad_norm $CLIPNORM --enable_privacy \
        --weighted_uniform_iid --sample_rate ${SAMPLERATE} --weight_decay ${wd} --seed $SEED \
        --log_dir ./logs/${DATASET}/weightederm-${MODEL}-dpsgd_1e-5_${SIGMA}_${CLIPNORM}_${SAMPLERATE}_wd${wd}_${SEED} \
        --algorithm ERM
    done
  done
done

for SEED in 1 2 3 4 5
do
  # Running standard ERM
  PYTHONPATH=. python examples/run_expt.py \
    --dataset $DATASET --model $MODEL --n_epochs $EPOCHS --batch_size $BATCHSIZE --root_dir $ROOTDIR \
    --weight_decay ${wd} --lr ${LR} \
    --log_dir ./logs/${DATASET}/erm-${MODEL}-lr${LR}_wd${wd}_${SEED} \
    --algorithm ERM --weight_decay ${wd} --download
done
