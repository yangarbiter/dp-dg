#!/usr/bin/env bash

ROOTDIR="data"

MODEL="dp_bert-base-uncased"
BATCHSIZE="16"
DATASET="civilcomments"
EPOCHS="5"
SAMPLERATE="0.0002"
SIGMA="0.01"
CLIPNORM="0.1"

mkdir -p ./logs/${DATASET}

SIGMA="0.001"
CLIPNORM="1.0"
SAMPLERATE="0.0002"

# weighted + DPSGD
MODEL="dpall_bert-base-uncased"
SAMPLERATE="0.000005"
PYTHONPATH=. python examples/run_expt.py \
  --dataset $DATASET --model $MODEL --n_epochs $EPOCHS --batch_size $BATCHSIZE --root_dir $ROOTDIR \
  --optimizer AdamW --delta 1e-5 --sigma ${SIGMA} --max_per_sample_grad_norm $CLIPNORM --enable_privacy \
  --weighted_uniform_iid --sample_rate ${SAMPLERATE} --weight_decay 0. --lr 1e-5 \
  --log_dir ./logs/${DATASET}/weightederm-${MODEL}-lr1e-5_dpAdamW_1e-5_${SIGMA}_${CLIPNORM}_${SAMPLERATE} \
  --algorithm ERM

#############################
# ISERM + noise
#############################
MODEL="dpall_bert-base-uncased"
BATCHSIZE="16"
LR="1e-5"
for SIGMA in 0.001 0.01 # 0.1 1.0
do
  python examples/run_expt.py \
    --dataset $DATASET --model $MODEL --n_epochs $EPOCHS --batch_size $BATCHSIZE --root_dir $ROOTDIR \
    --optimizer SGD --delta 1e-5 --sigma ${SIGMA} --apply_noise \
    --weight_decay 0. --lr ${LR} --uniform_over_groups \
    --log_dir ./logs/${DATASET}/erm_reweight-${MODEL}-lr${LR}-noisesgd_${SIGMA} \
    --algorithm ERM --download
done

# ERM + DPSGD
#python examples/run_expt.py \
#  --dataset $DATASET --model $MODEL --n_epochs $EPOCHS --batch_size $BATCHSIZE --root_dir $ROOTDIR \
#  --optimizer SGD --delta 1e-5 --sigma ${SIGMA} --max_per_sample_grad_norm $CLIPNORM --enable_privacy \
#  --uniform_iid --sample_rate $SAMPLERATE --weight_decay 0. \
#  --log_dir ./logs/${DATASET}/erm-${MODEL}-lr1e-5_dpAdamW_1e-5_${SIGMA}_${CLIPNORM}_${SAMPLERATE} \
#  --algorithm ERM --download

# IWERM + DPSGD
#python examples/run_expt.py \
#  --dataset $DATASET --model $MODEL --n_epochs $EPOCHS --batch_size $BATCHSIZE --root_dir $ROOTDIR \
#  --optimizer SGD --delta 1e-5 --sigma ${SIGMA} --max_per_sample_grad_norm $CLIPNORM --enable_privacy \
#  --uniform_iid --sample_rate $SAMPLERATE --weight_decay 0. \
#  --log_dir ./logs/${DATASET}/iwerm-${MODEL}-lr1e-5_dpAdamW_1e-5_${SIGMA}_${CLIPNORM}_${SAMPLERATE} \
#  --algorithm IWERM --download

# weighted + DPSGD
#PYTHONPATH=. python examples/run_expt.py \
#  --dataset $DATASET --model $MODEL --n_epochs $EPOCHS --batch_size $BATCHSIZE --root_dir $ROOTDIR \
#  --optimizer AdamW --delta 1e-5 --sigma ${SIGMA} --max_per_sample_grad_norm $CLIPNORM --enable_privacy \
#  --weighted_uniform_iid --sample_rate ${SAMPLERATE} --weight_decay 0. --lr 1e-6\
#  --log_dir ./logs/${DATASET}/weightederm-${MODEL}-lr1e-5_dpAdamW_1e-5_${SIGMA}_${CLIPNORM}_${SAMPLERATE} \
#  --algorithm ERM
#
## DRO + DPSGD
#python examples/run_expt.py \
#  --dataset $DATASET --model $MODEL --n_epochs $EPOCHS --batch_size $BATCHSIZE --root_dir $ROOTDIR \
#  --optimizer AdamW --delta 1e-5 --sigma ${SIGMA} --max_per_sample_grad_norm $CLIPNORM --enable_privacy \
#  --uniform_iid --sample_rate ${SAMPLERATE} --weight_decay 0. --lr 1e-6 \
#  --log_dir ./logs/${DATASET}/groupdro-${MODEL}-lr1e-5_dpAdamW_1e-5_${SIGMA}_${CLIPNORM}_${SAMPLERATE} \
#  --algorithm groupDRO --download

#######################################
############ Not used #################
#######################################

#SAMPLERATE="0.00025"
#CLIPNORM="0.1"
#EPOCHS="10"
#PYTHONPATH=. python examples/run_expt.py \
#  --dataset $DATASET --model $MODEL --n_epochs $EPOCHS --batch_size $BATCHSIZE --root_dir $ROOTDIR \
#  --optimizer AdamW --delta 1e-5 --sigma ${SIGMA} --max_per_sample_grad_norm $CLIPNORM --enable_privacy \
#  --uniform_iid  --sample_rate $SAMPLERATE --weight_decay 0. \
#  --log_dir ./logs/${DATASET}/subsamplederm-${MODEL}-dpAdamW_1e-5_ep${epochs}_${SIGMA}_${CLIPNORM}_${SAMPLERATE} \
#  --algorithm ERM --subsample
#
#SAMPLERATE="0.0025"
#CLIPNORM="10."
#EPOCHS="10"
#PYTHONPATH=. python examples/run_expt.py \
#  --dataset $DATASET --model $MODEL --n_epochs $EPOCHS --batch_size $BATCHSIZE --root_dir $ROOTDIR \
#  --optimizer AdamW --delta 1e-5 --sigma ${SIGMA} --max_per_sample_grad_norm $CLIPNORM --enable_privacy \
#  --uniform_iid  --sample_rate $SAMPLERATE --weight_decay 0. \
#  --log_dir ./logs/${DATASET}/subsamplederm-${MODEL}-dpAdamW_1e-5_ep${epochs}_${SIGMA}_${CLIPNORM}_${SAMPLERATE} \
#  --algorithm ERM --subsample
#
#SAMPLERATE="0.0025"
#CLIPNORM="1."
#SIGMA="0.001"
#EPOCHS="10"
#PYTHONPATH=. python examples/run_expt.py \
#  --dataset $DATASET --model $MODEL --n_epochs $EPOCHS --batch_size $BATCHSIZE --root_dir $ROOTDIR \
#  --optimizer AdamW --delta 1e-5 --sigma ${SIGMA} --max_per_sample_grad_norm $CLIPNORM --enable_privacy \
#  --uniform_iid  --sample_rate $SAMPLERATE --weight_decay 0. \
#  --log_dir ./logs/${DATASET}/subsamplederm-${MODEL}-dpAdamW_1e-5_ep${epochs}_${SIGMA}_${CLIPNORM}_${SAMPLERATE} \
#  --algorithm ERM --subsample

#SAMPLERATE="0.00025"
#CLIPNORM="0.1"
#EPOCHS="10"
#PYTHONPATH=. python examples/run_expt.py \
#  --dataset $DATASET --model $MODEL --n_epochs $EPOCHS --batch_size $BATCHSIZE --root_dir $ROOTDIR \
#  --optimizer SGD --delta 1e-5 --sigma ${SIGMA} --max_per_sample_grad_norm $CLIPNORM --enable_privacy \
#  --uniform_iid  --sample_rate $SAMPLERATE --weight_decay 0. \
#  --log_dir ./logs/${DATASET}/subsamplederm-${MODEL}-dpsgd_1e-5_ep${epochs}_${SIGMA}_${CLIPNORM}_${SAMPLERATE} \
#  --algorithm ERM --subsample
#
#SAMPLERATE="0.0025"
#CLIPNORM="10."
#EPOCHS="10"
#PYTHONPATH=. python examples/run_expt.py \
#  --dataset $DATASET --model $MODEL --n_epochs $EPOCHS --batch_size $BATCHSIZE --root_dir $ROOTDIR \
#  --optimizer SGD --delta 1e-5 --sigma ${SIGMA} --max_per_sample_grad_norm $CLIPNORM --enable_privacy \
#  --uniform_iid  --sample_rate $SAMPLERATE --weight_decay 0. \
#  --log_dir ./logs/${DATASET}/subsamplederm-${MODEL}-dpsgd_1e-5_ep${epochs}_${SIGMA}_${CLIPNORM}_${SAMPLERATE} \
#  --algorithm ERM --subsample
#
#SAMPLERATE="0.0025"
#CLIPNORM="1."
#SIGMA="0.001"
#EPOCHS="10"
#PYTHONPATH=. python examples/run_expt.py \
#  --dataset $DATASET --model $MODEL --n_epochs $EPOCHS --batch_size $BATCHSIZE --root_dir $ROOTDIR \
#  --optimizer SGD --delta 1e-5 --sigma ${SIGMA} --max_per_sample_grad_norm $CLIPNORM --enable_privacy \
#  --uniform_iid  --sample_rate $SAMPLERATE --weight_decay 0. \
#  --log_dir ./logs/${DATASET}/subsamplederm-${MODEL}-dpsgd_1e-5_ep${epochs}_${SIGMA}_${CLIPNORM}_${SAMPLERATE} \
#  --algorithm ERM --subsample

# ERM + DPSGD
#python examples/run_expt.py \
#  --dataset $DATASET --model $MODEL --n_epochs $EPOCHS --batch_size $BATCHSIZE --root_dir $ROOTDIR \
#  --optimizer SGD --delta 1e-5 --sigma ${SIGMA} --max_per_sample_grad_norm $CLIPNORM --enable_privacy \
#  --uniform_iid --sample_rate $SAMPLERATE --weight_decay 0. \
#  --log_dir ./logs/${DATASET}/erm-${MODEL}-dpsgd_1e-5_${SIGMA}_${CLIPNORM}_${SAMPLERATE} \
#  --algorithm ERM --download

# weighted + DPSGD
#PYTHONPATH=. python examples/run_expt.py \
#  --dataset $DATASET --model $MODEL --n_epochs $EPOCHS --batch_size $BATCHSIZE --root_dir $ROOTDIR \
#  --optimizer SGD --delta 1e-5 --sigma ${SIGMA} --max_per_sample_grad_norm $CLIPNORM --enable_privacy \
#  --weighted_uniform_iid --sample_rate ${SAMPLERATE} --weight_decay 0. --lr 1e-6\
#  --log_dir ./logs/${DATASET}/weightederm-${MODEL}-lr1e-6_dpsgd_1e-5_${SIGMA}_${CLIPNORM}_${SAMPLERATE} \
#  --algorithm ERM

## subsample + DPSGD
#PYTHONPATH=. python examples/run_expt.py \
#  --dataset $DATASET --model $MODEL --n_epochs $EPOCHS --batch_size $BATCHSIZE --root_dir $ROOTDIR \
#  --optimizer SGD --delta 1e-5 --sigma ${SIGMA} --max_per_sample_grad_norm 1.0 --enable_privacy \
#  --uniform_iid  --sample_rate $SAMPLERATE --weight_decay 0. \
#  --log_dir ./logs/${DATASET}/subsamplederm-${MODEL}-dpsgd_1e-5_${SIGMA}_1.0_${SAMPLERATE} \
#  --algorithm ERM --subsample

# DRO + DPSGD
#python examples/run_expt.py \
#  --dataset $DATASET --model $MODEL --n_epochs $EPOCHS --batch_size $BATCHSIZE --root_dir $ROOTDIR \
#  --optimizer SGD --delta 1e-5 --sigma ${SIGMA} --max_per_sample_grad_norm $CLIPNORM --enable_privacy \
#  --uniform_iid --sample_rate ${SAMPLERATE} --weight_decay 0. --lr 1e-6 \
#  --log_dir ./logs/${DATASET}/groupdro-${MODEL}-lr1e-6_dpsgd_1e-5_${SIGMA}_${CLIPNORM}_${SAMPLERATE} \
#  --algorithm groupDRO --download
