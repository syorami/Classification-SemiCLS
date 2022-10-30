#!/usr/bin/env bash

set -x
T=`date +%m%d%H%M`
PARTITION=$1
JOB_NAME=$2
CONFIG=$3
WORK_DIR=$4
GPUS=${GPUS:-4}
SEED=${SEED:-42}
GPUS_PER_NODE=${GPUS}
CPUS_PER_TASK=${CPUS_PER_TASK:-16}
SRUN_ARGS=${SRUN_ARGS:-""}
PY_ARGS=${@:5}

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
srun -p ${PARTITION} \
    --mpi=pmi2 \
    --job-name=${JOB_NAME} \
    --gres=gpu:${GPUS_PER_NODE} \
    --ntasks=${GPUS} \
    --ntasks-per-node=${GPUS_PER_NODE} \
    --cpus-per-task=${CPUS_PER_TASK} \
    --kill-on-bad-exit=1 \
    --quotatype=auto \
    ${SRUN_ARGS} \
    python -u train_slurm.py \
    --cfg ${CONFIG} \
    --out=${WORK_DIR} \
    --use_BN True \
    --seed ${SEED} \
    ${PY_ARGS}