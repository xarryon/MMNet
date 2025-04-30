LOGNAME=log$(date +"%Y%m%d_%H%M%S")
CONFIG='./configs/r50.json'
DATA_DIR='./'
DATASET_NAME='facial_components'
RESULTS_DIR='./results'

HOST='127.0.0.1'
PORT='1234'
NUM_GPU=0

PARTITION='YOUR-PARTITION-HERE'
NODE='YOUR-NODE-HERE'

python train.py \
        --log_name ${LOGNAME} \
        --cfg ${CONFIG} \
        --data_dir ${DATA_DIR} \
        --dataset_name ${DATASET_NAME} \
        --val_epoch 5 \
        --model_save_epoch 5 \
        --manual_seed 777 \
        --dist-url tcp://${HOST}:${PORT} \
        --world_size ${NUM_GPU} \
        --rank 0 \
        --launcher pytorch \
        --results_dir ${RESULTS_DIR}
