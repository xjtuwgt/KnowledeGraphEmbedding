#!/bin/sh

python -u -c 'import torch; print(torch.__version__)'

CODE_PATH=codes
DATA_PATH=data
SAVE_PATH=models

#The first four parameters must be provided
MODE=$1
MODEL=$2
DATASET=$3
GPU_DEVICE=$4
SAVE_ID=$5

FULL_DATA_PATH=$DATA_PATH/$DATASET
SAVE=$SAVE_PATH/"$MODEL"_"$DATASET"_"$SAVE_ID"

#Only used in training
BATCH_SIZE=$6
HIDDEN_DIM=$7
ENT_DIM=$8
REL_DIM=$9
EMB_DIM=${10}
GAMMA=${11}
ALPHA=${12}
LEARNING_RATE=${13}
MAX_STEPS=${14}
TEST_BATCH_SIZE=${15}
ATT_DROP=${16}
FEA_DROP=${17}
INP_DROP=${18}
TOPK=${19}
HOPS=${20}
LAYERS=${21}
TRANS=${22}
GRAPH=${23}
MASK=${24}
LOSSTYPE=${25}
NEG=${26}
PROJECT=${27}
HEADS=${28}
DECAY=${29}
CONV_SHAPE=${30}
CONV_FSIZE=${31}
CONV_CHAN=${32}
KTYPE=${33}
FFORWARD=${34}
EDROP=${35}
ZEROON=${36}
SEED=${37}


if [ $MODE == "train" ]
then

echo "Start Training......"

/mnt/cephfs2/asr/users/ming.tu/sgetools/run_gpu.sh python -u $CODE_PATH/run.py --do_train \
    --cuda \
    --do_valid \
    --do_test \
    --model $MODEL\
    --data_path $FULL_DATA_PATH \
    -b $BATCH_SIZE -d $HIDDEN_DIM -ee $ENT_DIM -er $REL_DIM -e $EMB_DIM\
    -g $GAMMA \
    -lr $LEARNING_RATE --max_steps $MAX_STEPS \
    -save $SAVE --test_batch_size $TEST_BATCH_SIZE \
    --att_drop $ATT_DROP\
    --fea_drop $FEA_DROP\
    --top_k $TOPK\
    --hops $HOPS\
    --layers $LAYERS\
    --alpha $ALPHA\
    --trans_on $TRANS\
    --graph_on $GRAPH\
    --mask_on $MASK\
    --reszero $ZEROON\
    --loss_type $LOSSTYPE\
    --neg_on $NEG\
    --input_drop $INP_DROP\
    --project_on $PROJECT\
    --adam_weight_decay $DECAY\
    --seed $SEED\
    --num_heads $HEADS\
    --conv_embed_shape1 $CONV_SHAPE\
    --conv_filter_size $CONV_FSIZE\
    --conv_channels $CONV_CHAN\
    --topk_type $KTYPE\
    --feed_forward $FFORWARD\
    --edge_drop $EDROP\
    --seed $SEED

elif [ $MODE == "valid" ]
then

echo "Start Evaluation on Valid Data Set......"

/mnt/cephfs2/asr/users/ming.tu/sgetools/run_gpu.sh python -u $CODE_PATH/run.py --do_valid --cuda -init $SAVE
    
elif [ $MODE == "test" ]
then

echo "Start Evaluation on Test Data Set......"

/mnt/cephfs2/asr/users/ming.tu/sgetools/run_gpu.sh python -u $CODE_PATH/run.py --do_test --cuda -init $SAVE

else
   echo "Unknown MODE" $MODE
fi