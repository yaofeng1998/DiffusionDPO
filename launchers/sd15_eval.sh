#!/bin/bash
echo "STARTED"
counted=0
selected_gpu=0

TXTLOGDIR="./toy/"
if [ ! -d $TXTLOGDIR ]; then
  mkdir $TXTLOGDIR
fi

main() {


    TASK="/home/huayu/git/DiffusionDPO/tmp-sd15-scoredpo_b1w10000/checkpoint-1000"
    seed=0
    setting="eval"
    txtname=$TXTLOGDIR`date '+%m-%d-%H-%M-%S'`_exp${TASK//\//-}${seed}$setting.txt
    CUDA_VISIBLE_DEVICES=4 python3 -u evaluate_unique_prompt.py --path ${TASK} > $txtname 2>&1 &


    TASK="/home/huayu/git/DiffusionDPO/tmp-sd15-scoredpo_b1w1000/checkpoint-1000"
    seed=0
    setting="eval"
    txtname=$TXTLOGDIR`date '+%m-%d-%H-%M-%S'`_exp${TASK//\//-}${seed}$setting.txt
    CUDA_VISIBLE_DEVICES=5 python3 -u evaluate_unique_prompt.py --path ${TASK} > $txtname 2>&1 &

    TASK="/home/huayu/git/DiffusionDPO/tmp-sd15-scoredpo_b1w100/checkpoint-1000"
    seed=0
    setting="eval"
    txtname=$TXTLOGDIR`date '+%m-%d-%H-%M-%S'`_exp${TASK//\//-}${seed}$setting.txt
    CUDA_VISIBLE_DEVICES=6 python3 -u evaluate_unique_prompt.py --path ${TASK} > $txtname 2>&1 &




    TASK="/home/huayu/git/DiffusionDPO/tmp-sd15-scoredpo_b10w1000/checkpoint-1000"
    seed=0
    setting="eval"
    txtname=$TXTLOGDIR`date '+%m-%d-%H-%M-%S'`_exp${TASK//\//-}${seed}$setting.txt
    CUDA_VISIBLE_DEVICES=7 python3 -u evaluate_unique_prompt.py --path ${TASK} > $txtname 2>&1 &

    TASK="/home/huayu/git/DiffusionDPO/tmp-sd15-scoredpo_b100w1000/checkpoint-1000"
    seed=0
    setting="eval"
    txtname=$TXTLOGDIR`date '+%m-%d-%H-%M-%S'`_exp${TASK//\//-}${seed}$setting.txt
    CUDA_VISIBLE_DEVICES=4 python3 -u evaluate_unique_prompt.py --path ${TASK} > $txtname 2>&1 &


    TASK="/home/huayu/git/DiffusionDPO/tmp-sd15-scoredpo_b01w1000/checkpoint-1000"
    seed=0
    setting="eval"
    txtname=$TXTLOGDIR`date '+%m-%d-%H-%M-%S'`_exp${TASK//\//-}${seed}$setting.txt
    CUDA_VISIBLE_DEVICES=5 python3 -u evaluate_unique_prompt.py --path ${TASK} > $txtname 2>&1 &

    TASK="/home/huayu/git/DiffusionDPO/tmp-sd15-scoredpo_b001w1000/checkpoint-1000"
    seed=0
    setting="eval"
    txtname=$TXTLOGDIR`date '+%m-%d-%H-%M-%S'`_exp${TASK//\//-}${seed}$setting.txt
    CUDA_VISIBLE_DEVICES=6 python3 -u evaluate_unique_prompt.py --path ${TASK} > $txtname 2>&1 &


    TASK="/home/huayu/git/DiffusionDPO/tmp-sd15-scoredpo_b01w100/checkpoint-1000"
    seed=0
    setting="eval"
    txtname=$TXTLOGDIR`date '+%m-%d-%H-%M-%S'`_exp${TASK//\//-}${seed}$setting.txt
    CUDA_VISIBLE_DEVICES=7 python3 -u evaluate_unique_prompt.py --path ${TASK} > $txtname 2>&1 &



}


main "$@"; exit