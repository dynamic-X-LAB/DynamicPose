device=$2
if [ $1 == 1 ]
then
    echo "stage 1 test"
    # stage 1
    CUDA_VISIBLE_DEVICES=$device python3 -m scripts.pose2img --config ./configs/prompts/animation_stage1.yaml -W 512 -H 768
else
    echo "stage 2 test"
    # stage 2
    CUDA_VISIBLE_DEVICES=$device python3 -m   scripts.pose2vid --config ./configs/prompts/animation_stage2.yaml -W 512 -H 768 -L 180

fi
