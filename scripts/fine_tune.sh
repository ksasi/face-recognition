#!/bin/bash
PYTHONUNBUFFERED=1; export PYTHONUNBUFFERED
if [ $1 = "LightCNN29" ] && [ $2 = "LFW" ]
then
    python -u ../src/fine_tune.py --save_path=../checkpoints/ --model="LightCNN_29" --dataset="LFW" --num_classes=1180 --arch="LightCNN_29" --epochs=50 --batch_size=128 --learning_rate=1e-5 --weight_decay=1e-4 --momentum=0.9 >> ../results/LightCNN29_out.log
elif [ $1 = "VGGFace2" ] && [ $2 = "LFW" ]
then
    python -u ../src/fine_tune.py --save_path=../checkpoints/ --model="VGGFace2" --dataset="LFW" --num_classes=1180 --arch="VGGFace2" --epochs=50 --batch_size=128 --learning_rate=1e-5 --weight_decay=1e-2 --momentum=0.9 >> ../results/VGGFace2_out.log
elif [ $1 = "ArcFace" ] && [ $2 = "LFW" ]
then
    python -u ../src/fine_tune.py --save_path=../checkpoints/ --model="ArcFace" --dataset="LFW" --num_classes=1180 --arch="ArcFace" --epochs=50 --batch_size=128 --learning_rate=1e-5 --weight_decay=1e-2 --momentum=0.9 >> ../results/ArcFace_out.log
elif [ $1 = "LightCNN29" ] && [ $2 = "SurvFace" ]
then
    python -u ../src/fine_tune.py --save_path=../checkpoints/ --model="LightCNN_29" --dataset="SurvFace" --num_classes=1180 --arch="LightCNN_29" --epochs=50 --batch_size=1024 --learning_rate=1e-2 --weight_decay=1e-4 --momentum=0.9 >> ../results/LightCNN29_SurvFace_out.log
elif [ $1 = "VGGFace2" ] && [ $2 = "SurvFace" ]
then
    python -u ../src/fine_tune.py --save_path=../checkpoints/ --model="VGGFace2" --dataset="SurvFace" --num_classes=1180 --arch="VGGFace2" --epochs=50 --batch_size=128 --learning_rate=1e-2 --weight_decay=1e-4 --momentum=0.9 >> ../results/VGGFace2_SurvFace_out.log
elif [ $1 = "ArcFace" ] && [ $2 = "SurvFace" ]
then
    python -u ../src/fine_tune.py --save_path=../checkpoints/ --model="ArcFace" --dataset="SurvFace" --num_classes=1180 --arch="ArcFace" --epochs=50 --batch_size=128 --learning_rate=1e-2 --weight_decay=1e-4 --momentum=0.9 >> ../results/ArcFace_SurvFace_out.log
else
    echo "Incorrect Arguments"
fi
