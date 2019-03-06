#!/usr/bin/env bash

# To run with sbatch: sbatch --qos=unkillable --gres=gpu:titanxp --mem=32000 run_slurm1.sh

#export HOME=`getent passwd $USER | cut -d':' -f6`
#source ~/.bashrc
source activate mean_teacher

#echo Running on $HOSTNAME

#### mixup sup, mixup usup, mean teacher####
python main_vikas_mixup.py  --dataset cifar10  --num_labelled 400 --num_valid_samples 500 --root_dir /network/tmp1/vermavik/experiments/mean_teacher/ \
--data_dir /network/tmp1/vermavik/data/cifar10/ --batch_size 100  --arch cnn13 --dropout 0.0 --consistency 0.0  --mixup_consistency 10.0 --psuedo_label mean_teacher \
--consistency_rampup_starts 0 --consistency_rampup_ends 100 --epochs 300  --lr_rampdown_epochs 350 --print_freq 200 --momentum 0.9 --lr 0.1 \
--ema_decay 0.999  --mixup_sup_alpha 1.0 --mixup_usup_alpha 1.0


#### mixup sup, mean teacher####
#python ~/workspace/mean_teacher/pytorch/main_vikas_mixup.py  --dataset cifar10  --root_dir /data/milatmp1/vermavik/experiments/mean_teacher/ \
#--data_dir /data/milatmp1/vermavik/data/cifar10/ --batch_size 100  --arch WRN28_2 --consistency 50.0  --mixup_consistency 1.0 \
#--consistency_rampup_starts 0 --consistency_rampup_ends 50 --epochs 200  --print_freq 200 --momentum 0.9 --lr 0.1 \
#--ema_decay 0.999 --mixup_sup_alpha 1.0  --lr_rampdown_epochs 300


##### mean teacher####
#python ~/workspace/mean_teacher/pytorch/main_vikas_mixup.py  --dataset cifar10  --root_dir /data/milatmp1/vermavik/experiments/mean_teacher/ \
#--data_dir /data/milatmp1/vermavik/data/cifar10/ --batch_size 100  --arch WRN28_2 --consistency 50.0 \
#--consistency_rampup_starts 0 --consistency_rampup_ends 50 --epochs 200  --print_freq 200 --momentum 0.9 --lr 0.1 \
#--ema_decay 0.999 --schedule 100 150 --gamma 0.5 0.5  --lr_rampdown_epochs 300
