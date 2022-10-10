#!/bin/bash
#SBATCH --job-name=GPL_LBF
#SBATCH --cpus-per-task=4
#SBATCH --time=5-12:00:00
#SBATCH --mem-per-cpu=4G
#SBATCH --gres=gpu:1

source activate myenv

export OMP_NUM_THREADS=1

#export CUDA_VISIBLE_DEVICES="4"



python gpl_only_train.py --lr=0.00025 --env-name='PO-Adhoc-Foraging-2-12x12-3f-v0' \
    --google-cloud="False" --designated-cuda="cuda:0" \
    --seed=2 --eval-seed=700 \
	--q-loss-weight=1.0 --act-reconstruction-weight=0.05 --lrf-rank=6  \
	--exp-name="exp_1" --logging-dir="logs_lbf_gpl" --saving-dir="param_lbf_gpl" 
