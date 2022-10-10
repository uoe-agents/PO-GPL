#!/bin/bash
#SBATCH --job-name=Liam_LBF
#SBATCH --cpus-per-task=4
#SBATCH --time=5-12:00:00
#SBATCH --mem-per-cpu=4G
#SBATCH --gres=gpu:1

source activate myenv

export OMP_NUM_THREADS=1

# To prevent memory leak in LIAM select cuda device here


python gpl_only_state_recon_train.py --lr=0.00025  --env-name='PO-Adhoc-Foraging-2-12x12-3f-v0' \
	--google-cloud="False" --designated-cuda="cuda:0" \
	--seed=2 --eval-seed=700 \
	--act-reconstruction-weight=0.005  --states-reconstruction-weight=0.001  \
	--agent-existence-reconstruction-weight=0.02 --s-dim=100 --h-dim=100 --lrf-rank=6  \
	--q-loss-weight=1.0 --exp-name="exp_3"  --logging-dir="logs_fat_liam" --saving-dir="param_fat_liam" 
