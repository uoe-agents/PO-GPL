#!/bin/bash
#SBATCH --job-name=LIAM_Wolf
#SBATCH --cpus-per-task=4
#SBATCH --time=5-12:00:00
#SBATCH --mem-per-cpu=4G
#SBATCH --gres=gpu:1

source activate myenv

export OMP_NUM_THREADS=1

export CUDA_VISIBLE_DEVICES="0"
python gpl_only_stochastic_state_recon_train.py --lr=0.00025 --env-name='PO-Adhoc-Foraging-2-12x12-3f-v0' \
    --seed=6 --eval-seed=5 --kl-div-loss-weight=0.001 \
	--q-loss-weight=1.0 --states-reconstruction-weight=0.001 --agent-existence-reconstruction-weight=0.02 --act-reconstruction-weight=0.05 --lrf-rank=6 --num-particles=10 \
	--exp-name="exp_1" --logging-dir="logs_wolf_stochastic_liam" --saving-dir="param_wolf_stochastic_liam" 

