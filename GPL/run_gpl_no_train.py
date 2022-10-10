#!/bin/bash
#SBATCH --job-name=P_GPL_LITE7
#SBATCH --cpus-per-task=4
#SBATCH --time=48:00:00
#SBATCH --mem-per-cpu=4G
#SBATCH --gres=gpu:1

source activate myenv

export OMP_NUM_THREADS=1
python run_gpl_no_train.py --lr=0.00025 --q-loss-weight=1.0 --agent-existence-prediction-weight=0.5 --encoding-weight=0.05 --act-reconstruction-weight=0.05 --entropy-weight=0.005 --exp-name="exp_8" --logging-dir="logs1" --saving-dir="parameters1"
