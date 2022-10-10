#!/bin/bash
#SBATCH --job-name=GLANCE_1_lbf
#SBATCH --cpus-per-task=4
#SBATCH --time=8-24:00:00
#SBATCH --mem-per-cpu=4G
#SBATCH --gres=gpu:1

source activate myenv

export OMP_NUM_THREADS=1 
export CUDA_VISIBLE_DEVICES="0"
python train_state.py --lr=0.00025 --num-particles=10  --env-name='PO-Adhoc-Foraging-2-12x12-3f-v0' \
 --google-cloud="False" --designated-cuda="cuda:0" \
 --seed=6 --eval-init-seed=65 --eval-seed=5 \
 --logging-dir="logs_lbf_glance_1" --saving-dir="param_lbf_glance_1" --exp-name="exp_1" --load-from-checkpoint=-1 \
 --state-gnn-hid-dims1=50 --state-gnn-hid-dims2=100 --state-gnn-hid-dims3=100 --state-gnn-hid-dims4=50 --s-dim=100 \
 --with-noise="False" --no-bptt="False" --add-prev-log-prob="False" --no-gradient-joint-action="False" --with-rnn-s-processing="False" --lrf-rank=6 \
 --stdev-regularizer-weight=0.0 --s-var-noise=0.0 --q-loss-weight=1.0  --agent-existence-prediction-weight=5. --encoding-weight=0.5 --act-reconstruction-weight=0.5 --state-reconstruction-weight=0.5 --entropy-weight=0.05

