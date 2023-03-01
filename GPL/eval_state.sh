#!/bin/bash


# number of checkpoints used
max=40
# number of seeds 
exp=8

############################################################
# With this code you can generate the analysis shown in the paper. 
# For this example, the folder structure is: 
# ```
# GPL
# │   README.md
# │   eval_state.sh    
# │
# └───test_lbf
# │   │
# │   └───GPL
# │   │    │   exp_1 # 
# │   │    │   exp_2 # 
# │   │    │   ...
# │   │
# │   └─VAE-GPL
# │   │    │   exp_1 # 
# │   │    │   exp_2 # 
# │   │    │   ...
# │   │
# ```
#
## Then the following code will go trough the folder, running the evaluation for each checkpoint and for each seed. 


## Fist time you run this it will generate a demo episode to run the evaluation. Using PO-GPL: 

# python test_po_demo.py --lr=0.00025 --num-particles=20 --env-name="PO-Adhoc-Foraging-2-12x12-3f-v0" \
#       --google-cloud="False" --designated-cuda="cuda:0" \
#       --seed=6 --eval-init-seed=65 --eval-seed=5 \
#       --logging-dir="test_lbf_log" --saving-dir="test_lbf" --exp-name="PF-GPL-20/exp_1" --load-from-checkpoint=40 \
#       --state-gnn-hid-dims1=50 --state-gnn-hid-dims2=100 --state-gnn-hid-dims3=100 --state-gnn-hid-dims4=50 --s-dim=100 \
#       --with-noise="False" --no-bptt="False" --add-prev-log-prob="False" --no-gradient-joint-action="False" --with-rnn-s-processing="False" --lrf-rank=6 \
#       --stdev-regularizer-weight=0.0 --s-var-noise=0.0 --q-loss-weight=1.0  --agent-existence-prediction-weight=5. --encoding-weight=0.5 --act-reconstruction-weight=0.5 --state-reconstruction-weight=0.5 --entropy-weight=0.05


for j in `seq 1 $exp`
do
    for i in `seq 0 $max`
    do
    echo " "
    echo "AE-GPL LBF"
    echo "seed number" $j "checkpoint number" $i
    python test_ae_demo.py --lr=0.00025 --env-name='PO-Adhoc-Foraging-2-12x12-3f-v0' \
    --google-cloud="False" --designated-cuda="gpu:0" \
    --seed=6 --eval-init-seed=65 --eval-seed=5 --load-from-checkpoint=$i \
    --q-loss-weight=1.0 --act-reconstruction-weight=0.05 --lrf-rank=6  \
    --exp-name="AE/exp_$j" --logging-dir="test_lbf_log" --saving-dir="test_lbf" 
    done
done



for j in `seq 1 $exp`
do
    for i in `seq 0 $max`
    do
    echo " "
    echo "GPL-Q LBF"
    echo "seed number" $j "checkpoint number" $i
    python test_gpl_demo.py --lr=0.00025 --env-name='PO-Adhoc-Foraging-2-12x12-3f-v0' \
      --google-cloud="False" --designated-cuda="cuda:0" \
      --seed=6 --eval-init-seed=65 --eval-seed=5 --load-from-checkpoint=$i \
      --q-loss-weight=1.0 --act-reconstruction-weight=0.05 --lrf-rank=6  \
      --exp-name="GPL/exp_$j" --logging-dir="test_lbf_log" --saving-dir="test_lbf" 
    done
done



for j in `seq 1 $exp`
do
    for i in `seq 0 $max`
    do
    echo " "
    echo "PO-GPL-10 LBF"
    echo "seed number" $j "checkpoint number" $i
    python test_po_demo.py --lr=0.00025 --num-particles=10 --env-name="PO-Adhoc-Foraging-2-12x12-3f-v0" \
     --google-cloud="False" --designated-cuda="cuda:0" \
     --seed=6 --eval-init-seed=65 --eval-seed=5 \
     --logging-dir="test_lbf_log" --saving-dir="test_lbf" --exp-name="PF-GPL-10/exp_$j" --load-from-checkpoint=$i \
     --state-gnn-hid-dims1=50 --state-gnn-hid-dims2=100 --state-gnn-hid-dims3=100 --state-gnn-hid-dims4=50 --s-dim=100 \
     --with-noise="False" --no-bptt="False" --add-prev-log-prob="False" --no-gradient-joint-action="False" --with-rnn-s-processing="False" --lrf-rank=6 \
     --stdev-regularizer-weight=0.0 --s-var-noise=0.0 --q-loss-weight=1.0  --agent-existence-prediction-weight=5. --encoding-weight=0.5 --act-reconstruction-weight=0.5 --state-reconstruction-weight=0.5 --entropy-weight=0.05
    done
done


for j in `seq 1 $exp`
do
    for i in `seq 0 $max`
    do
    echo " "
    echo "VAE-GPL LBF"
    echo "seed number" $j "checkpoint number" $i
    python test_vae_demo.py --lr=0.00025 --env-name='PO-Adhoc-Foraging-2-12x12-3f-v0' \
        --seed=6 --eval-init-seed=65 --eval-seed=5 --load-from-checkpoint=$i \
        --q-loss-weight=1.0 --act-reconstruction-weight=0.05 --lrf-rank=6  --num-particles=10 \
        --exp-name="VAE/exp_$j" --logging-dir="test_lbf_log" --saving-dir="test_lbf" 
    done
done


############################################################################
## This code will produce the graphs seen in the paper. 

python plot_demo_all_seeds_checkpoint.py   --saving-dir="test_lbf" --env-name="LBF"


