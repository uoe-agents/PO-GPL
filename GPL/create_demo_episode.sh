


## This script run the policy of one of the algorithms to create a demo episode used for evaluation. Using PO-GPL: 

python test_po_demo.py --lr=0.00025 --num-particles=20 --env-name="PO-Adhoc-Foraging-2-12x12-3f-v0" \
      --google-cloud="False" --designated-cuda="cuda:0" \
      --seed=6 --eval-init-seed=65 --eval-seed=5 \
      --logging-dir="test_lbf_log" --saving-dir="test_lbf" --exp-name="PO-GPL/exp_1" --load-from-checkpoint=40 \
      --state-gnn-hid-dims1=50 --state-gnn-hid-dims2=100 --state-gnn-hid-dims3=100 --state-gnn-hid-dims4=50 --s-dim=100 \
      --with-noise="False" --no-bptt="False" --add-prev-log-prob="False" --no-gradient-joint-action="False" --with-rnn-s-processing="False" --lrf-rank=6 \
      --stdev-regularizer-weight=0.0 --s-var-noise=0.0 --q-loss-weight=1.0  --agent-existence-prediction-weight=5. --encoding-weight=0.5 --act-reconstruction-weight=0.5 --state-reconstruction-weight=0.5 --entropy-weight=0.05
