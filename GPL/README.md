# Further Instructions

This document contains further details for running the experiments presented in our work. We particularly list all the environments, training scripts, and visualisation codes to produce the results of our experiments. 
## Environments: 

The specific names of the gym environments we use in our experiments are listed below:
- LBF: PO-Adhoc-Foraging-2-12x12-3f-v0
- Wolfpack: Adhoc-wolfpack-v5
- Cooperative navigation: PO-Navigation-2-12x12-v1
- FortAttack: fortattack-v0

To run an experiment in each environment, simply change the value of the ``--env-name`` variable in the shell scripts we list in the next section.

## How to run
This section lists the different scripts responsible for running the methods used in our experiments. These scripts invoke a python script inside this folder while specifying important parameters required to run each python script. To find information regarding the different parameters for each python script, please run the following command:
```python
python <executed python script name> -h
``` 

The shell scripts used to execute our methods are listed below:
### PO-GPL 
```
./run_po_10.sh
```

### GPL 

```
./run_gpl.sh
```

### AE-GPL 

```
./run_ae.sh
```

### VAE-GPL

```
./run_vae.sh
```


## State reconstruction

Once you have executed one of the previous commands to train a model for the learner, you can perform the state reconstruction analysis by running these scripts: 

At first, you have to create a short episode that you will use in the analysis. The generation of this episode is possible by executing the following command: 
```
./create_demo_episode.sh
``` 

After an episode has been generated, please execute the following command to generate the visualisations we have in the paper:
```
./eval_state.sh
``` 

`eval_state.sh` calls two different types of scripts which respective responsibilities are defined below: 
- test_xxx_demo.py : These scripts evaluates the belief inference on the demo episode for each algorithm, and generates metrics presented in the article
- plot_demo_all_seeds_checkpoint.py : This scripts generates the plots shown in the article. 
