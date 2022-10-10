import argparse
import os
import sys
import torch
import shutil


def get_args():
    args_dict = {}
    args_dict["gamma"] = 0.99
    args_dict["tau"] = 0.001
    args_dict["num_steps"] = 1000
    args_dict["num_processes"] = 1
    args_dict["clip_param"] = 0.2
    args_dict["ppo_epoch"] = 4
    args_dict["num_mini_batch"] = 32
    args_dict["value_loss_coef"] = 0.5
    args_dict["entropy_coef"] = 0.01
    args_dict["lr"] = 0.001
    args_dict["max_grad_norm"] = 0.5
    args_dict["clipped_value_loss"] = False
    args_dict["device"] = torch.device("cuda")
    args_dict["no_cuda"] = False
    args_dict["attacker_load_dir"] = './marlsave/tmp'
    args_dict["attacker_ckpts"] = [220, 650, 1240, 1600, 2520]
    
    return args_dict
