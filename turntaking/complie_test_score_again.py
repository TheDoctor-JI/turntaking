# coding: UTF-8
import datetime
import json
import argparse
import os
import random
import time
import warnings
from os.path import join, dirname
from pprint import pprint

import hydra
import pandas as pd
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf 
from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingLR
from timm.scheduler import CosineLRScheduler
from tqdm import tqdm

from turntaking.callbacks import EarlyStopping
from turntaking.evaluation import roc_to_threshold
from turntaking.model import Model
from turntaking.test import Test
from turntaking.utils import to_device, repo_root, everything_deterministic, write_json, set_seed, set_debug_mode
from turntaking.dataload import DialogAudioDM
import shutil

everything_deterministic()
warnings.simplefilter("ignore")


@hydra.main(config_path="conf", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    
    cfg_dict = dict(OmegaConf.to_object(cfg))


    def compile_scores(score_json_path, output_dir):
        df = pd.DataFrame()
        for i, path in enumerate(score_json_path):
            with open(path, 'r') as f:
                data = json.load(f)
            temp_df = pd.json_normalize(data)
            temp_df['model'] = f'model{i:02}'
            temp_df['score_json_path'] = path
            df = pd.concat([df, temp_df], ignore_index=True)
        
        avg_row = df.select_dtypes(include=[np.number]).mean()
        avg_row['model'] = 'Average'
        avg_row['score_json_path'] = f'{join(output_dir, "final_score.csv")}'
        avg_df = pd.DataFrame([avg_row])
        df = pd.concat([df, avg_df], ignore_index=True)
        df = df[['model', 'score_json_path'] + [col for col in df.columns if col not in ['model', 'score_json_path']]]
        return df

    nickname = 'audio_no_va_hist'
    d = '2025_06_10'
    id = '224052'
    output_parent_folder = os.path.join(repo_root(), "output", f"{nickname}_{d}_{id}")

    score_json_path = []

    for i in range(10):
        ### Preparation ###

        # Create subdirectory for each trial
        output_dir = os.path.join(output_parent_folder, str(i).zfill(2))

        score_json_path.append(join(output_dir, "score.json"))

    df = compile_scores(score_json_path, output_parent_folder)
    print("-" * 60)
    print(f"Output Final Score -> {join(output_parent_folder, 'final_score.csv')}")
    print(df)
    print("-" * 60)
    write_json(cfg_dict, os.path.join(output_parent_folder, "log.json")) # log file
    df.to_csv(join(output_parent_folder, "final_score.csv"), index=False)


if __name__ == "__main__":

    main()