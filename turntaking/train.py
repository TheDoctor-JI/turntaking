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

everything_deterministic()
warnings.simplefilter("ignore")

class Train():
    def __init__(self, conf, dm, output_path, verbose=True) -> None:
        super().__init__()
        self.conf = conf
        self.model = Model(self.conf).to(self.conf["train"]["device"])

        if verbose == True:
            self.model.model_summary
            mean, var = self.model.inference_speed
            print(f"inference speed: {mean}({var})")
            # print(self.model)
            exit(1)
        
        # self.model = torch.nn.DataParallel(self.model).to(self.conf["train"]["device"])
        self.dm = dm
        self.dm.change_frame_mode(False)
        self.output_path = output_path

        self.optimizer = self._create_optimizer()
        # self.scheduler = LambdaLR(self.optimizer, lr_lambda=lambda epoch: 0.95**epoch)
        # self.scheduler = CosineAnnealingLR(self.optimizer, T_max=self.conf["train"]["max_epochs"], eta_min=0)
        self.scheduler = CosineLRScheduler(self.optimizer, t_initial=self.conf["train"]["max_epochs"], lr_min=0, warmup_t=3, warmup_lr_init=5e-5, warmup_prefix=True)
        

        self.early_stopping = EarlyStopping(patience=self.conf["train"]["patience"], verbose=self.conf["train"]["verbose"], path=self.output_path)
        self.checkpoint = self._create_checkpoints()

    def train(self):
        self.model.net.train()
        # initial_weights = {name: weight.clone() for name, weight in self.model.named_parameters()}
        for i in range(self.conf["train"]["max_epochs"]):
            pbar = tqdm(enumerate(self.dm.train_dataloader()), total=len(self.dm.train_dataloader()), dynamic_ncols=True, leave=False)
            
            for ii, batch in pbar:
                self.optimizer.zero_grad()
                loss, _, _ = self.model.shared_step(batch)
                loss["total"].backward()
                self.optimizer.step()
                postfix = f"epoch={i}, loss={loss['total']:>3f}"
                pbar.set_postfix_str(postfix)

                if ii in self.checkpoint:
                    val_loss = self._run_validation()
                    self.model.net.train()
                    print(f"val_loss: {val_loss}")

                    self.early_stopping(val_loss, self.model)
                    if self.early_stopping.early_stop:
                        print("Early stopping triggered.")
                        break
                        
            self.scheduler.step(i+1)

            if self.early_stopping.early_stop:
                break

        # for name, weight in self.model.named_parameters():
        #     if torch.equal(weight, initial_weights[name]):
        #         print(f"Weight {name} did not change!")

        checkpoint = {
            "model": self.early_stopping.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "random": random.getstate(),
            "torch": torch.get_rng_state(),
            "torch_random": torch.random.get_rng_state(),
            "cuda_random": torch.cuda.get_rng_state(),
            "cuda_random_all": torch.cuda.get_rng_state_all(),
        }
        torch.save(checkpoint, join(dirname(self.output_path), "checkpoint.bin"))
        print(f"### END ###")

        return self.early_stopping.model


    def _create_optimizer(self):
        if self.conf["train"]["optimizer"] == "AdamW":
            return torch.optim.AdamW(self.model.parameters(), lr=self.conf["train"]["learning_rate"])
        else:
            print(f"Error optimizer")
            exit(1)


    def _create_checkpoints(self):
        def frange(start, end, step):
            list = [start]
            n = start
            while n + step < end:
                n = n + step
                list.append(n)
            return list
        
        if 0.1 <= self.conf["train"]["checkpoint"] <= 1.0:
            return [int(i * len(self.dm.train_dataloader())) - 1 for i in frange(0.0, 1.1, self.conf["train"]["checkpoint"]) if i != 0.0]
        else:
            print(f"checkpoint must be greater than 0 and less than 1")
            exit(1)

    def _run_validation(self):
        self.model.net.eval()
        pbar_val = tqdm(enumerate(self.dm.val_dataloader()), total=len(self.dm.val_dataloader()), dynamic_ncols=True, leave=False)

        val_loss = 0
        for ii, batch in pbar_val:
            if ii == 1000:
                break 
            with torch.no_grad():
                loss, _, _ = self.model.shared_step(batch)
                val_loss += loss["total"]

        val_loss /= len(self.dm.val_dataloader())

        return val_loss

@hydra.main(config_path="conf", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
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
    
    cfg_dict = dict(OmegaConf.to_object(cfg))

    if cfg_dict["info"]["debug"]:
        set_debug_mode(cfg_dict)
        cfg_dict["info"]["dir_name"] = "debug"
    
    train = Train(cfg_dict, None, None, True)

    dm = DialogAudioDM(**cfg_dict["data"])
    dm.setup(None)

    score_json_path = []
    id = datetime.datetime.now().strftime("%H%M%S")
    d = datetime.datetime.now().strftime("%Y_%m_%d")

    # Run
    for i in range(cfg_dict["train"]["trial_count"]):
        ### Preparation ###
        if cfg_dict["info"]["dir_name"] == None:
            output_dir = os.path.join(repo_root(), "output", d, id, str(i).zfill(2))
        else:
            output_dir = os.path.join(repo_root(), "output", cfg_dict["info"]["dir_name"], str(i).zfill(2))
        # output_dir = os.path.join(repo_root(), "output", d, id, str(i).zfill(2))
        output_path = os.path.join(output_dir, "model.pt")
        os.makedirs(output_dir, exist_ok=True)
        set_seed(i)

        ### Train ###
        train = Train(cfg_dict, dm, output_path, cfg_dict["train"]["verbose"])
        model = train.train()

        ### Test ###
        test = Test(cfg_dict, dm, output_path, output_dir, True)
        score, turn_taking_probs, probs, events = test.test()
        write_json(score, join(output_dir, "score.json"))

        print(f"Output Model -> {output_path}")
        print("Saved score -> ", join(output_dir, "score.json"))

        score_json_path.append(join(output_dir, "score.json"))

    if cfg_dict["info"]["dir_name"] == None:
        output_dir = os.path.join(repo_root(), "output", d, id)
    else:
        output_dir = os.path.join(repo_root(), "output", cfg_dict["info"]["dir_name"])
    df = compile_scores(score_json_path, output_dir)
    print("-" * 60)
    print(f"Output Final Score -> {join(output_dir, 'final_score.csv')}")
    print(df)
    print("-" * 60)
    write_json(cfg_dict, os.path.join(output_dir, "log.json")) # log file
    df.to_csv(join(output_dir, "final_score.csv"), index=False)


if __name__ == "__main__":
    main()
