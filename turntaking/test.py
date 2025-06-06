# coding: UTF-8
from omegaconf import DictConfig, OmegaConf
from os import makedirs
from os.path import join, dirname, basename, exists
import hydra
import torch
import json
import pandas as pd
import numpy as np
import datetime
import json
import os

from tqdm import tqdm

from turntaking.model import Model
from turntaking.utils import (
    everything_deterministic,
    set_seed,
    to_device,
    set_debug_mode,
    write_json,
    repo_root,
)
from turntaking.dataload import DialogAudioDM
from turntaking.evaluation import roc_to_threshold, events_plot, compute_classification_metrics, compute_regression_metrics, compute_comparative_metrics, compute_confusion_matrix
import warnings

everything_deterministic()
warnings.simplefilter("ignore")


class Test:
    def __init__(self, conf, dm, model_path, output_dir=None, find_threshold = False):
        self.conf = conf
        if output_dir == None:
            self.output_dir = dirname(model_path)
        else:
            self.output_dir = output_dir

        self.find_threshold = find_threshold

        self.dm = dm
        self.dm.change_frame_mode(True)

        self.model_path = model_path
        self.model = Model(conf).to(conf["train"]["device"])
        self.model.load_state_dict(
            torch.load(model_path, map_location=conf["train"]["device"])
        )

        self.result = None
        self.turn_taking_probs = None
        self.probs = None
        self.events = None

        self.model = self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False

        if exists(join(dirname(model_path), "thresholds.json")):
            if self.find_threshold:
                self.model = self._find_threshold()
                return

            with open(join(dirname(model_path), "thresholds.json")) as f:
                thresholds_dict = json.load(f)
                print(
                    "Loading thresholds: ", join(dirname(model_path), "thresholds.json")
                )
                self.thresholds = {
                    "shift_hold": torch.tensor(thresholds_dict["shift_hold"]),
                    "pred_shift": torch.tensor(thresholds_dict["pred_shift"]),
                    "pred_ov": torch.tensor(thresholds_dict["pred_ov"]),
                    "pred_bc": torch.tensor(thresholds_dict["pred_bc"]),
                    "short_long": torch.tensor(thresholds_dict["short_long"]),
                }
                print("-" * 60)
                print("### Thresholds ###")
                print(pd.DataFrame([self.thresholds]))
                print("-" * 60)
            self.model.test_metric = self.model.init_metric(
                threshold_shift_hold=self.thresholds.get("shift_hold", 0.5),
                threshold_pred_shift=self.thresholds.get("pred_shift", 0.3),
                threshold_pred_ov=self.thresholds.get("pred_ov", 0.3),
                threshold_short_long=self.thresholds.get("short_long", 0.5),
                threshold_bc_pred=self.thresholds.get("pred_bc", 0.1),
            )
        else:
            self.model = self._find_threshold()

    def test(self):
        test_loss = 0
        probs = []
        labels = []
        for ii, batch in tqdm(
            enumerate(self.dm.test_dataloader()),
            total=len(self.dm.test_dataloader()),
            dynamic_ncols=True,
            leave=False,
        ):
            # Forward Pass through the model
            loss, out, batch = self.model.shared_step(batch)
            test_loss += loss["total"]
            for o in out["logits_vp"]:
                probs.append(o)
            labels.append(out["va_labels"])

        probs = torch.cat(probs).unsqueeze(0).to(self.conf["train"]["device"])
        labels = torch.cat(labels).unsqueeze(0).to(self.conf["train"]["device"])
        torch.cuda.empty_cache()

        if self.conf["model"]["vap"]["type"] == "discrete":
            metrics = compute_classification_metrics(probs, labels)
            conf_matrix = compute_confusion_matrix(probs, labels)
            conf_matrix = pd.DataFrame(conf_matrix.cpu().numpy())
            print("Saved metrix -> ", join(self.output_dir, "metrix.csv"))
            conf_matrix.to_csv(join(self.output_dir, "metrix.csv"), index=True)
        elif self.conf["model"]["vap"]["type"] == "independent":
            metrics = compute_regression_metrics(probs, labels)
        else:
            metrics = compute_comparative_metrics(probs, labels)

        d = to_device(self.dm.get_full_sample("test"), self.conf["train"]["device"])
        events = self.model.test_metric.extract_events(va=d["vad"])
        turn_taking_probs = self.model.VAP(logits=probs, va=d["vad"])
        self.model.test_metric.update(
            p=turn_taking_probs["p"],
            bc_pred_probs=turn_taking_probs.get("bc_prediction", None),
            events=events,
        )

        test_loss /= len(self.dm.test_dataloader())
        events_score = self.model.test_metric.compute()
        result = {
            "test_loss": test_loss.item(),
            "shift_hold": events_score["f1_hold_shift"].item(),
            "short_long": events_score["f1_short_long"].item(),
            "short_long_0": events_score["f1_short_long_0"].item(),
            "short_long_1": events_score["f1_short_long_1"].item(),
            "shift_pred": events_score["f1_predict_shift"].item(),
            "shift_pred_0": events_score["f1_predict_shift_0"].item(),
            "shift_pred_1": events_score["f1_predict_shift_1"].item(),
            "ov_pred": events_score["f1_predict_ov"].item(),
            "ov_pred_0": events_score["f1_predict_ov_0"].item(),
            "ov_pred_1": events_score["f1_predict_ov_1"].item(),
            "bc_pred": events_score["f1_bc_prediction"].item(),
            "bc_pred_0": events_score["f1_bc_prediction_0"].item(),
            "bc_pred_1": events_score["f1_bc_prediction_1"].item(),
            "shift_f1": events_score["shift"]["f1"].item(),
            "shift_precision": events_score["shift"]["precision"].item(),
            "shift_recall": events_score["shift"]["recall"].item(),
            "hold_f1": events_score["hold"]["f1"].item(),
            "hold_precision": events_score["hold"]["precision"].item(),
            "hold_recall": events_score["hold"]["recall"].item(),
        }
        result |= metrics
        print("-" * 60)
        print("### Test ###")
        print("-" * 60)
        print(pd.DataFrame([result]))
        print("-" * 60)

        self.result = result
        self.turn_taking_probs = turn_taking_probs
        self.probs = probs
        self.events = events

        return result, turn_taking_probs, probs, events

    def _find_threshold(self):
        print("#" * 60)
        print("Finding Thresholds (val-set)...")
        print("#" * 60)
        thresholds, prediction, curves = roc_to_threshold(
            self.conf, self.model, self.dm
        )
        print("-" * 60)
        print("### Thresholds ###")
        print(pd.DataFrame([thresholds]))
        print("-" * 60)
        self.model.test_metric = self.model.init_metric(
            threshold_shift_hold=thresholds.get("shift_hold", 0.5),
            threshold_pred_shift=thresholds.get("pred_shift", 0.3),
            threshold_short_long=thresholds.get("short_long", 0.5),
            threshold_bc_pred=thresholds.get("pred_bc", 0.1),
        )

        ### Write ###
        th = {
            k: v.item() if isinstance(v, torch.Tensor) else v
            for k, v in thresholds.items()
        }
        write_json(th, join(self.output_dir, "thresholds.json"))
        torch.save(prediction, join(self.output_dir, "predictions.pt"))
        torch.save(curves, join(self.output_dir, "curves.pt"))
        print("Saved Thresholds -> ", join(self.output_dir, "thresholds.json"))
        print("Saved Curves -> ", join(self.output_dir, "curves.pt"))

        return self.model

    def _extract_data(self, data, keys, start_idx, end_idx, waveform_scale_factor=640):
        extracted_data = {}
        for key in keys:
            if key in {"waveform", "waveform_user1", "waveform_user2"}:
                extracted_data[key] = data[key][
                    :,
                    start_idx * waveform_scale_factor : end_idx * waveform_scale_factor,
                ]
            else:
                extracted_data[key] = data[key][:, start_idx:end_idx, :]
        return extracted_data

    def _extract_time_slice(self, cfg_dict, data, ii, time):
        time_slice = ii * cfg_dict["model"]["encoder"]["frame_hz"] * time
        next_time_slice = (ii + 1) * cfg_dict["model"]["encoder"]["frame_hz"] * time
        return time_slice, next_time_slice

    def img(self, time_duration=30):
        makedirs(join(dirname(self.model_path), "img"), exist_ok=True)
        data = to_device(self.dm.get_full_sample("test"), "cpu")

        data_keys = ["waveform_user1", "waveform_user2", "vad"]

        if self.conf["data"]["multimodal"]:
            data_keys.extend(
                [
                    "gaze_user1",
                    "au_user1",
                    "head_user1",
                    "pose_user1",
                    "gaze_user2",
                    "au_user2",
                    "head_user2",
                    "pose_user2",
                ]
            )

        prob_keys = ["p", "bc_prediction"]
        event_keys = ["shift", "predict_bc_pos"]

        for ii in tqdm(
            range(
                int(
                    data["vad"].size(-2)
                    / (time_duration * self.conf["model"]["encoder"]["frame_hz"])
                )
            )
        ):
            start_time, end_time = self._extract_time_slice(
                self.conf, data, ii, time_duration
            )

            # Prepare data for the current time frame
            d = self._extract_data(
                data,
                data_keys,
                start_time,
                end_time,
                waveform_scale_factor=int(
                    self.conf["model"]["encoder"]["sample_rate"]
                    / self.conf["model"]["encoder"]["frame_hz"]
                ),
            )

            # Prepare probabilities for the current time frame
            p = self._extract_data(
                self.turn_taking_probs, prob_keys, start_time, end_time
            )

            # Prepare events for the current time frame
            e = self._extract_data(self.events, event_keys, start_time, end_time)

            output_path = join(dirname(self.model_path), "img", f"{ii:04}.pdf")
            events_plot(
                d,
                p,
                e,
                output_path,
                audio_duration=self.conf["data"]["audio_duration"],
                frame_hz=self.conf["model"]["encoder"]["frame_hz"],
                sample_rate=self.conf["model"]["encoder"]["sample_rate"],
                # multimodal=cfg_dict["data"]["multimodal"],
                multimodal=False,
            )


# @hydra.main(config_path="conf", config_name="config", version_base=None)
# def main(cfg: DictConfig) -> None:
#     # def main():
#     cfg_dict = dict(OmegaConf.to_object(cfg))
#     debug = cfg_dict["info"]["debug"]

#     model_path = input("Please enter the path of the model.pt file: ")

#     img_input = input("Output images? (True/False): ").strip().lower()
#     img = img_input == "true"

#     with open(join(dirname(dirname(model_path)), "log.json")) as f:
#         cfg_dict = json.load(f)
#         if debug:
#             set_debug_mode(cfg_dict)

#     cfg_dict["num_workers"] = 0

#     dm = DialogAudioDM(**cfg_dict["data"])
#     dm.setup("test")
#     dm.change_frame_mode(True)

#     set_seed(int(basename(dirname(model_path))))

#     test = Test(cfg_dict, dm, model_path)
#     score, turn_taking_probs, probs, events = test.test()

#     if img:
#         test.img()

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
    debug = cfg_dict["info"]["debug"]
    device = cfg_dict["train"]["device"]
    model_dir_path = input("Please enter the path of the model file dir.: ")

    with open(join(model_dir_path, "log.json")) as f:
        cfg_dict = json.load(f)
        cfg_dict["train"]["device"] = device
        if debug:
            set_debug_mode(cfg_dict)

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
        model_path = os.path.join(model_dir_path, str(i).zfill(2), "model.pt")
        os.makedirs(output_dir, exist_ok=True)
        set_seed(i)

        ### Test ###
        test = Test(cfg_dict, dm, model_path, output_dir, True)
        score, turn_taking_probs, probs, events = test.test()
        write_json(score, join(output_dir, "score.json"))

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
    df.to_csv(join(output_dir, "final_score.csv"), index=False)


if __name__ == "__main__":
    main()
