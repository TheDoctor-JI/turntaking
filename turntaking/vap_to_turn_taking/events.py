from pprint import pprint

import numpy as np
import torch
from turntaking.vap_to_turntaking.backchannel import Backchannel
from turntaking.vap_to_turntaking.hold_shifts import HoldShift
from turntaking.vap_to_turntaking.utils import (find_island_idx_len,
                                                get_dialog_states,
                                                get_last_speaker,
                                                time_to_frames)
from turntaking.utils import set_seed


class TurnTakingEvents:
    def __init__(
        self,
        hs_kwargs,
        bc_kwargs,
        metric_kwargs,
        frame_hz=100,
        seed=42,
    ):
        self.frame_hz = frame_hz

        # Times to frames
        self.metric_kwargs = self.kwargs_to_frames(metric_kwargs, frame_hz)
        self.hs_kwargs = self.kwargs_to_frames(hs_kwargs, frame_hz)
        self.bc_kwargs = self.kwargs_to_frames(bc_kwargs, frame_hz)

        # values for metrics
        self.metric_min_context = self.metric_kwargs["min_context"]
        self.metric_pad = self.metric_kwargs["pad"]
        self.metric_dur = self.metric_kwargs["dur"]
        self.metric_onset_dur = self.metric_kwargs["onset_dur"]
        self.metric_pre_label_dur = self.metric_kwargs["pre_label_dur"]

        self.HS = HoldShift(**self.hs_kwargs)
        self.BS = Backchannel(**self.bc_kwargs)
        self.tt = None
        self.bcs = None
        self.seed = seed

    def kwargs_to_frames(self, kwargs, frame_hz):
        new_kwargs = {}
        for k, v in kwargs.items():
            new_kwargs[k] = time_to_frames(v, frame_hz)
        return new_kwargs

    def __repr__(self):
        s = "TurnTakingEvents\n"
        s += str(self.HS) + "\n"
        s += str(self.BS)
        return s

    def count_occurances(self, x):
        n = 0
        for b in range(x.shape[0]):
            for sp in [0, 1]:
                _, _, v = find_island_idx_len(x[b, :, sp])
                n += (v == 1).sum().item()
        return n
    
    # def count(self, x):
    #     counts = [0, 0]
    #     for b in range(x.shape[0]):
    #         for sp in [0, 1]:
    #             _, _, v = find_island_idx_len(x[b, :, sp])
    #             counts[sp] += (v == 1).sum().item()
    #     return counts

    def sample_negative_segments(self, x, n):
        """
        Used to pick a subset of negative segments.
        That is on events where the negatives are constrained in certain
        single chunk segments.

        Used to sample negatives for LONG/SHORT prediction.

        all start onsets result in either longer or shorter utterances.
        Utterances defined as backchannels are considered short and utterances
        after pauses or at shifts are considered long.

        """
        neg_candidates = []
        for b in range(x.shape[0]):
            for sp in [0, 1]:
                starts, durs, v = find_island_idx_len(x[b, :, sp])

                starts = starts[v == 1]
                durs = durs[v == 1]
                for s, d in zip(starts, durs):
                    neg_candidates.append([b, s, s + d, sp])

        sampled_negs = torch.arange(len(neg_candidates))
        if len(neg_candidates) > n:
            sampled_negs = np.random.choice(sampled_negs, size=n, replace=False)

        negs = torch.zeros_like(x)
        for ni in sampled_negs:
            b, s, e, sp = neg_candidates[ni]
            negs[b, s:e, sp] = 1.0

        return negs.float()

    def sample_negatives(self, x, n, dur):
        """

        Choose negative segments from x which contains long stretches of
        possible starts of the negative segments.

        Used to sample negatives from NON-SHIFTS which represent longer segments
        where every frame is a possible prediction point.

        """

        onset_pad_min = 3
        onset_pad_max = 10

        neg_candidates = []
        for b in range(x.shape[0]):
            for sp in [0, 1]:
                starts, durs, v = find_island_idx_len(x[b, :, sp])

                starts = starts[v == 1]
                durs = durs[v == 1]

                # Min context condition
                durs = durs[starts >= self.metric_min_context]
                starts = starts[starts >= self.metric_min_context]

                # Over minimum duration condition
                starts = starts[durs > dur]
                durs = durs[durs > dur]

                if len(starts) == 0:
                    continue

                for s, d in zip(starts, durs):
                    # end of valid frames minus duration of concurrent segment
                    end = s + d - dur

                    if end - s <= onset_pad_min:
                        onset_pad = 0
                    elif end - s <= onset_pad_max:
                        onset_pad = onset_pad_min
                    else:

                        onset_pad = torch.randint(onset_pad_min, onset_pad_max, (1,))[
                            0
                        ].item()

                    for neg_start in torch.arange(s + onset_pad, end, dur):
                        neg_candidates.append([b, neg_start, sp])

        sampled_negs = torch.arange(len(neg_candidates))
        if len(neg_candidates) > n:
            sampled_negs = np.random.choice(sampled_negs, size=n, replace=False)
            
        negs = torch.zeros_like(x)
        for ni in sampled_negs:
            b, s, sp = neg_candidates[ni]
            negs[b, s : s + dur, sp] = 1.0
        return negs.float()

    def __call__(self, vad, max_frame=None):
        set_seed(self.seed)
        ds = get_dialog_states(vad)
        last_speaker = get_last_speaker(vad, ds)
        # pprint(ds)
        # pprint(last_speaker)
        # TODO:
        # TODO: having all events as a list/dict with (b, start, end, speaker) may be very much faster?
        # TODO:

        # HOLDS/SHIFTS:
        # shift, pre_shift, long_shift_onset,
        # hold, pre_hold, long_hold_onset,
        # shift_overlap, pre_shift_overlap, non_shift
        self.tt = self.HS(
            vad=vad, ds=ds, max_frame=max_frame, min_context=self.metric_min_context
        )
        # pprint(self.tt)

        # Backchannels: backchannel, pre_backchannel
        self.bcs = self.BS(
            vad=vad,
            last_speaker=last_speaker,
            max_frame=max_frame,
            min_context=self.metric_min_context,
        )

        #######################################################
        # LONG/SHORT
        #######################################################
        # Investigate the model output at the start of an IPU
        # where SHORT segments are "backchannel" and LONG har onset on new TURN (SHIFTs)
        # or onset of HOLD ipus
        short = self.bcs["backchannel"]
        long = self.sample_negative_segments(self.tt["long_shift_onset"], 1000)

        #######################################################
        # Predict shift
        #######################################################
        # Pos: window, on activity, prior to EOT before a SHIFT
        # Neg: Sampled from NON-SHIFT, on activity.
        n_predict_shift = self.count_occurances(self.tt["pre_shift"])
        if n_predict_shift == 0:
            predict_shift_neg = torch.zeros_like(self.tt["pre_shift"])
        else:
            # NON-SHIFT where someone is active
            activity = ds == 0  # only A
            activity = torch.logical_or(activity, ds == 3)  # AND only B
            activity = activity[:, : self.tt["non_shift"].shape[1]].unsqueeze(-1)
            non_shift_on_activity = torch.logical_and(self.tt["non_shift"], activity)
            predict_shift_neg = self.sample_negatives(
                non_shift_on_activity, n_predict_shift, dur=self.metric_pre_label_dur
            )

        #######################################################
        # Predict backchannels
        #######################################################
        # Pos: 0.5 second prior a backchannel
        # Neg: Sampled from NON-SHIFT, everywhere
        n_pre_bc = self.count_occurances(self.bcs["pre_backchannel"])
        if n_pre_bc == 0:
            predict_bc_neg = torch.zeros_like(self.bcs["pre_backchannel"])
        else:
            predict_bc_neg = self.sample_negatives(
                self.tt["non_shift"], n_pre_bc, dur=self.metric_pre_label_dur
            )
        
        #######################################################
        # Predict shift-overlap
        #######################################################
        # Pos: window, on activity, prior a overlap-SHIFT
        # Neg: Sampled from NON-SHIFT, on activity.
        n_predict_shift_ov = self.count_occurances(self.tt["pre_shift_ov_oh"])
        if n_predict_shift_ov == 0:
            predict_shift_ov_neg = torch.zeros_like(self.tt["pre_shift_ov_oh"])
        else:
            # NON-SHIFT where someone is active
            activity = ds == 0  # only A
            activity = torch.logical_or(activity, ds == 3)  # AND only B
            activity = activity[:, : self.tt["non_shift"].shape[1]].unsqueeze(-1)
            non_shift_ov_on_activity = torch.logical_and(self.tt["non_shift"], activity)
            predict_shift_ov_neg = self.sample_negatives(
                non_shift_ov_on_activity, n_predict_shift_ov, dur=self.metric_pre_label_dur
            )

        # pprint(self.bcs["pre_backchannel"][0])
        # pprint(n_pre_bc)
        # pprint(self.tt["shift"].shape)
        # return self.tt
        return {
            "shift": self.tt["shift"][:, :max_frame],
            "hold": self.tt["hold"][:, :max_frame],
            "short": short[:, :max_frame],
            "long": long[:, :max_frame],
            "predict_shift_pos": self.tt["pre_shift"][:, :max_frame],
            "predict_shift_neg": predict_shift_neg[:, :max_frame],
            "predict_bc_pos": self.bcs["pre_backchannel"][:, :max_frame],
            "predict_bc_neg": predict_bc_neg[:, :max_frame],
            "predict_shift_ov_pos": self.tt["pre_shift_ov_oh"][:, :max_frame],
            "predict_shift_ov_neg": predict_shift_ov_neg[:, :max_frame],
        }


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from turntaking.vap_to_turntaking.config.example_data import (event_conf,
                                                                  example)
    from turntaking.vap_to_turntaking.plot_utils import plot_event, plot_vad_oh

    eventer = TurnTakingEvents(
        hs_kwargs=event_conf["hs"],
        bc_kwargs=event_conf["bc"],
        metric_kwargs=event_conf["metric"],
        frame_hz=100,
    )
    pprint(event_conf["hs"])
    pprint(event_conf["bc"])
    va = example["va"]
    print(va)
    events = eventer(va, max_frame=None)
    print("long: ", (events["long"] != example["long"]).sum())
    print("short: ", (events["short"] != example["short"]).sum())
    print("shift: ", (events["shift"] != example["shift"]).sum())
    print("hold: ", (events["hold"] != example["hold"]).sum())
    # for k, v in events.items():
    #     if isinstance(v, torch.Tensor):
    #         print(f"{k}: {tuple(v.shape)}")
    #     else:
    #         print(f"{k}: {v}")

    fig, ax = plot_vad_oh(va[0])
    # _, ax = plot_event(events["shift"][0], ax=ax)
    # _, ax = plot_event(events["hold"][0], color=["r", "r"], ax=ax)
    # _, ax = plot_event(events["short"][0], ax=ax)
    # _, ax = plot_event(events["long"][0], color=["r", "r"], ax=ax)
    _, ax = plot_event(events["predict_shift_pos"][0], ax=ax)
    _, ax = plot_event(events["predict_shift_neg"][0], color=["r", "r"], ax=ax)
    # _, ax = plot_event(events["predict_bc_pos"][0], ax=ax)
    # _, ax = plot_event(events["predict_bc_neg"][0], color=["r", "r"], ax=ax)

    # _, ax = plot_event(example["shift"][0], ax=ax)
    # _, ax = plot_event(example["hold"][0], color=["r", "r"], ax=ax)
    # _, ax = plot_event(example["short"][0], ax=ax)
    # _, ax = plot_event(example["long"][0], color=["r", "r"], ax=ax)

    # plt.pause(0.1)
    plt.savefig("/ahc/work2/kazuyo-oni/hoge.png")
