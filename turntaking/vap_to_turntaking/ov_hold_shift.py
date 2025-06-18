from pprint import pprint

import torch
from turntaking.vap_to_turntaking.utils import (find_island_idx_len,
                                                get_dialog_states,
                                                get_last_speaker)

class OverlapHoldShift:
    """
    Overlapping Hold/Shift extraction from VAD. Operates on Frames.

    Detects events where:
    - One speaker is originally speaking (A)
    - Other speaker starts speaking causing overlap (B joins)
    - Either A stops and B continues (shift) or A continues and B stops (hold)

    Arguments:
        post_onset_shift:           int, frames for shift onset condition
        pre_offset_shift:           int, frames for shift offset condition  
        post_onset_hold:            int, frames for hold onset condition
        pre_offset_hold:            int, frames for hold offset condition
        metric_pad:                 int, pad on overlap onset used for evaluating
        metric_dur:                 int, duration of overlap used for evaluating
        metric_pre_label_dur:       int, frames prior to overlap for prediction
        metric_onset_dur:           int, frames for onset detection
        min_overlap:                int, minimum overlap duration to consider

    Return:
        dict: {'ov_shift', 'pre_ov_shift', 'ov_hold', 'pre_ov_hold'}

    Active: "---"
    Overlap: "==="

    # OVERLAPPING SHIFTS
    
    onset:                                 |<-- only B -->|
    A:          ------------------=========...............
    B:          ..................=========---------------
    offset:     |<--  only A -->|
    OV_SHIFT:                    |=========|

    -----------------------------------------------------------
    # OVERLAPPING HOLDS

    onset:                                 |<-- only A -->|
    A:          ------------------=========---------------
    B:          ..................=========...............
    offset:     |<--  only A -->|
    OV_HOLD:                     |=========|

    -----------------------------------------------------------

    Using 'dialog states' consisting of 4 different states:
    0. Only A is speaking
    1. Silence  
    2. Overlap
    3. Only B is speaking

    Overlap Shift:   0 -> 2 -> 3          3 -> 2 -> 0
    Overlap Hold:    0 -> 2 -> 0          3 -> 2 -> 3
    
    For filling gaps within overlaps:
    Overlap Fill:    2 -> non-overlap -> 2  (where non-overlap could be 0, 1, or 3)
    """

    def __init__(
        self,
        post_onset_shift,
        pre_offset_shift,
        post_onset_hold,
        pre_offset_hold,
        metric_pad,
        metric_dur,
        metric_pre_label_dur,
        metric_onset_dur,
        min_overlap,
    ):
        assert (
            metric_onset_dur <= post_onset_shift
        ), "`metric_onset_dur` must be less or equal to `post_onset_shift`"

        self.post_onset_shift = post_onset_shift
        self.pre_offset_shift = pre_offset_shift
        self.post_onset_hold = post_onset_hold
        self.pre_offset_hold = pre_offset_hold

        self.metric_pad = metric_pad
        self.metric_dur = metric_dur
        self.min_overlap = max(metric_pad + metric_dur, min_overlap)
        self.metric_pre_label_dur = metric_pre_label_dur
        self.metric_onset_dur = metric_onset_dur

        # Templates for overlap-based events
        # [prev_state, overlap_state, next_state]
        self.ov_shift_template = torch.tensor([[0, 2, 3], [3, 2, 0]])  # A->overlap->B, B->overlap->A
        self.ov_hold_template = torch.tensor([[0, 2, 0], [3, 2, 3]])   # A->overlap->A, B->overlap->B
        
        # Template for filling regular hold patterns (same as hold_shift.py)
        self.hold_template = torch.tensor([[0, 1, 0], [3, 1, 3]])      # A->silence->A, B->silence->B
        
        # Templates for filling gaps within overlaps
        # These handle cases where overlap is briefly interrupted
        self.overlap_fill_templates = [
            torch.tensor([[2, 0, 2]]),  # overlap -> A_only -> overlap
            torch.tensor([[2, 1, 2]]),  # overlap -> silence -> overlap  
            torch.tensor([[2, 3, 2]]),  # overlap -> B_only -> overlap
        ]

    def __repr__(self):
        s = "Overlapping Holds & Shifts"
        s += f"\n  post_onset_shift: {self.post_onset_shift}"
        s += f"\n  pre_offset_shift: {self.pre_offset_shift}"
        s += f"\n  post_onset_hold: {self.post_onset_hold}"
        s += f"\n  pre_offset_hold: {self.pre_offset_hold}"
        s += f"\n  min_overlap: {self.min_overlap}"
        s += f"\n  metric_pad: {self.metric_pad}"
        s += f"\n  metric_dur: {self.metric_dur}"
        s += f"\n  metric_pre_label_dur: {self.metric_pre_label_dur}"
        return s

    def fill_template(self, vad, ds, template):
        """
        Fill gaps based on template patterns.
        
        For regular hold patterns (A->silence->A), fill with that speaker's activity.
        For overlap fill patterns (overlap->gap->overlap), fill with both speakers active.
        """
        filled_vad = vad.clone()
        
        for b in range(ds.shape[0]):
            s, d, v = find_island_idx_len(ds[b])
            if len(v) < 3:
                continue

            triads = v.unfold(0, size=3, step=1)
            next_speaker, steps = torch.where(
                (triads == template.unsqueeze(1)).sum(-1) == 3
            )

            for ns, pre in zip(next_speaker, steps):
                cur = pre + 1  # middle segment
                
                # Check if this is an overlap fill template
                is_overlap_fill = template.shape[0] == 1 and template[0, 0] == 2 and template[0, -1] == 2
                
                if is_overlap_fill:
                    # Fill gap within overlap - mark both speakers as active
                    filled_vad[b, s[cur] : s[cur] + d[cur], 0] = 1.0
                    filled_vad[b, s[cur] : s[cur] + d[cur], 1] = 1.0
                else:
                    # Regular hold pattern - fill with the specific speaker
                    filled_vad[b, s[cur] : s[cur] + d[cur], ns] = 1.0

        return filled_vad

    def apply_all_fills(self, vad, ds):
        """
        Apply all filling templates in sequence:
        1. Regular hold patterns (A->silence->A, B->silence->B)
        2. Overlap fill patterns (overlap->gap->overlap)
        """
        filled_vad = vad.clone()
        
        # First fill regular hold patterns (same as hold_shift.py)
        filled_vad = self.fill_template(filled_vad, ds, self.hold_template)
        
        # Then fill gaps within overlaps
        for overlap_template in self.overlap_fill_templates:
            filled_vad = self.fill_template(filled_vad, ds, overlap_template)
            
        return filled_vad

    def match_template(
        self,
        vad,
        ds,
        template,
        pre_cond_frames,
        post_cond_frames,
        pre_match=False,
        onset_match=False,
        max_frame=None,
        min_context=0,
    ):
        """
        Creates a onehot vector where overlap events matching the template occur.
        
        Returns:
            match_oh:       torch.Tensor (B, N, 2), where last dim is next speaker
            pre_match_oh:   torch.Tensor (B, N, 2), pre-event prediction window
            onset_match_oh: torch.Tensor (B, N, 2), onset detection window
            event_location: torch.Tensor (B, N, 2), full event span
        """
        
        # Check if this is a hold template (same speaker before and after overlap)
        hold_cond = template[0, 0] == template[0, -1]

        match_oh = torch.zeros((*ds.shape, 2), device=ds.device, dtype=torch.float)
        event_location = torch.zeros((*ds.shape, 2), device=ds.device, dtype=torch.float)

        pre_match_oh = None
        if pre_match:
            pre_match_oh = torch.zeros(
                (*ds.shape, 2), device=ds.device, dtype=torch.float
            )

        onset_match_oh = None
        if onset_match:
            onset_match_oh = torch.zeros(
                (*ds.shape, 2), device=ds.device, dtype=torch.float
            )

        for b in range(ds.shape[0]):
            s, d, v = find_island_idx_len(ds[b])
            if len(v) < 3:
                continue

            # Match overlap-based templates
            triads = v.unfold(0, size=3, step=1)
            next_speaker, steps = torch.where(
                (triads == template.unsqueeze(1)).sum(-1) == 3
            )

            for ns, pre_step in zip(next_speaker, steps):
                # Determine speakers
                nos = 0 if ns == 1 else 1  # other next speaker
                ps = ns if hold_cond else nos  # previous speaker

                cur = pre_step + 1  # overlap segment
                post = pre_step + 2  # post-overlap segment

                # Overlap condition: must be overlap (state 2) and meet minimum duration
                if v[cur] != 2 or d[cur] < self.min_overlap:
                    continue

                # Pre-condition: previous speaker was exclusively active
                pre_start = s[cur] - pre_cond_frames
                if pre_start < 0:
                    continue

                pre_cond1 = vad[b, pre_start : s[cur], ps].sum() == pre_cond_frames
                not_ps = 0 if ps == 1 else 1
                pre_cond2 = vad[b, pre_start : s[cur], not_ps].sum() == 0
                pre_cond = torch.logical_and(pre_cond1, pre_cond2)

                if not pre_cond:
                    continue

                # Post-condition: next speaker becomes exclusively active
                post_start = s[post]
                post_end = post_start + post_cond_frames
                if post_end > vad.shape[1]:
                    continue

                post_cond1 = vad[b, post_start:post_end, ns].sum() == post_cond_frames
                post_cond2 = vad[b, post_start:post_end, nos].sum() == 0
                post_cond = torch.logical_and(post_cond1, post_cond2)

                if not post_cond:
                    continue

                # Frame boundary checks
                if max_frame is not None and s[cur] >= max_frame:
                    continue

                if (s[cur] + self.metric_pad) < min_context:
                    continue

                # Set pre-match window (before overlap starts)
                if pre_match:
                    pre_start_window = s[cur] - self.metric_pre_label_dur
                    if pre_start_window >= 0:
                        pre_match_oh[b, pre_start_window : s[cur], ns] = 1.0

                # Set main event window (during overlap, with padding)
                # From this we can tell that, even if the overlap is long, the event duration only covers the first self.metric_dur frames (of course including the pad frames)
                end = s[cur] + self.metric_pad + self.metric_dur
                if max_frame is not None and end >= max_frame:
                    continue

                match_oh[b, s[cur] + self.metric_pad : end, ns] = 1.0
                event_location[b, s[cur] : s[post], ns] = 1.0

                # Set onset window (when next speaker takes over)
                if onset_match:
                    onset_end = s[post] + self.metric_onset_dur
                    if max_frame is None or onset_end < max_frame:
                        onset_match_oh[b, s[post] : onset_end, ns] = 1.0

        return match_oh, pre_match_oh, onset_match_oh, event_location

    def __call__(
        self,
        vad,
        ds=None,
        filled_vad=None,
        max_frame=None,
        min_context=0,
        return_list=False,
    ):
        
        if ds is None:
            ds = get_dialog_states(vad)

        if vad.device != self.ov_hold_template.device:
            self.ov_shift_template = self.ov_shift_template.to(vad.device)
            self.ov_hold_template = self.ov_hold_template.to(vad.device)
            self.hold_template = self.hold_template.to(vad.device)
            for i, template in enumerate(self.overlap_fill_templates):
                self.overlap_fill_templates[i] = template.to(vad.device)

        # Apply comprehensive filling if not provided
        if filled_vad is None:
            filled_vad = self.apply_all_fills(vad, ds)

        # Extract overlapping shifts
        ov_shift_oh, pre_ov_shift_oh, long_ov_shift_onset, ov_shift_dur = self.match_template(
            filled_vad,
            ds,
            self.ov_shift_template,
            pre_cond_frames=self.pre_offset_shift,
            post_cond_frames=self.post_onset_shift,
            pre_match=True,
            onset_match=True,
            max_frame=max_frame,
            min_context=min_context,
        )

        # Extract overlapping holds
        ov_hold_oh, pre_ov_hold_oh, long_ov_hold_onset, ov_hold_dur = self.match_template(
            filled_vad,
            ds,
            self.ov_hold_template,
            pre_cond_frames=self.pre_offset_hold,
            post_cond_frames=self.post_onset_hold,
            pre_match=True,
            onset_match=True,
            max_frame=max_frame,
            min_context=min_context,
        )

        return {
            "ov_shift": ov_shift_oh,
            "pre_ov_shift": pre_ov_shift_oh,
            "long_ov_shift_onset": long_ov_shift_onset,
            "ov_hold": ov_hold_oh,
            "pre_ov_hold": pre_ov_hold_oh,
            "long_ov_hold_onset": long_ov_hold_onset,
            "ov_shift_dur": ov_shift_dur,
            "ov_hold_dur": ov_hold_dur,
        }


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from turntaking.vap_to_turntaking.config.example_data import (
        event_conf_frames, example)
    from turntaking.vap_to_turntaking.plot_utils import plot_event, plot_vad_oh

    plt.close("all")
    va = example["va"]

    # Example configuration for overlapping hold/shift detection
    metric_kwargs = dict(
        pad=5,
        dur=10,
        pre_label_dur=50,
        onset_dur=20,
        min_context=30,
        min_overlap=15,  # minimum overlap duration
    )
    
    ovhs_kwargs = dict(
        post_onset_shift=100,
        pre_offset_shift=100,
        post_onset_hold=100,
        pre_offset_hold=100,
        metric_pad=metric_kwargs["pad"],
        metric_dur=metric_kwargs["dur"],
        metric_pre_label_dur=metric_kwargs["pre_label_dur"],
        metric_onset_dur=metric_kwargs["onset_dur"],
        min_overlap=metric_kwargs["min_overlap"],
    )

    pprint(ovhs_kwargs)
    OVHS = OverlapHoldShift(**ovhs_kwargs)
    ov_events = OVHS(example["va"], max_frame=None)
    
    print("Overlapping Hold/Shift Events:")
    for k, v in ov_events.items():
        if isinstance(v, torch.Tensor):
            print(f"{k}: {tuple(v.shape)} - sum: {v.sum().item()}")
        else:
            print(f"{k}: {v}")

    # Visualization
    fig, ax = plot_vad_oh(va[0])
    _, ax = plot_event(ov_events["ov_shift"][0], ax=ax, color=["blue", "blue"], alpha=0.7)
    _, ax = plot_event(ov_events["ov_hold"][0], ax=ax, color=["orange", "orange"], alpha=0.7)
    plt.title("Overlapping Hold/Shift Events")
    plt.savefig("./overlapping_hold_shift_events.png")
    print("Visualization saved to ./overlapping_hold_shift_events.png")