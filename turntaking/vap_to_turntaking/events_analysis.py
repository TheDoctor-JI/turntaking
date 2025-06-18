from pprint import pprint

import numpy as np
import matplotlib.pyplot as plt
import torch
import pandas as pd
from collections import defaultdict
import hydra
from omegaconf import DictConfig, OmegaConf 
from tqdm import tqdm

from turntaking.vap_to_turntaking.backchannel import Backchannel
from turntaking.vap_to_turntaking.hold_shifts import HoldShift
from turntaking.vap_to_turntaking.events import TurnTakingEvents
from turntaking.vap_to_turntaking.utils import (find_island_idx_len,
                                                get_dialog_states,
                                                get_last_speaker,
                                                time_to_frames)
from turntaking.dataload import DialogAudioDM
from turntaking.dataload.utils import (
    load_waveform,
    get_audio_info,
    time_to_frames,
    load_multimodal_features,
)
from turntaking.vap_to_turntaking.utils import vad_list_to_onehot, get_activity_history
from turntaking.utils import (
    everything_deterministic,
    set_seed,
    to_device,
    set_debug_mode,
    write_json,
    repo_root,
)

from decimal import Decimal
from os.path import join, exists
import re
import wave

from turntaking.dataload.utils import read_txt

import os
import soundfile as sf
from turntaking.vap_to_turntaking.plot_utils import plot_vad_oh, plot_event



FRAME_HZ = 100
HOT_TIME = 1/FRAME_HZ

HIST = False
DATA = 0

def count_occurances(x):
    n = 0
    for b in range(x.shape[0]):
        for sp in [0, 1]:
            _, _, v = find_island_idx_len(x[b, :, sp])
            n += (v == 1).sum().item()
    return n

def find_continuous_ones(tensor):
    def calculate(x):
        differences = {}
        for key in x:
            differences[key] = [round((x[key][i][1] - x[key][i][0] + 1) * HOT_TIME, 3) for i in range(len(x[key]))]
            # for i in range(len(x[key])):
            #     if round((x[key][i][1] - x[key][i][0] + 1) * HOT_TIME, 3) > 10:
            #         print(f"{round((x[key][i][0] + 1) * HOT_TIME, 3)} to {round((x[key][i][1])* HOT_TIME, 3)} ({round((x[key][i][1] - x[key][i][0] + 1) * HOT_TIME, 3)})")
        return differences
    
    continuous_segments = {0: [], 1: []}
    for dim in range(2):
        start = None
        for i, value in enumerate(tensor[0, :, dim]):
            if value == 1 and start is None:
                start = i
            elif value == 0 and start is not None:
                continuous_segments[dim].append((start, i - 1))
                start = None
        if start is not None:
            continuous_segments[dim].append((start, len(tensor[0, :, dim]) - 1))

    return calculate(continuous_segments)

def calculate_statistics(array):
    return {
        'num': len(array),
        'mean': round(np.mean(array), 3),
        'median': round(np.median(array), 3),
        'max': round(np.max(array), 3),
        'min': round(np.min(array), 3),
        'variance': round(np.var(array), 3)
    }

def plot_event_with_audio_context(
    vad, 
    event_tensor, 
    event_type, 
    session_id, 
    speaker_idx, 
    event_idx, 
    output_dir,
    context_frames=500,  # ~5 seconds at 100Hz
    audio_path=None,
    sample_rate=16000
):
    """
    Plot VAD and event with audio context around the event.
    
    Args:
        vad: VAD tensor (N, 2)
        event_tensor: Event tensor (N, 2) 
        event_type: String identifier for event type
        session_id: Session identifier
        speaker_idx: Speaker index (0 or 1)
        event_idx: Index of this event occurrence
        output_dir: Directory to save plots
        context_frames: Frames of context around event
        audio_path: Path to audio file (optional)
        sample_rate: Audio sample rate
    """
    import matplotlib.pyplot as plt
    
    # Find event boundaries
    event_frames = torch.where(event_tensor[:, speaker_idx] == 1)[0]
    if len(event_frames) == 0:
        return
        
    start_frame = event_frames[0].item()
    end_frame = event_frames[-1].item()
    
    # Define context window
    context_start = max(0, start_frame - context_frames)
    context_end = min(vad.shape[0], end_frame + context_frames)
    
    # Extract context
    vad_context = vad[context_start:context_end]
    event_context = event_tensor[context_start:context_end]
    
    # Create plot
    fig, axes = plt.subplots(2, 1, figsize=(15, 8), sharex=True)
    
    # Plot VAD
    time_frames = torch.arange(context_end - context_start)
    axes[0].fill_between(time_frames, 0, vad_context[:, 0], alpha=0.7, label='Speaker A', color='blue')
    axes[0].fill_between(time_frames, 0, -vad_context[:, 1], alpha=0.7, label='Speaker B', color='red')
    axes[0].set_ylabel('Voice Activity')
    axes[0].legend()
    axes[0].set_title(f'{event_type} Event - {session_id} - Speaker {speaker_idx} - Event #{event_idx}')
    
    # Plot event overlay
    event_frames_context = event_context[:, speaker_idx]
    axes[1].fill_between(time_frames, 0, event_frames_context, alpha=0.8, 
                        color='green' if 'shift' in event_type.lower() else 'orange',
                        label=f'{event_type} Event')
    axes[1].set_ylabel('Event Activity')
    axes[1].set_xlabel('Frames (10ms each)')
    axes[1].legend()
    
    # Mark actual event boundaries
    event_start_in_context = start_frame - context_start
    event_end_in_context = end_frame - context_start
    for ax in axes:
        ax.axvline(event_start_in_context, color='black', linestyle='--', alpha=0.7)
        ax.axvline(event_end_in_context, color='black', linestyle='--', alpha=0.7)
    
    # Save plot
    os.makedirs(output_dir, exist_ok=True)
    filename = f"{output_dir}/{event_type}_{session_id}_spk{speaker_idx}_evt{event_idx}.png"
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()
    
    return {
        'plot_path': filename,
        'event_start_frame': start_frame,
        'event_end_frame': end_frame,
        'context_start_frame': context_start,
        'context_end_frame': context_end,
        'duration_seconds': (end_frame - start_frame) * 0.01,  # 10ms per frame
        'audio_path': audio_path
    }

def extract_audio_segment(audio_path, start_frame, end_frame, context_frames, output_path, sample_rate=16000, frame_hz=100):
    """
    Extract audio segment corresponding to event with context.
    """
    if not os.path.exists(audio_path):
        return None
        
    try:
        # Convert frames to samples
        frames_per_second = frame_hz
        samples_per_frame = sample_rate // frames_per_second
        
        context_start = max(0, start_frame - context_frames)
        context_end = end_frame + context_frames
        
        start_sample = context_start * samples_per_frame
        end_sample = context_end * samples_per_frame
        
        # Load and extract audio
        audio, sr = sf.read(audio_path, start=start_sample, stop=end_sample)
        
        # Save extracted segment
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        sf.write(output_path, audio, sr)
        
        return output_path
    except Exception as e:
        print(f"Error extracting audio: {e}")
        return None

def get_audio_path_for_session(session_id, dm, split="all"):
    """
    Get audio path for a given session from the dataset.
    
    Args:
        session_id: Session identifier
        dm: DialogAudioDM instance
        split: Which split to check ("train", "val", "test", or "all")
    """
    datasets_to_check = []
    
    if split == "all" or split == "train":
        if dm.train_dset is not None:
            datasets_to_check.append(dm.train_dset)
    if split == "all" or split == "val": 
        if dm.val_dset is not None:
            datasets_to_check.append(dm.val_dset)
    if split == "all" or split == "test":
        if dm.test_dset is not None:
            datasets_to_check.append(dm.test_dset)
    
    for dset in datasets_to_check:
        # Check if session exists in this dataset
        if session_id in dset.data["session"]:
            session_idx = dset.data["session"].index(session_id)
            
            # Get the raw dataset entry to access audio paths
            raw_data = dset.dataset[session_idx]
            
            return {
                'path': raw_data['audio_path'],
                'user1_path': raw_data.get('user1_audio_path'),
                'user2_path': raw_data.get('user2_audio_path'),
                'dataset_name': raw_data['dataset_name']
            }
    
    return None

def analyze_ovhs_events(vad, session_id, eventer, output_dir, audio_info=None):
    """
    Analyze and visualize overlapping hold/shift events for a single session.
    
    Args:
        vad: VAD tensor for the session
        session_id: Session identifier
        eventer: TurnTakingEvents instance
        output_dir: Output directory for plots and audio
        audio_info: Dictionary with audio file paths and metadata
    """
    if eventer.OVHS is None:
        print("No overlapping hold/shift detector available")
        return []
    
    # Get overlapping events
    events = eventer(vad, max_frame=None)
    
    if "ov_shift" not in events and "ov_hold" not in events:
        return []
    
    results = []
    
    # Process overlapping shifts
    if "ov_shift" in events:
        ov_shift_tensor = events["ov_shift"][0]  # Assuming batch size 1
        
        for speaker_idx in [0, 1]:
            # Find individual event occurrences
            speaker_events = ov_shift_tensor[:, speaker_idx]
            starts, durations, values = find_island_idx_len(speaker_events)
            
            event_starts = starts[values == 1]
            event_durations = durations[values == 1]
            
            for event_idx, (start, duration) in enumerate(zip(event_starts, event_durations)):
                # Create event-specific tensor for plotting
                event_tensor = torch.zeros_like(ov_shift_tensor)
                event_tensor[start:start+duration, speaker_idx] = 1
                
                # Plot event
                plot_info = plot_event_with_audio_context(
                    vad=vad[0],  # Assuming batch size 1
                    event_tensor=event_tensor,
                    event_type="OV_SHIFT",
                    session_id=session_id,
                    speaker_idx=speaker_idx,
                    event_idx=event_idx,
                    output_dir=f"{output_dir}/ov_shift_plots",
                    audio_path=audio_info.get('path') if audio_info else None
                )
                
                # Extract audio if available
                audio_segment_path = None
                if audio_info and os.path.exists(audio_info['path']):
                    audio_output_path = f"{output_dir}/ov_shift_audio/{session_id}_spk{speaker_idx}_evt{event_idx}.wav"
                    audio_segment_path = extract_audio_segment(
                        audio_path=audio_info['path'],
                        start_frame=start,
                        end_frame=start + duration,
                        context_frames=500,
                        output_path=audio_output_path
                    )
                
                result = {
                    'session_id': session_id,
                    'event_type': 'ov_shift',
                    'speaker': speaker_idx,
                    'event_index': event_idx,
                    'start_frame': start.item(),
                    'end_frame': (start + duration).item(),
                    'duration_frames': duration.item(),
                    'duration_seconds': duration.item() * 0.01,
                    'plot_path': plot_info['plot_path'] if plot_info else None,
                    'audio_path': audio_segment_path
                }
                results.append(result)
    
    # Process overlapping holds (similar structure)
    if "ov_hold" in events:
        ov_hold_tensor = events["ov_hold"][0]
        
        for speaker_idx in [0, 1]:
            speaker_events = ov_hold_tensor[:, speaker_idx]
            starts, durations, values = find_island_idx_len(speaker_events)
            
            event_starts = starts[values == 1]
            event_durations = durations[values == 1]
            
            for event_idx, (start, duration) in enumerate(zip(event_starts, event_durations)):
                event_tensor = torch.zeros_like(ov_hold_tensor)
                event_tensor[start:start+duration, speaker_idx] = 1
                
                plot_info = plot_event_with_audio_context(
                    vad=vad[0],
                    event_tensor=event_tensor,
                    event_type="OV_HOLD",
                    session_id=session_id,
                    speaker_idx=speaker_idx,
                    event_idx=event_idx,
                    output_dir=f"{output_dir}/ov_hold_plots",
                    audio_path=audio_info.get('path') if audio_info else None
                )
                
                audio_segment_path = None
                if audio_info and os.path.exists(audio_info['path']):
                    audio_output_path = f"{output_dir}/ov_hold_audio/{session_id}_spk{speaker_idx}_evt{event_idx}.wav"
                    audio_segment_path = extract_audio_segment(
                        audio_path=audio_info['path'],
                        start_frame=start,
                        end_frame=start + duration,
                        context_frames=500,
                        output_path=audio_output_path
                    )
                
                result = {
                    'session_id': session_id,
                    'event_type': 'ov_hold',
                    'speaker': speaker_idx,
                    'event_index': event_idx,
                    'start_frame': start.item(),
                    'end_frame': (start + duration).item(),
                    'duration_frames': duration.item(),
                    'duration_seconds': duration.item() * 0.01,
                    'plot_path': plot_info['plot_path'] if plot_info else None,
                    'audio_path': audio_segment_path
                }
                results.append(result)
    
    return results

@hydra.main(config_path="../conf", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    cfg_dict = dict(OmegaConf.to_object(cfg))
    cfg_dict["data"]["vad_hz"] = FRAME_HZ
    cfg_dict["data"]["oversampling"] = False
    cfg_dict["data"]["flip_channels"] = False
    cfg_dict["data"]["vad_history"] = False

    if DATA is not None:
        cfg_dict["data"]["train_files"] = f"turntaking/dataload/dataset/{cfg_dict['data']['datasets']}/files/train.txt"
        # cfg_dict["data"]["train_files"] = f"turntaking/dataload/dataset/{cfg_dict['data']['datasets']}/files/train_{DATA}.txt"

        cfg_dict["data"]["val_files"] = f"turntaking/dataload/dataset/{cfg_dict['data']['datasets']}/files/val.txt"
        # cfg_dict["data"]["val_files"] = f"turntaking/dataload/dataset/{cfg_dict['data']['datasets']}/files/val_{DATA}.txt"

        cfg_dict["data"]["test_files"] = f"turntaking/dataload/dataset/{cfg_dict['data']['datasets']}/files/test.txt"  
        # cfg_dict["data"]["test_files"] = f"turntaking/dataload/dataset/{cfg_dict['data']['datasets']}/files/test_{DATA}.txt"  

    cfg_dict["events"]["SH"]["metric_pad"] = 0.0
    cfg_dict["events"]["SH"]["metric_dur"] = 0.0

    dm = DialogAudioDM(**cfg_dict["data"])
    dm.setup(None)

    eventer = TurnTakingEvents(
        hs_kwargs=cfg_dict["events"]["SH"],
        bc_kwargs=cfg_dict["events"]["BC"],
        metric_kwargs=cfg_dict["events"]["metric"],
        ovhs_kwargs=cfg_dict["events"].get("OVHS", None),  # NEW: Add OVHS kwargs
        frame_hz=FRAME_HZ,
    )

    results = []
    all_shift0 = []
    all_shift1 = []
    all_hold0 = []
    all_hold1 = []
    all_bc0 = []
    all_bc1 = []

    # NEW: Add OVHS tracking
    all_ov_shift0 = []
    all_ov_shift1 = []
    all_ov_hold0 = []
    all_ov_hold1 = []
    
    # vad = dm.test_dset.data["vad"]
    # sessions = dm.test_dset.data["session"]

    vad = dm.train_dset.data["vad"] + dm.val_dset.data["vad"] + dm.test_dset.data["vad"]
    sessions = dm.train_dset.data["session"] + dm.val_dset.data["session"] + dm.test_dset.data["session"]

    '''
    For OVHS analysis and visualization
    '''
    # Add this after your existing analysis
    ovhs_detailed_analysis_results = []

    # Process each session for detailed overlapping event analysis
    for d, s in tqdm(zip(vad[:2], sessions[:2]), desc="Detailed OVHS Analysis"):  # Limit to first 10 for testing
        # Get audio file path from dataset
        audio_info = get_audio_path_for_session(s, dm)
        
        if audio_info:
            print(f"Found audio for session {s}: {audio_info['path']}")
        else:
            print(f"No audio path found for session {s}")

        session_results = analyze_ovhs_events(
            vad=d,  # Add batch dimension
            session_id=s,
            eventer=eventer,
            output_dir=os.path.join(repo_root(), 'event_analysis', cfg_dict['data']['datasets'], 'detailed_analysis'),
            audio_info=audio_info
        )
        
        ovhs_detailed_analysis_results.extend(session_results)

    # Save detailed results
    if ovhs_detailed_analysis_results:
        detailed_df = pd.DataFrame(ovhs_detailed_analysis_results)
        output_path = os.path.join(repo_root(), 'event_analysis', cfg_dict['data']['datasets'], 'overlapping_events_detailed.csv')
        # f"/ahc/work2/kazuyo-oni/turntaking/output/{cfg_dict['data']['datasets']}/overlapping_events_detailed.csv"
        detailed_df.to_csv(output_path, index=False)
        print(f"Detailed overlapping events analysis saved to: {output_path}")
        
        # Print summary
        print("\nOverlapping Events Summary:")
        print(detailed_df.groupby(['event_type', 'speaker']).agg({
            'duration_seconds': ['count', 'mean', 'std', 'min', 'max']
        }).round(3))

    '''
    OVHS over
    '''


    all_ov_count = {0: 0, 1: 0}

    for d, s in tqdm(zip(vad, sessions), desc="Processing"):
        # print(f"{s}: {d.shape}")
        e = eventer(d, max_frame=None)
        # print(s)
        # print("shift")
        shift = find_continuous_ones(eventer.tt["shift_dur"])
        # print("hold")
        hold = find_continuous_ones(eventer.tt["hold_dur"])
        ov = find_continuous_ones(eventer.tt["ov_dur"])
        bc = find_continuous_ones(eventer.bcs["bc_dur"])


        for key, values in ov.items():
            neg_values = [-x for x in values]
            shift.setdefault(key, []).extend(neg_values)


        # NEW: Process OVHS events if available
        ov_shift = {}
        ov_hold = {}
        if eventer.OVHS is not None and hasattr(eventer, 'ov_tt'):
            ov_shift = find_continuous_ones(eventer.ov_tt["ov_shift_dur"])
            ov_hold = find_continuous_ones(eventer.ov_tt["ov_hold_dur"])



        all_shift0 += shift[0]
        all_shift1 += shift[1]
        all_hold0 += hold[0]
        all_hold1 += hold[1]
        all_bc0 += bc[0]
        all_bc1 += bc[1]


        
        # NEW: Update OVHS tracking
        all_ov_shift0 += ov_shift.get(0, [])
        all_ov_shift1 += ov_shift.get(1, [])
        all_ov_hold0 += ov_hold.get(0, [])
        all_ov_hold1 += ov_hold.get(1, [])


        all_ov_count[0] += len(ov[0])
        all_ov_count[1] += len(ov[1])

        results.append(
            [
                s, 
                shift[0], 
                shift[1], 
                shift[0] + shift[1], 

                hold[0], 
                hold[1], 
                hold[0] + hold[1], 

                bc[0], 
                bc[1], 
                bc[0] + bc[1],
                # NEW: Add OVHS results
                ov_shift.get(0, []), ov_shift.get(1, []), 
                ov_shift.get(0, []) + ov_shift.get(1, []),
                ov_hold.get(0, []), ov_hold.get(1, []), 
                ov_hold.get(0, []) + ov_hold.get(1, [])
            ]
        )

    shift_ov_ratio_0 = all_ov_count[0] / len(all_shift0)
    shift_ov_ratio_1 = all_ov_count[1] / len(all_shift1)
    print(f"Shift to OV Ratio for 0: {shift_ov_ratio_0}")
    print(f"Shift to OV Ratio for 1: {shift_ov_ratio_1}")
    exit(1)

    df = pd.DataFrame(results, columns=[
        'session', 
        'shift0', 'shift1', 'shift', 
        'hold0', 'hold1', 'hold', 
        'bc0', 'bc1', "bc",
        # NEW: Add OVHS columns
        'ov_shift0', 'ov_shift1', 'ov_shift',
        'ov_hold0', 'ov_hold1', 'ov_hold'
    ])

    all_session_data = ['all', all_shift0, all_shift1, all_shift0 + all_shift1, 
                        all_hold0, all_hold1, all_hold0 + all_hold1, 
                        all_bc0, all_bc1, all_bc0 + all_bc1]
    df.loc[len(df)] = all_session_data

    all_row = df[df['session'] == 'all']

    statistics = {}
    categories = [
        'shift0', 'shift1', 'shift', 
        'hold0', 'hold1', 'hold',  
        'bc0', 'bc1', 'bc',
        # NEW: Add OVHS categories
        'ov_shift0', 'ov_shift1', 'ov_shift',
        'ov_hold0', 'ov_hold1', 'ov_hold'
    ]
    for category in categories:
        statistics[category] = calculate_statistics(all_row[category].iloc[0])

    stats_df = pd.DataFrame(statistics).transpose()
    print(stats_df)

    if HIST:
        for index, row in df.iterrows():
            session = row['session']
            for column in df.columns[1:]:
                if DATA is not None:
                    plot_histogram(row[column], f'{session}{DATA}', column, f"/ahc/work2/kazuyo-oni/turntaking/output/{cfg_dict['data']['datasets']}")
                else:
                    plot_histogram(row[column], f'{session}', column, f"/ahc/work2/kazuyo-oni/turntaking/output/{cfg_dict['data']['datasets']}")
    else:
        for index, row in df.iterrows():
            session = row['session']
            if session == 'all':
                for column in df.columns[1:]:
                    if DATA is not None:
                        plot_histogram(row[column], f'{session}{DATA}', column, f"/ahc/work2/kazuyo-oni/turntaking/output/{cfg_dict['data']['datasets']}")
                    else:
                        plot_histogram(row[column], f'{session}', column, f"/ahc/work2/kazuyo-oni/turntaking/output/{cfg_dict['data']['datasets']}") 

def plot_histogram(data, session, column, output_dir):
    if not data:
        return

    plt.figure()

    if column in ["shift", "shift0", "shift1"]:
        plt.hist(data, bins=[i/20 for i in range(-40, 61)], edgecolor='black', range=(-2, 3), color='#035894')
    elif column in ["hold", "hold0", "hold1"]:
        plt.hist(data, bins=[i/20 for i in range(0, 61)], edgecolor='black', range=(0, 3), color='#035894')
    elif column in ["bc", "bc0", "bc1"]:
        plt.hist(data, bins=[i/20 for i in range(5, 21)], edgecolor='black', range=(0.2, 1), color='#035894')
    else:
        plt.hist(data, bins=[i/20 for i in range(int(min(data)*20), int(max(data)*20)+1)], edgecolor='black', color='#035894')

    # plt.title(f'{session} - {column}')

    plt.xlabel('Time (seconds)', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)

    filename = f'{output_dir}/{session}_{column}.pdf'
    plt.savefig(filename)
    plt.close()


if __name__ == "__main__":
    main()