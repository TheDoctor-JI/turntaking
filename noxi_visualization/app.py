from flask import Flask, render_template, jsonify, request, send_file
import os
import json
import pandas as pd
import numpy as np
from pathlib import Path

app = Flask(__name__)

# Configuration
DATASET_PATH = "/home/eeyifanshen/e2e_audio_LLM/Datasets/raw/MultiModal/Noxi/NoXi_dataset/For_MMVAP_REPRODUCE"

def get_available_sessions():
    """Get list of available sessions from the dataset"""
    sessions = []
    if os.path.exists(DATASET_PATH):
        for item in os.listdir(DATASET_PATH):
            item_path = os.path.join(DATASET_PATH, item)
            if os.path.isdir(item_path) and not item.startswith('.'):
                # Check if the session has the required video files
                expert_video = os.path.join(item_path, "expert.video.mp4")
                novice_video = os.path.join(item_path, "novice.video.mp4")
                if os.path.exists(expert_video) and os.path.exists(novice_video):
                    sessions.append(item)
    return sorted(sessions)

@app.route('/')
def index():
    """Main page with session selector"""
    sessions = get_available_sessions()
    return render_template('index.html', sessions=sessions)

@app.route('/api/sessions')
def api_sessions():
    """API endpoint to get available sessions"""
    sessions = get_available_sessions()
    return jsonify(sessions)

@app.route('/video/<session>/<role>')
def serve_video(session, role):
    """Serve video files for expert or novice"""
    if role not in ['expert', 'novice']:
        return "Invalid role", 400
    
    video_path = os.path.join(DATASET_PATH, session, f"{role}.video.mp4")
    if not os.path.exists(video_path):
        return "Video not found", 404
    
    return send_file(video_path, mimetype='video/mp4')

@app.route('/api/session/<session>/info')
def session_info(session):
    """Get information about a specific session"""
    session_path = os.path.join(DATASET_PATH, session)
    if not os.path.exists(session_path):
        return "Session not found", 404
    
    info = {
        'session': session,
        'expert_video': f"/video/{session}/expert",
        'novice_video': f"/video/{session}/novice",
        'has_expert_features': os.path.exists(os.path.join(session_path, "non_varbal_expert.csv")),
        'has_novice_features': os.path.exists(os.path.join(session_path, "non_varbal_novice.csv"))
    }
    
    return jsonify(info)

@app.route('/api/session/<session>/gaze_data')
def get_gaze_data(session):
    """Get gaze data for both expert and novice"""
    session_path = os.path.join(DATASET_PATH, session)
    if not os.path.exists(session_path):
        return "Session not found", 404
    
    expert_csv = os.path.join(session_path, "non_varbal_expert.csv")
    novice_csv = os.path.join(session_path, "non_varbal_novice.csv")
    
    data = {}
    
    # Load expert data
    if os.path.exists(expert_csv):
        try:
            df_expert = pd.read_csv(expert_csv)
            # Extract gaze data (gaze_x, gaze_y, gaze_confidence)
            expert_gaze = {
                'gaze_x': df_expert['gaze_x'].tolist(),
                'gaze_y': df_expert['gaze_y'].tolist(),
                'gaze_confidence': df_expert['gaze_confidence'].tolist(),
                'frame_count': len(df_expert),
                'fps': 25  # 25Hz as mentioned
            }
            data['expert'] = expert_gaze
        except Exception as e:
            print(f"Error loading expert data: {e}")
            data['expert'] = None
    
    # Load novice data
    if os.path.exists(novice_csv):
        try:
            df_novice = pd.read_csv(novice_csv)
            # Extract gaze data (gaze_x, gaze_y, gaze_confidence)
            novice_gaze = {
                'gaze_x': df_novice['gaze_x'].tolist(),
                'gaze_y': df_novice['gaze_y'].tolist(),
                'gaze_confidence': df_novice['gaze_confidence'].tolist(),
                'frame_count': len(df_novice),
                'fps': 25  # 25Hz as mentioned
            }
            data['novice'] = novice_gaze
        except Exception as e:
            print(f"Error loading novice data: {e}")
            data['novice'] = None
    
    return jsonify(data)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)