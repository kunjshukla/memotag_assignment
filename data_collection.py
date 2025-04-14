"""
Data Collection Script for Voice-Based Cognitive Decline Detection

This script downloads and prepares sample voice data for the cognitive decline detection pipeline.
It includes:
1. Functions to download public datasets
2. Tools to synthesize cognitive impairment indicators
3. Setup of a proper folder structure for the project
"""

import os
import requests
import zipfile
import io
import random
import numpy as np
import librosa
import soundfile as sf
from pydub import AudioSegment
from tqdm import tqdm
import shutil
import urllib.request
import tarfile

# Create necessary directories
def create_project_structure():
    """Create the project directory structure."""
    dirs = [
        'audio_samples',
        'audio_samples/original',
        'audio_samples/processed',
        'output',
        'models'
    ]
    
    for dir_path in dirs:
        os.makedirs(dir_path, exist_ok=True)
        print(f"Created directory: {dir_path}")

# Download sample audio from Mozilla Common Voice
# def download_common_voice_samples(limit=10):
#     """
#     Download a few samples from Mozilla Common Voice.
    
#     Parameters:
#     -----------
#     limit : int
#         Number of samples to download
#     """
#     print("Since Mozilla Common Voice requires authentication, we'll use alternative sources.")
#     print("Downloading sample English speech clips...")
    
#     # Source: LibriSpeech samples (smaller, more accessible)
#     librispeech_url = "https://www.openslr.org/resources/12/dev-clean.tar.gz"
#     target_path = "temp_librispeech.tar.gz"
    
#     print(f"Downloading LibriSpeech samples from {librispeech_url}...")
    
#     # Create a directory to extract to
#     extract_dir = "temp_librispeech"
#     os.makedirs(extract_dir, exist_ok=True)
    
#     # Download the file
#     try:
#         urllib.request.urlretrieve(librispeech_url, target_path)
        
#         # Extract the archive
#         with tarfile.open(target_path) as tar:
#             # Extract a subset of files
#             members = tar.getmembers()
#             audio_files = [m for m in members if m.name.endswith('.flac')]
            
#             # Select a random sample of audio files
#             if len(audio_files) > limit:
#                 audio_files = random.sample(audio_files, limit)
            
#             # Extract selected files
#             for i, member in enumerate(audio_files):
#                 print(f"Extracting sample {i+1}/{len(audio_files)}: {member.name}")
#                 tar.extract(member, path=extract_dir)
                
#                 # Copy to our project structure
#                 source_path = os.path.join(extract_dir, member.name)
#                 dest_filename = f"sample_{i+1}_control.flac"
#                 dest_path = os.path.join('audio_samples/original', dest_filename)
                
#                 # Make sure the directory exists
#                 os.makedirs(os.path.dirname(dest_path), exist_ok=True)
                
#                 # Copy the file
#                 shutil.copy(source_path, dest_path)
#                 print(f"Saved as: {dest_path}")
        
#         print(f"Successfully downloaded {limit} audio samples")
        
#         # Clean up
#         os.remove(target_path)
#         shutil.rmtree(extract_dir)
    
#     except Exception as e:
#         print(f"Error downloading samples: {e}")
#         print("Please download audio samples manually and place them in the audio_samples/original directory")


def simulate_cognitive_impairment(input_dir, output_dir, num_samples=5):
    """
    Create simulated cognitive impairment audio samples by modifying existing recordings.
    
    Parameters:
    -----------
    input_dir : str
        Directory with original audio files
    output_dir : str
        Directory to save modified files
    num_samples : int
        Number of simulated samples to create
    """
    print("Simulating cognitive impairment in audio samples...")
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Get list of original audio files
    original_files = [f for f in os.listdir(input_dir) if f.endswith(('.wav', '.mp3', '.flac'))]
    
    if not original_files:
        print("No audio files found in the input directory!")
        return
    
    # Select a subset of files to modify
    if len(original_files) > num_samples:
        files_to_modify = random.sample(original_files, num_samples)
    else:
        files_to_modify = original_files
    
    for i, filename in enumerate(files_to_modify):
        input_path = os.path.join(input_dir, filename)
        
        # Determine cognitive impairment simulation type
        sim_type = random.choice(['hesitation', 'slow_speech', 'pauses', 'pitch_variation', 'mixed'])
        
        # Load the audio file
        audio, sr = librosa.load(input_path, sr=None)
        
        # Apply modifications based on simulation type
        if sim_type == 'hesitation':
            # Insert silence gaps to simulate hesitation
            modified_audio = simulate_hesitation(audio, sr)
            severity = "mild"
        
        elif sim_type == 'slow_speech':
            # Slow down the speech rate
            modified_audio = simulate_slow_speech(audio, sr)
            severity = "moderate"
            
        elif sim_type == 'pauses':
            # Insert longer pauses
            modified_audio = simulate_pauses(audio, sr)
            severity = "severe"
            
        elif sim_type == 'pitch_variation':
            # Reduce pitch variation
            modified_audio = simulate_reduced_pitch_variation(audio, sr)
            severity = "mild"
            
        else:  # mixed
            # Apply multiple modifications
            temp_audio = simulate_hesitation(audio, sr)
            temp_audio = simulate_pauses(temp_audio, sr)
            modified_audio = simulate_reduced_pitch_variation(temp_audio, sr)
            severity = "severe"
        
        # Save the modified audio
        base_name = os.path.splitext(filename)[0]
        output_filename = f"{base_name}_{sim_type}_{severity}_impaired.wav"
        output_path = os.path.join(output_dir, output_filename)
        
        sf.write(output_path, modified_audio, sr)
        print(f"Created simulated sample ({i+1}/{len(files_to_modify)}): {output_filename}")


def simulate_hesitation(audio, sr):
    """Insert hesitation markers (short silences) into audio."""
    # Define parameters
    audio_length = len(audio)
    hesitation_length = int(0.2 * sr)  # 200ms hesitation
    num_hesitations = random.randint(5, 10)  # Random number of hesitations
    
    # Create modified audio
    modified_audio = np.copy(audio)
    
    # Insert hesitations at random positions
    for _ in range(num_hesitations):
        position = random.randint(0, audio_length - hesitation_length)
        # Reduce volume at hesitation points rather than complete silence
        modified_audio[position:position + hesitation_length] *= 0.1
    
    return modified_audio


def simulate_slow_speech(audio, sr):
    """Slow down speech rate."""
    # Use librosa to time-stretch the audio (make it slower)
    rate = random.uniform(0.7, 0.85)  # Slow down by 15-30%
    modified_audio = librosa.effects.time_stretch(audio, rate=rate)
    
    return modified_audio


def simulate_pauses(audio, sr):
    """Insert longer pauses into speech."""
    # Define parameters
    audio_segments = []
    segment_length = len(audio) // random.randint(4, 6)  # Divide audio into segments
    pause_length = int(sr * random.uniform(0.8, 1.5))  # 0.8-1.5 second pause
    
    # Split audio into segments and add pauses
    for i in range(0, len(audio), segment_length):
        segment = audio[i:min(i + segment_length, len(audio))]
        audio_segments.append(segment)
        
        # Add pause after each segment except the last
        if i + segment_length < len(audio):
            audio_segments.append(np.zeros(pause_length))
    
    # Concatenate segments with pauses
    modified_audio = np.concatenate(audio_segments)
    
    return modified_audio


def simulate_reduced_pitch_variation(audio, sr):
    """Reduce pitch variation to simulate flatter affect."""
    # Extract pitch
    pitches, magnitudes = librosa.piptrack(y=audio, sr=sr)
    
    # Apply pitch flattening
    y_harmonic, y_percussive = librosa.effects.hpss(audio)
    modified_audio = y_percussive + y_harmonic * 0.7  # Reduce harmonic content
    
    return modified_audio


if __name__ == "__main__":
    # Set up project structure
    create_project_structure()
    
    # # Download sample data
    # download_common_voice_samples(limit=5)
    
    # Create simulated cognitive impairment samples
    simulate_cognitive_impairment('audio_samples/original', 'audio_samples/processed')
    
    print("Data preparation completed! You now have:")
    print("1. Control samples in audio_samples/original/")
    print("2. Simulated cognitive impairment samples in audio_samples/processed/")
    print("\nNext step: Run the cognitive decline detection pipeline on these samples.")