# Voice-Based Cognitive Decline Detection

This project implements a proof-of-concept pipeline for detecting cognitive decline markers in speech samples. It uses both acoustic and linguistic features to analyze speech patterns and identify potential indicators of cognitive impairment.

## Features

- Acoustic feature extraction (MFCC, pitch, spectral features)
- Linguistic feature extraction (word count, sentence structure, etc.)
- Unsupervised learning for pattern detection
- Risk score calculation
- Visualization of results
- Comprehensive reporting

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd memotag_assignment
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Place audio samples in the `audio_samples/original` directory (supported formats: .wav, .mp3, .flac)

2. Run the pipeline:
```bash
python run_pipeline.py
```

3. Check the results:
- Risk assessment report will be displayed in the console
- Visualizations will be saved as PNG files

## Project Structure

- `cognitive_detection.py`: Main implementation of the cognitive decline detection pipeline
- `run_pipeline.py`: Script to run the complete analysis pipeline
- `requirements.txt`: Python dependencies
- `audio_samples/`: Directory for audio files
- `cache/`: Directory for cached features and intermediate results

## Requirements

- Python 3.8+
- Required Python packages are listed in requirements.txt
- Audio files in .wav, .mp3, or .flac format

## Results

The pipeline generates:
- Feature importance analysis
- Risk scores for each audio sample
- Visualizations of patterns and clusters
- Detailed analysis report

## Recommendations

For clinical deployment:
1. Expand dataset with validated clinical samples
2. Incorporate supervised learning with confirmed diagnoses
3. Add contextual task performance features
4. Develop longitudinal tracking
5. Validate against established assessment tools