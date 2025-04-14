from cognitive_detection import CognitiveDeclineDetector
import os
import nltk

def download_nltk_resources():
    """Download required NLTK resources."""
    resources = ['punkt', 'stopwords', 'averaged_perceptron_tagger']
    for resource in resources:
        try:
            nltk.download(resource, quiet=True)
        except Exception as e:
            print(f"Error downloading {resource}: {e}")

def main():
    # Download required NLTK resources
    print("Downloading required NLTK resources...")
    download_nltk_resources()
    
    # Initialize the detector
    detector = CognitiveDeclineDetector(
        audio_dir='audio_samples/original',
        cache_dir='cache'
    )
    
    # Load samples
    print("Loading audio samples...")
    detector.load_samples()
    
    if not detector.samples:
        print("\nError: No audio samples found. Please ensure:")
        print("1. You have placed audio files in 'audio_samples/original' directory")
        print("2. The audio files are in .wav, .mp3, or .flac format")
        return
    
    # Extract features
    print("\nExtracting features...")
    linguistic_features = detector.extract_linguistic_features()
    acoustic_features = detector.extract_acoustic_features()
    
    # Combine features
    print("\nCombining features...")
    combined_features = detector.combine_features()
    
    # Apply unsupervised learning
    print("\nApplying unsupervised learning...")
    detector.apply_unsupervised_learning()
    
    # Generate visualizations
    print("\nGenerating visualizations...")
    detector.visualize_results()
    
    # Generate report
    print("\nGenerating report...")
    detector.generate_report()
    
    print("\nPipeline completed successfully!")

if __name__ == "__main__":
    main() 