"""
MemoTag AI/ML Task: Voice-Based Cognitive Decline Pattern Detection
This notebook implements a proof-of-concept pipeline for detecting cognitive decline markers in speech.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import librosa
import librosa.display
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
from sklearn.metrics.pairwise import cosine_similarity
import warnings
import speech_recognition as sr
from pydub import AudioSegment
import re
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
import joblib
from functools import lru_cache
import concurrent.futures
from pathlib import Path
import json
nltk.download('punkt')
nltk.download('stopwords')
from nltk.corpus import stopwords
import scipy.stats as stats

# Suppress warnings
warnings.filterwarnings('ignore')

class CognitiveDeclineDetector:
    """
    A class to detect patterns of cognitive decline from voice samples.
    """
    
    def __init__(self, audio_dir='audio_samples/', cache_dir='cache/'):
        """
        Initialize the detector with path to audio samples.
        
        Parameters:
        -----------
        audio_dir : str
            Directory containing audio samples
        cache_dir : str
            Directory for caching processed data
        """
        self.audio_dir = audio_dir
        self.cache_dir = cache_dir
        self.samples = []
        self.features = pd.DataFrame()
        self.transcriptions = {}
        self.recognizer = sr.Recognizer()
        
        # Create directories if they don't exist
        for dir_path in [audio_dir, cache_dir]:
            if not os.path.exists(dir_path):
                os.makedirs(dir_path)
        
        # Initialize cache paths
        self.cache_paths = {
            'features': os.path.join(cache_dir, 'features.pkl'),
            'transcriptions': os.path.join(cache_dir, 'transcriptions.json'),
            'audio_metadata': os.path.join(cache_dir, 'audio_metadata.json')
        }
        
        # Load cached data if available
        self._load_cached_data()
    
    def _load_cached_data(self):
        """Load cached data if available."""
        try:
            if os.path.exists(self.cache_paths['features']):
                self.features = pd.read_pickle(self.cache_paths['features'])
            if os.path.exists(self.cache_paths['transcriptions']):
                with open(self.cache_paths['transcriptions'], 'r') as f:
                    self.transcriptions = json.load(f)
        except Exception as e:
            print(f"Warning: Could not load cached data: {e}")
    
    def _save_cached_data(self):
        """Save processed data to cache."""
        try:
            if not self.features.empty:
                self.features.to_pickle(self.cache_paths['features'])
            if self.transcriptions:
                with open(self.cache_paths['transcriptions'], 'w') as f:
                    json.dump(self.transcriptions, f)
        except Exception as e:
            print(f"Warning: Could not save cached data: {e}")
    
    @lru_cache(maxsize=100)
    def _load_audio_file(self, file_path):
        """Cached function to load audio files."""
        try:
            audio, sr = librosa.load(file_path, sr=None)
            return audio, sr
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            return None, None
    
    def load_samples(self, sample_files=None, batch_size=10):
        """
        Load audio samples from directory or list of files in batches.
        
        Parameters:
        -----------
        sample_files : list, optional
            List of file paths to load. If None, load all files from audio_dir.
        batch_size : int
            Number of files to process in each batch
        """
        if sample_files is None:
            print(f"Searching for audio files recursively in: {self.audio_dir}")
            sample_files = []
            for dirpath, dirnames, filenames in os.walk(self.audio_dir):
                for f in filenames:
                    if f.lower().endswith(('.wav', '.mp3', '.flac')):
                        full_path = os.path.join(dirpath, f)
                        sample_files.append(full_path)
            
            if not sample_files:
                print(f"Warning: No audio files found in {self.audio_dir}")
                return
        
        print(f"Found {len(sample_files)} potential audio samples...")
        
        # Process files in batches
        self.samples = []
        for i in range(0, len(sample_files), batch_size):
            batch = sample_files[i:i + batch_size]
            with concurrent.futures.ThreadPoolExecutor() as executor:
                futures = [executor.submit(self._load_audio_file, file_path) for file_path in batch]
                for file_path, future in zip(batch, futures):
                    audio, sr = future.result()
                    if audio is not None:
                        self.samples.append({
                            'file_path': file_path,
                            'audio': audio,
                            'sr': sr,
                            'filename': os.path.basename(file_path)
                        })
                        print(f"Loaded: {os.path.basename(file_path)}")
        
        print(f"Successfully loaded {len(self.samples)} samples")
    
    @lru_cache(maxsize=100)
    def speech_to_text(self, audio_file):
        """
        Convert speech to text with caching.
        """
        try:
            with sr.AudioFile(audio_file) as source:
                audio = self.recognizer.record(source)
                text = self.recognizer.recognize_google(audio)
                return text
        except Exception as e:
            print(f"Error in speech recognition for {audio_file}: {e}")
            return ""
    
    def extract_linguistic_features(self):
        """
        Extract linguistic features using sequential processing.
        """
        if not self.samples:
            print("No samples loaded. Please load samples first.")
            return None
        
        print("Extracting linguistic features...")
        
        features_list = []
        for sample in self.samples:
            file_path = sample['file_path']
            if file_path in self.transcriptions:
                text = self.transcriptions[file_path]
            else:
                text = self.speech_to_text(file_path)
                self.transcriptions[file_path] = text
            
            # Basic text features
            words = text.split()  # Simple word tokenization
            sentences = text.split('.')  # Simple sentence tokenization
            
            features = {
                'word_count': len(words),
                'sentence_count': len([s for s in sentences if s.strip()]),
                'avg_word_length': np.mean([len(word) for word in words]) if words else 0,
                'filename': sample['filename'],
                'file_path': file_path
            }
            
            # Word frequency features
            words_lower = [w.lower() for w in words]
            stop_words = set(stopwords.words('english'))
            features['stopword_ratio'] = len([w for w in words_lower if w in stop_words]) / len(words) if words else 0
            
            features_list.append(features)
        
        # Convert features to DataFrame and store as class attribute
        self.linguistic_features = pd.DataFrame(features_list)
        
        # Save to cache
        self.linguistic_features.to_pickle(os.path.join(self.cache_dir, 'linguistic_features.pkl'))
        print(f"Extracted {len(self.linguistic_features.columns) - 2} linguistic features")  # -2 for filename and file_path
        
        return self.linguistic_features
    
    def extract_acoustic_features(self):
        """
        Extract acoustic features from audio samples.
        """
        if not self.samples:
            print("No samples loaded. Please load samples first.")
            return None
        
        print("Extracting acoustic features...")
        
        features_list = []
        for sample in self.samples:
            audio = sample['audio']
            sr = sample['sr']
            
            features = {
                'filename': sample['filename'],
                'file_path': sample['file_path'],
                'duration': librosa.get_duration(y=audio, sr=sr),
                'zero_crossing_rate': np.mean(librosa.feature.zero_crossing_rate(y=audio)),
                'spectral_centroid': np.mean(librosa.feature.spectral_centroid(y=audio, sr=sr)),
                'spectral_rolloff': np.mean(librosa.feature.spectral_rolloff(y=audio, sr=sr))
            }
            
            # Add MFCC features
            mfccs = librosa.feature.mfcc(y=audio, sr=sr)
            for i, mfcc in enumerate(np.mean(mfccs, axis=1)):
                features[f'mfcc_{i}'] = mfcc
            
            # Add pitch features
            pitches, magnitudes = librosa.piptrack(y=audio, sr=sr)
            features['pitch_mean'] = np.mean(pitches[magnitudes > np.max(magnitudes)/10])
            features['pitch_std'] = np.std(pitches[magnitudes > np.max(magnitudes)/10])
            
            features_list.append(features)
        
        # Convert features to DataFrame and store as class attribute
        self.acoustic_features = pd.DataFrame(features_list)
        
        # Save to cache
        self.acoustic_features.to_pickle(os.path.join(self.cache_dir, 'acoustic_features.pkl'))
        print(f"Extracted {len(self.acoustic_features.columns) - 2} acoustic features")  # -2 for filename and file_path
        
        return self.acoustic_features
    
    def combine_features(self):
        """
        Combine linguistic and acoustic features into a single feature set.
        """
        print("Combining features...")
        
        if not hasattr(self, 'linguistic_features') or not hasattr(self, 'acoustic_features'):
            print("Features not extracted yet!")
            return None
            
        # Check if features are available
        has_linguistic = hasattr(self, 'linguistic_features') and not self.linguistic_features.empty
        has_acoustic = hasattr(self, 'acoustic_features') and not self.acoustic_features.empty
        
        if not has_linguistic and not has_acoustic:
            print("No features available!")
            return None
        
        # Merge features if both are available
        if has_linguistic and has_acoustic:
            self.features = pd.merge(
                self.linguistic_features,
                self.acoustic_features,
                on=['filename', 'file_path'],
                how='outer'
            )
        elif has_linguistic:
            self.features = self.linguistic_features.copy()
        else:
            self.features = self.acoustic_features.copy()
        
        # Fill any missing values
        self.features.fillna(0, inplace=True)
        
        # Remove duplicate columns if any
        self.features = self.features.loc[:, ~self.features.columns.duplicated()]
        
        print(f"Combined features shape: {self.features.shape}")
        print(f"Features available: {list(self.features.columns)}")
        
        return self.features
    
    def apply_unsupervised_learning(self):
        """
        Apply unsupervised learning techniques to identify patterns.
        """
        print("Applying unsupervised learning...")
        
        if not hasattr(self, 'features') or self.features.empty:
            print("No features available! Extract features first.")
            return
        
        # Prepare data for modeling
        feature_cols = [col for col in self.features.columns if col not in ['filename', 'file_path']]
        if not feature_cols:
            print("No numeric features available for analysis!")
            return
            
        X = self.features[feature_cols]
        
        # Standardize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # 1. PCA for dimensionality reduction
        n_components = min(2, X.shape[1])
        pca = PCA(n_components=n_components)
        X_pca = pca.fit_transform(X_scaled)
        
        # Create visualization dataframe
        vis_df = pd.DataFrame({
            'filename': self.features['filename'],
            'PC1': X_pca[:, 0],
            'PC2': X_pca[:, 1] if n_components > 1 else np.zeros(len(X_pca))
        })
        
        # 2. K-means clustering
        n_clusters = min(2, len(X))
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        clusters = kmeans.fit_predict(X_scaled)
        vis_df['cluster'] = clusters
        
        # 3. Anomaly detection
        iso_forest = IsolationForest(contamination=0.1, random_state=42)
        outliers = iso_forest.fit_predict(X_scaled)
        vis_df['is_outlier'] = [0 if x == -1 else 1 for x in outliers]
        
        # 4. Calculate risk scores
        self.risk_scores = {}
        for i, row in enumerate(X_scaled):
            filename = self.features.iloc[i]['filename']
            # Anomaly score from Isolation Forest
            anomaly_score = -iso_forest.score_samples([row])[0]
            self.risk_scores[filename] = anomaly_score
        
        # Add risk scores to visualization dataframe
        vis_df['risk_score'] = [self.risk_scores[f] for f in vis_df['filename']]
        
        # Sort samples by risk score
        risk_df = pd.DataFrame({
            'filename': list(self.risk_scores.keys()),
            'risk_score': list(self.risk_scores.values())
        }).sort_values('risk_score', ascending=False)
        
        # Store results
        self.vis_df = vis_df
        self.feature_importance = pd.Series(pca.components_[0], index=feature_cols)
        self.risk_df = risk_df
        
        print("Unsupervised learning completed")
        return vis_df, self.feature_importance, risk_df
    
    def visualize_results(self):
        """
        Create visualizations of the analysis results.
        """
        if not hasattr(self, 'vis_df') or not hasattr(self, 'feature_importance') or not hasattr(self, 'risk_df'):
            print("\nError: Analysis results (vis_df, feature_importance, risk_df) not found.")
            print("This usually means apply_unsupervised_learning() did not complete successfully.")
            print("Possible causes: No audio files loaded, or feature extraction failed.")
        return

        print("Generating visualizations...")
        
        # 1. PCA plot with cluster assignments
        plt.figure(figsize=(12, 10))
        
        plt.subplot(2, 2, 1)
        sns.scatterplot(data=self.vis_df, x='PC1', y='PC2', hue='cluster', palette='viridis', style='is_outlier', s=100)
        plt.title('PCA: Sample Clustering and Outliers')
        plt.xlabel('Principal Component 1')
        plt.ylabel('Principal Component 2')
        plt.legend(title='Cluster')
        
        # 2. Top 10 most important features
        plt.subplot(2, 2, 2)
        top_features = self.feature_importance.nlargest(10)
        sns.barplot(x=top_features.values, y=top_features.index, palette='rocket')
        plt.title('Top 10 Most Discriminative Features')
        plt.xlabel('Feature Importance')
        plt.tight_layout()
        
        # 3. Risk scores distribution
        plt.subplot(2, 2, 3)
        sns.histplot(self.risk_df['risk_score'], kde=True, color='purple')
        plt.title('Distribution of Cognitive Decline Risk Scores')
        plt.xlabel('Risk Score')
        plt.ylabel('Count')
        
        # 4. Risk score by sample
        plt.subplot(2, 2, 4)
        sns.barplot(data=self.risk_df, x='risk_score', y='filename', palette='rocket_r')
        plt.title('Ranked Risk Scores by Sample')
        plt.xlabel('Risk Score (Higher = More Risk Indicators)')
        plt.ylabel('Sample ID')
        
        plt.tight_layout()
        plt.savefig('cognitive_decline_analysis.png', dpi=300)
        plt.show()
        
        # Additional feature-specific visualizations
        plt.figure(figsize=(12, 10))
        
        # 1. Hesitation ratio vs. Words per minute
        plt.subplot(2, 2, 1)
        sns.scatterplot(data=self.features, x='words_per_minute', y='hesitation_ratio', 
                       hue=self.vis_df['cluster'], size=self.vis_df['risk_score'], sizes=(50, 200))
        plt.title('Speech Rate vs. Hesitation')
        plt.xlabel('Words per Minute')
        plt.ylabel('Hesitation Ratio')
        
        # 2. Pauses vs. Pitch variability
        plt.subplot(2, 2, 2)
        sns.scatterplot(data=self.features, x='pitch_variability', y='pauses_per_minute', hue=self.vis_df['cluster'], size=self.vis_df['risk_score'], sizes=(50, 200))
        plt.title('Pitch Variability vs. Pause Frequency')
        plt.xlabel('Pitch Variability')
        plt.ylabel('Pauses per Minute')
        
        # 3. Lexical diversity vs. Repetition
        plt.subplot(2, 2, 3)
        sns.scatterplot(data=self.features, x='lexical_diversity', y='repetition_score', hue=self.vis_df['cluster'], size=self.vis_df['risk_score'], sizes=(50, 200))
        plt.title('Lexical Diversity vs. Word Repetition')
        plt.xlabel('Lexical Diversity')
        plt.ylabel('Repetition Score')
        
        # 4. Incomplete sentences vs. Average pause duration
        plt.subplot(2, 2, 4)
        sns.scatterplot(data=self.features, x='avg_pause_duration', y='incomplete_sentence_ratio', hue=self.vis_df['cluster'], size=self.vis_df['risk_score'], sizes=(50, 200))
        plt.title('Pause Duration vs. Incomplete Sentences')
        plt.xlabel('Average Pause Duration (s)')
        plt.ylabel('Incomplete Sentence Ratio')
        
        plt.tight_layout()
        plt.savefig('feature_relationships.png', dpi=300)
        plt.show()
        
        print("Visualizations generated and saved as PNG files")
    
    def generate_report(self):

        required_attrs = ['samples', 'features', 'vis_df', 'feature_importance', 'risk_df']
        # missing_attrs = [attr for attr in required_attrs if not hasattr(self, attr) or \
                    # (isinstance(getattr(self, attr), (pd.DataFrame, list, dict)) and not getattr(self, attr))] # Check if exists and is not empty

        # if missing_attrs:
        #     print("\nError: Cannot generate report. Missing or empty analysis components:")
        #     return
        
        #     for attr in missing_attrs:
        #         print(f"- self.{attr}")
        #     print("Please ensure the pipeline ran successfully up to apply_unsupervised_learning().")
        #     return
        """
            Generate a report of the analysis results.
        """
        print("\n========= COGNITIVE DECLINE DETECTION REPORT =========\n")
        
        # Sample Summary
        print(f"Number of samples analyzed: {len(self.samples)}")
        print(f"Features extracted: {self.features.shape[1] - 1}\n")  # -1 for filename column
        
        # Most discriminative features
        print("TOP 5 MOST DISCRIMINATIVE FEATURES:")
        if hasattr(self, 'feature_importance') and not self.feature_importance.empty:
            for feature, importance in self.feature_importance.nlargest(5).items():
                print(f"- {feature}: {importance:.4f}")
        print()
        
        # Risk Assessment
        print("SAMPLE RISK ASSESSMENT:")
        if hasattr(self, 'risk_df') and not self.risk_df.empty:
            for i, row in self.risk_df.iterrows():
                risk_category = "HIGH RISK" if row['risk_score'] > 1.0 else "MEDIUM RISK" if row['risk_score'] > 0.5 else "LOW RISK"
                print(f"- {row['filename']}: {row['risk_score']:.4f} ({risk_category})")
        print()
        
        # Key findings
        print("KEY FINDINGS:")
        if hasattr(self, 'vis_df') and not self.vis_df.empty and hasattr(self, 'features') and not self.features.empty and hasattr(self, 'feature_importance') and not self.feature_importance.empty:
            print("1. Speech patterns most predictive of cognitive decline:")

            if 'cluster' not in self.vis_df.columns or 'risk_score' not in self.vis_df.columns:
                print("   - Warning: 'cluster' or 'risk_score' column missing in vis_df. Skipping cluster comparison.")
            else:
                cluster0_risk = self.vis_df[self.vis_df['cluster'] == 0]['risk_score'].mean()
                cluster1_risk = self.vis_df[self.vis_df['cluster'] == 1]['risk_score'].mean()
                higher_risk_cluster = 0 if cluster0_risk > cluster1_risk else 1
        
        # Get mean values for each cluster
                high_risk_samples = self.features[self.vis_df['cluster'] == higher_risk_cluster]
                low_risk_samples = self.features[self.vis_df['cluster'] != higher_risk_cluster]
        
        # Compare top features between clusters
                for feature in self.feature_importance.nlargest(5).index:
                    if feature in high_risk_samples.columns and feature in low_risk_samples.columns:
                        high_mean = high_risk_samples[feature].mean()
                        low_mean = low_risk_samples[feature].mean()
                        difference = high_mean - low_mean
                        direction = "higher" if difference > 0 else "lower"
                print(f"   - {feature}: {abs(difference):.2f} {direction} in high-risk group")
        
        # print("\n2. Anomaly Detection:")
        # if 'is_outlier' not in self.vis_df.columns:
        #     print("   - Warning: 'is_outlier' column missing in vis_df. Skipping outlier reporting.")
        # else:
        #     outliers = self.vis_df[self.vis_df['is_outlier'] == 0]
        #     if not outliers.empty:
        #         print(f"   - {len(outliers)} samples identified as potential outliers")
        #         for i, row in outliers.iterrows():
        #             print(f"     * {row['filename']} (Risk Score: {row['risk_score']:.4f})")
        #     else:
        #         print("   - No significant outliers detected")
        
        print("\nRECOMMENDATIONS FOR CLINICAL DEPLOYMENT:")
        print("1. Expand dataset with validated clinical samples")
        print("2. Incorporate supervised learning with confirmed diagnoses")
        print("3. Add contextual task performance features (e.g., verbal fluency tests)")
        print("4. Develop longitudinal tracking to monitor changes over time")
        print("5. Validate against established cognitive assessment tools")
        
        print("\n=================================================\n")
    
    def get_risk_score(self, audio_file):
        """
        Score a new audio file for cognitive decline risk.
        Can be called as an API endpoint.
        
        Parameters:
        -----------
        audio_file : str
            Path to audio file to analyze
            
        Returns:
        --------
        float
            Risk score (higher = more risk indicators)
        """
        # Temporary add this file to our samples
        original_samples = self.samples.copy()
        original_features = self.features.copy() if hasattr(self, 'features') else None
        
        # Load the new sample
        self.load_samples([audio_file])
        if not self.samples:
            print(f"Error loading audio file: {audio_file}")
            return None
        
        # Extract features for this sample
        new_sample = self.samples[-1]
        filename = new_sample['filename']
        
        # Convert speech to text
        self.speech_to_text()
        
        # Extract features
        self.extract_linguistic_features()
        self.extract_acoustic_features()
        self.combine_features()
        
        # Apply the existing model to get a risk score
        X = self.features.drop('filename', axis=1)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Use Isolation Forest for anomaly detection
        iso_forest = IsolationForest(contamination=0.1, random_state=42)
        iso_forest.fit(X_scaled)
        
        # Get anomaly score for the new sample
        new_sample_idx = self.features[self.features['filename'] == filename].index[0]
        new_sample_features = X_scaled[new_sample_idx].reshape(1, -1)
        anomaly_score = -iso_forest.score_samples(new_sample_features)[0]
        
        # Calculate risk score
        risk_score = anomaly_score
        
        # Restore original state
        self.samples = original_samples
        if original_features is not None:
            self.features = original_features
        
        return risk_score

# Example usage of the pipeline
if __name__ == "__main__":
    # Create detector instance
    detector = CognitiveDeclineDetector(audio_dir='audio_samples/dev-clean')
    
    # Load samples
    detector.load_samples()

    if not detector.samples:
        print("\nError: No audio samples found. Please ensure:")
        print("1. You have placed initial audio files in 'audio_samples/original'.")
        print("2. You have run 'data_collection.py' to potentially create processed samples.")
        exit()
    
    # Process the data
    detector.speech_to_text()

    if not detector.transcriptions:
        print("\nWarning: Speech-to-text failed for all samples. Skipping linguistic features.")
    else:
        detector.extract_linguistic_features()

    detector.extract_acoustic_features()
    detector.combine_features()
    
    # Apply ML
    detector.apply_unsupervised_learning()
    
    # Generate visualizations and report
    detector.visualize_results()
    detector.generate_report()
    
    # Example API-like function call
    # risk_score = detector.get_risk_score('path/to/new/audio.wav')
    # print(f"Risk score for new sample: {risk_score}")