import numpy as np
import pandas as pd
import streamlit as st
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.utils.class_weight import compute_class_weight
import time
import joblib
import os

class SimpleEmotionClassifiers:
    def __init__(self):
        self.nb_classifier = None
        self.rf_classifier = None
        self.pca = None
        
        self.emotion_labels = [
            'admiration', 'amusement', 'anger', 'annoyance', 'approval', 'caring',
            'confusion', 'curiosity', 'desire', 'disappointment', 'disapproval',
            'disgust', 'embarrassment', 'excitement', 'fear', 'gratitude', 'grief',
            'joy', 'love', 'nervousness', 'optimism', 'pride', 'realization',
            'relief', 'remorse', 'sadness', 'surprise', 'neutral'
        ]
    
    def compute_emotion_weights(self, y):
        """Compute class weights for each emotion to handle imbalance"""
        weights_per_emotion = {}
        
        for i, emotion in enumerate(self.emotion_labels):
            if i < y.shape[1]:
                emotion_column = y[:, i]
                unique_classes = np.unique(emotion_column)
                
                if len(unique_classes) > 1:
                    # Compute weights for this emotion
                    class_weights = compute_class_weight(
                        'balanced', 
                        classes=unique_classes, 
                        y=emotion_column
                    )
                    weights_per_emotion[emotion] = dict(zip(unique_classes, class_weights))
                else:
                    # Only one class present
                    weights_per_emotion[emotion] = {unique_classes[0]: 1.0}
        
        return weights_per_emotion
    
    def train_naive_bayes(self, X, y):
        """Train Naive Bayes with optimized PCA and GaussianNB for BERT embeddings"""
        try:
            st.write("Training Naive Bayes with optimized PCA and Gaussian distribution...")
            
            # Optimize PCA components based on explained variance
            n_samples, n_features = X.shape
            
            # Choose PCA components more intelligently
            if n_features > 500:
                # For BERT embeddings (768 dims), use explained variance approach
                pca_test = PCA(n_components=0.95, random_state=42)  # Keep 95% variance
                pca_test.fit(X)
                optimal_components = pca_test.n_components_
                
                # But cap it for performance
                optimal_components = min(optimal_components, 200, n_features // 3)
            else:
                optimal_components = min(100, n_features // 2)
            
            self.pca = PCA(n_components=optimal_components, random_state=42)
            X_reduced = self.pca.fit_transform(X)
            
            # Show PCA effectiveness
            explained_variance = self.pca.explained_variance_ratio_.sum()
            st.info(f"PCA: {n_features} → {optimal_components} features (keeping {explained_variance:.1%} variance)")
            
            # Use GaussianNB instead of MultinomialNB for continuous features
            nb = GaussianNB(
                var_smoothing=1e-8  # Small smoothing for numerical stability
            )
            
            # Use MultiOutputClassifier
            classifier = MultiOutputClassifier(nb, n_jobs=-1)
            
            # Fit the model with reduced features
            start_time = time.time()
            classifier.fit(X_reduced, y)
            training_time = time.time() - start_time
            
            self.nb_classifier = classifier
            
            st.success(f"Naive Bayes trained with optimized PCA + GaussianNB ({training_time:.1f}s)")
            return True
            
        except Exception as e:
            st.error(f"Error training Naive Bayes: {str(e)}")
            st.exception(e)
            return False
    
    def train_random_forest(self, X, y):
        """Train Random Forest with adaptive settings based on dataset size"""
        try:
            n_samples, n_features = X.shape
            st.write(f"Training Random Forest with adaptive settings for {n_samples:,} samples...")
            
            # Adaptive hyperparameters based on dataset size
            if n_samples > 50000:
                # Large dataset settings
                n_estimators = 100
                max_depth = 12
                min_samples_split = 10
                min_samples_leaf = 5
                st.info("Using large dataset settings (faster training)")
            elif n_samples > 10000:
                # Medium dataset settings
                n_estimators = 150
                max_depth = 15
                min_samples_split = 5
                min_samples_leaf = 2
                st.info("Using medium dataset settings (balanced)")
            else:
                # Small dataset settings
                n_estimators = 200
                max_depth = 20
                min_samples_split = 2
                min_samples_leaf = 1
                st.info("Using small dataset settings (higher performance)")
            
            # Create Random Forest with balanced class weights and optimized parameters
            rf = RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                min_samples_leaf=min_samples_leaf,
                class_weight='balanced',
                random_state=42,
                n_jobs=-1,
                bootstrap=True,
                max_features='sqrt',
                criterion='gini',
                warm_start=False,
                oob_score=True
            )
            
            # Use MultiOutputClassifier with balanced settings
            classifier = MultiOutputClassifier(rf, n_jobs=-1)
            
            # Fit the model with progress tracking
            start_time = time.time()
            classifier.fit(X, y)
            training_time = time.time() - start_time
            
            self.rf_classifier = classifier
            
            # Show OOB score if available
            try:
                if hasattr(classifier.estimators_[0], 'oob_score_'):
                    avg_oob = np.mean([est.oob_score_ for est in classifier.estimators_])
                    st.info(f"Average OOB Score: {avg_oob:.3f} (higher is better)")
            except:
                pass
            
            st.success(f"Random Forest trained with adaptive settings ({training_time:.1f}s)")
            return True
            
        except Exception as e:
            st.error(f"Error training Random Forest: {str(e)}")
            st.exception(e)
            return False
    
    def predict_single_text(self, text, bert_embedder, model_type='random_forest', threshold=0.5):
        """Standard single text prediction"""
        try:
            # Generate embedding
            embedding = bert_embedder.get_single_embedding(text)
            if embedding is None:
                st.error("Failed to generate embedding for text")
                return None
            
            # Reshape for prediction
            embedding = embedding.reshape(1, -1)
            
            # Make prediction with error handling
            try:
                if model_type == 'naive_bayes' and self.nb_classifier:
                    # Apply PCA transformation for NB
                    if self.pca is not None:
                        embedding = self.pca.transform(embedding)
                    probabilities = self.nb_classifier.predict_proba(embedding)
                elif model_type == 'random_forest' and self.rf_classifier:
                    probabilities = self.rf_classifier.predict_proba(embedding)
                else:
                    st.error(f"Model {model_type} not available or not trained")
                    return None
                    
            except Exception as pred_error:
                st.error(f"Prediction error: {str(pred_error)}")
                return None
            
            # Handle MultiOutputClassifier probability format properly
            if isinstance(probabilities, list):
                # Extract positive class probabilities
                emotion_probs = []
                for i, emotion_proba in enumerate(probabilities):
                    if i < len(self.emotion_labels):
                        if emotion_proba.shape[1] == 2:  # Binary classifier
                            emotion_probs.append(emotion_proba[0, 1])  # Positive class probability
                        else:
                            emotion_probs.append(emotion_proba[0, 0])  # Single value
                probabilities = np.array(emotion_probs)
            else:
                probabilities = probabilities[0]
            
            # Ensure we have the right number of probabilities
            probabilities = probabilities[:len(self.emotion_labels)]
            
            # Get top 3 emotions with proper confidence scores
            top_3_indices = np.argsort(probabilities)[-3:][::-1]
            top_3_emotions = [self.emotion_labels[i] for i in top_3_indices]
            top_3_probabilities = [float(probabilities[i]) for i in top_3_indices]
            
            # Create comprehensive results
            results = {
                'top_3_emotions': top_3_emotions,
                'top_3_probabilities': top_3_probabilities,
                'all_probabilities': {emotion: float(prob) for emotion, prob in zip(self.emotion_labels, probabilities)},
                'threshold_met': any(prob >= threshold for prob in top_3_probabilities)
            }
            
            return results
            
        except Exception as e:
            st.error(f"Error in single text prediction: {str(e)}")
            return None
    
    def predict_single_text_with_attention(self, text, bert_embedder, model_type='random_forest', 
                                         threshold=0.5, thresholds=None, return_attention=True):
        """Enhanced single text prediction with attention weights"""
        try:
            # Get basic prediction first
            basic_results = self.predict_single_text(text, bert_embedder, model_type, threshold)
            if not basic_results:
                return None
            
            # Add attention weights if requested
            if return_attention:
                try:
                    # Get attention weights from BERT embedder
                    attention_weights = bert_embedder.get_attention_weights(text, basic_results['top_3_emotions'][0])
                    if attention_weights:
                        basic_results['attention_weights'] = attention_weights
                except Exception as e:
                    # Graceful fallback if attention extraction fails
                    basic_results['attention_weights'] = None
            
            return basic_results
            
        except Exception as e:
            st.error(f"Error in enhanced single text prediction: {str(e)}")
            return None
    
    def predict_batch(self, df, bert_embedder, model_type='random_forest', threshold=0.3):
        """Standard batch prediction"""
        try:
            if 'text' not in df.columns:
                st.error("DataFrame must contain 'text' column")
                return None
            
            # Clean and prepare texts
            texts = df['text'].tolist()
            texts = [str(text).strip() for text in texts if text and str(text).strip()]
            
            if not texts:
                st.error("No valid texts found after cleaning")
                return None
            
            st.info(f"Processing {len(texts)} texts with {model_type.replace('_', ' ').title()}...")
            
            # Show processing estimate for large batches
            if len(texts) > 10000:
                processing_est = bert_embedder.estimate_processing_requirements(len(texts))
                st.info(f"Large batch: {processing_est['recommendation']}")
            
            # Generate embeddings for all texts
            embeddings = bert_embedder.generate_embeddings(texts)
            if embeddings is None:
                st.error("Failed to generate embeddings")
                return None
            
            # Make predictions with progress tracking
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            try:
                status_text.text("Making predictions...")
                
                if model_type == 'naive_bayes' and self.nb_classifier:
                    # Apply PCA transformation for NB
                    if self.pca is not None:
                        status_text.text("Applying PCA transformation...")
                        embeddings = self.pca.transform(embeddings)
                    batch_probabilities = self.nb_classifier.predict_proba(embeddings)
                elif model_type == 'random_forest' and self.rf_classifier:
                    batch_probabilities = self.rf_classifier.predict_proba(embeddings)
                else:
                    st.error(f"Model {model_type} not available")
                    return None
                
                progress_bar.progress(0.5)
                
            except Exception as e:
                st.error(f"Prediction error: {str(e)}")
                return None
            
            # Process predictions properly with error handling
            results = []
            
            try:
                status_text.text("Processing predictions...")
                
                # Handle MultiOutputClassifier format
                if isinstance(batch_probabilities, list):
                    # Convert list of arrays to matrix
                    proba_matrix = np.zeros((len(texts), len(self.emotion_labels)))
                    for i, emotion_proba in enumerate(batch_probabilities):
                        if i < len(self.emotion_labels):
                            if emotion_proba.shape[1] == 2:  # Binary classifier
                                proba_matrix[:, i] = emotion_proba[:, 1]  # Positive class
                            else:
                                proba_matrix[:, i] = emotion_proba[:, 0]  # Single value
                else:
                    proba_matrix = batch_probabilities
                
                progress_bar.progress(0.75)
                
                # Process each text with error handling
                successful_predictions = 0
                for i, (text, probabilities) in enumerate(zip(texts, proba_matrix)):
                    try:
                        # Ensure probabilities are valid
                        if np.any(np.isnan(probabilities)) or np.any(np.isinf(probabilities)):
                            st.warning(f"Invalid probabilities for text {i}, skipping...")
                            continue
                        
                        # Find emotions above threshold
                        above_threshold_mask = probabilities >= threshold
                        above_threshold_emotions = np.where(above_threshold_mask)[0]
                        
                        if len(above_threshold_emotions) > 0:
                            # Use highest emotion above threshold
                            best_idx = above_threshold_emotions[np.argmax(probabilities[above_threshold_emotions])]
                        else:
                            # Use highest overall emotion
                            best_idx = np.argmax(probabilities)
                        
                        top_emotion = self.emotion_labels[best_idx]
                        top_confidence = float(probabilities[best_idx])
                        
                        # Get top 3 emotions
                        top_3_idx = np.argsort(probabilities)[-3:][::-1]
                        top_3_emotions = [self.emotion_labels[idx] for idx in top_3_idx]
                        top_3_scores = [float(probabilities[idx]) for idx in top_3_idx]
                        
                        results.append({
                            'text': text[:100] + "..." if len(text) > 100 else text,
                            'top_emotion': top_emotion,
                            'confidence': top_confidence,
                            'top_3_emotions': ', '.join(top_3_emotions),
                            'top_3_scores': ', '.join([f"{score:.3f}" for score in top_3_scores])
                        })
                        
                        successful_predictions += 1
                        
                    except Exception as e:
                        st.warning(f"Error processing text {i}: {str(e)}")
                        continue
                
                progress_bar.progress(1.0)
                status_text.text(f"Completed: {successful_predictions}/{len(texts)} successful predictions")
                
            except Exception as e:
                st.error(f"Error processing batch results: {str(e)}")
                return None
            
            if not results:
                st.error("No results generated - check your text data and model")
                return None
            
            # Create results DataFrame
            results_df = pd.DataFrame(results)
            
            # Show emotion distribution with enhanced statistics
            emotion_dist = results_df['top_emotion'].value_counts()
            st.subheader("Predicted Emotion Distribution")
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                total_results = len(results_df)
                st.metric("Total Processed", f"{total_results:,}")
            
            with col2:
                avg_confidence = results_df['confidence'].mean()
                st.metric("Avg Confidence", f"{avg_confidence:.1%}")
            
            with col3:
                unique_emotions = results_df['top_emotion'].nunique()
                st.metric("Unique Emotions", unique_emotions)
            
            with col4:
                success_rate = (len(results_df) / len(texts)) * 100
                st.metric("Success Rate", f"{success_rate:.1f}%")
            
            # Show top emotions with better formatting
            st.write("**Top 5 Predicted Emotions:**")
            top_5_emotions = emotion_dist.head(5)
            for emotion, count in top_5_emotions.items():
                pct = (count / len(results_df)) * 100
                st.write(f"   • **{emotion.title()}**: {count:,} samples ({pct:.1f}%)")
            
            # Add processing quality note
            if avg_confidence < 0.5:
                st.info("Consider adjusting prediction thresholds for better results.")
            elif avg_confidence > 0.8:
                st.success("High confidence predictions achieved!")
            
            return results_df
            
        except Exception as e:
            st.error(f"Error in batch prediction: {str(e)}")
            st.exception(e)
            return None
    
    def predict_batch_with_trajectory(self, df, bert_embedder, model_type='random_forest', 
                                    threshold=0.3, thresholds=None):
        """Enhanced batch prediction with trajectory tracking"""
        try:
            # Get basic batch prediction
            results_df = self.predict_batch(df, bert_embedder, model_type, threshold)
            if results_df is None:
                return None
            
            # Add trajectory index for time-series analysis
            results_df['trajectory_index'] = range(len(results_df))
            
            # Add timestamp if not present
            if 'timestamp' not in results_df.columns:
                results_df['timestamp'] = pd.date_range(start='2024-01-01', periods=len(results_df), freq='H')
            
            return results_df
            
        except Exception as e:
            st.error(f"Error in enhanced batch prediction: {str(e)}")
            return None
    
    def save_model_components(self, dataset_name="demo"):
        """Save individual model components using joblib for better compatibility"""
        try:
            cache_dir = "demo_cache/"
            os.makedirs(cache_dir, exist_ok=True)
            
            saved_components = []
            
            # Save Naive Bayes model and PCA
            if self.nb_classifier is not None:
                nb_path = f"{cache_dir}/nb_classifier_{dataset_name}.joblib"
                joblib.dump(self.nb_classifier, nb_path)
                saved_components.append("Naive Bayes")
                
                if self.pca is not None:
                    pca_path = f"{cache_dir}/pca_{dataset_name}.joblib"
                    joblib.dump(self.pca, pca_path)
                    saved_components.append("PCA")
            
            # Save Random Forest model
            if self.rf_classifier is not None:
                rf_path = f"{cache_dir}/rf_classifier_{dataset_name}.joblib"
                joblib.dump(self.rf_classifier, rf_path)
                saved_components.append("Random Forest")
            
            return True, saved_components
            
        except Exception as e:
            st.error(f"Error saving model components: {str(e)}")
            return False, []
    
    def load_model_components(self, dataset_name="demo"):
        """Load individual model components using joblib"""
        try:
            cache_dir = "demo_cache/"
            loaded_components = []
            
            # Load Naive Bayes model
            nb_path = f"{cache_dir}/nb_classifier_{dataset_name}.joblib"
            if os.path.exists(nb_path):
                self.nb_classifier = joblib.load(nb_path)
                loaded_components.append("Naive Bayes")
            
            # Load PCA
            pca_path = f"{cache_dir}/pca_{dataset_name}.joblib"
            if os.path.exists(pca_path):
                self.pca = joblib.load(pca_path)
                loaded_components.append("PCA")
            
            # Load Random Forest model
            rf_path = f"{cache_dir}/rf_classifier_{dataset_name}.joblib"
            if os.path.exists(rf_path):
                self.rf_classifier = joblib.load(rf_path)
                loaded_components.append("Random Forest")
            
            return True, loaded_components
            
        except Exception as e:
            st.error(f"Error loading model components: {str(e)}")
            return False, []
    
    def get_model_info(self):
        """Get information about trained models"""
        info = {
            'naive_bayes_trained': self.nb_classifier is not None,
            'random_forest_trained': self.rf_classifier is not None,
            'pca_applied': self.pca is not None
        }
        
        if self.pca is not None:
            info['pca_components'] = self.pca.n_components_
            info['pca_explained_variance'] = f"{self.pca.explained_variance_ratio_.sum():.1%}"
        
        return info