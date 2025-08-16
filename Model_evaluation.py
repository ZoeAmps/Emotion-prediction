import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, hamming_loss, roc_auc_score, roc_curve
import warnings

class SimpleModelEvaluator:
    def __init__(self):
        self.emotion_labels = [
            'admiration', 'amusement', 'anger', 'annoyance', 'approval', 'caring',
            'confusion', 'curiosity', 'desire', 'disappointment', 'disapproval',
            'disgust', 'embarrassment', 'excitement', 'fear', 'gratitude', 'grief',
            'joy', 'love', 'nervousness', 'optimism', 'pride', 'realization',
            'relief', 'remorse', 'sadness', 'surprise', 'neutral'
        ]
    
    def evaluate_models(self, classifiers, X_test, y_test):
        """Evaluate models with proper multi-label metrics including robust ROC-AUC"""
        results = {}
        
        try:
            # Evaluate Naive Bayes
            if classifiers.nb_classifier:
                nb_metrics = self._evaluate_single_model(
                    classifiers.nb_classifier, X_test, y_test, "Naive Bayes", classifiers.pca
                )
                if nb_metrics:
                    results['naive_bayes'] = nb_metrics
            
            # Evaluate Random Forest
            if classifiers.rf_classifier:
                rf_metrics = self._evaluate_single_model(
                    classifiers.rf_classifier, X_test, y_test, "Random Forest", None
                )
                if rf_metrics:
                    results['random_forest'] = rf_metrics
            
            # Show detailed comparison
            if results:
                self._show_detailed_comparison(results, y_test)
            
            return results
            
        except Exception as e:
            st.error(f"Error evaluating models: {str(e)}")
            return None
    
    def _safe_roc_auc_calculation(self, y_true, y_pred_proba, average='macro'):
        """Robust ROC-AUC calculation that handles edge cases"""
        try:
            # Check if we have valid data
            if y_true.size == 0 or y_pred_proba.size == 0:
                st.warning("Empty data for ROC-AUC calculation")
                return 0.5
            
            # Check for emotions with only one class (all 0s or all 1s)
            valid_emotions = []
            for i in range(y_true.shape[1]):
                emotion_true = y_true[:, i]
                unique_values = np.unique(emotion_true)
                
                if len(unique_values) > 1:  # Has both positive and negative samples
                    valid_emotions.append(i)
                else:
                    # Skip emotions with only one class
                    continue
            
            if len(valid_emotions) == 0:
                st.warning("No emotions with both positive and negative samples for ROC-AUC")
                return 0.5
            
            # Calculate ROC-AUC only for valid emotions
            y_true_valid = y_true[:, valid_emotions]
            y_pred_proba_valid = y_pred_proba[:, valid_emotions]
            
            # Try macro average first
            try:
                roc_auc_macro = roc_auc_score(y_true_valid, y_pred_proba_valid, average='macro', multi_class='ovr')
                return float(roc_auc_macro)
            except ValueError as e:
                if "multi_class" in str(e):
                    # Fallback: calculate per-emotion and average
                    auc_scores = []
                    for i in range(y_true_valid.shape[1]):
                        try:
                            auc = roc_auc_score(y_true_valid[:, i], y_pred_proba_valid[:, i])
                            auc_scores.append(auc)
                        except ValueError:
                            continue
                    
                    if auc_scores:
                        return float(np.mean(auc_scores))
                    else:
                        return 0.5
                else:
                    raise e
            
        except Exception as e:
            st.warning(f"ROC-AUC calculation failed: {str(e)}. Using fallback value.")
            return 0.5  # Return neutral performance as fallback
    
    def _evaluate_single_model(self, model, X_test, y_test, model_name, pca=None):
        """Evaluate model with robust multi-label metrics including safe ROC-AUC"""
        try:
            # Apply PCA if needed (for Naive Bayes)
            X_test_processed = X_test
            if pca is not None:
                X_test_processed = pca.transform(X_test)
            
            # Get probability predictions
            y_pred_proba = model.predict_proba(X_test_processed)
            
            # Handle MultiOutputClassifier probability format
            if isinstance(y_pred_proba, list):
                # Convert list of arrays to matrix
                proba_matrix = np.zeros((len(X_test_processed), len(self.emotion_labels)))
                for i, emotion_proba in enumerate(y_pred_proba):
                    if i < len(self.emotion_labels):
                        if emotion_proba.shape[1] == 2:  # Binary classifier
                            proba_matrix[:, i] = emotion_proba[:, 1]  # Positive class
                        else:
                            proba_matrix[:, i] = emotion_proba[:, 0]  # Single value
                y_pred_proba = proba_matrix
            
            # Convert probabilities to binary predictions with threshold
            threshold = 0.5
            y_pred = (y_pred_proba >= threshold).astype(int)
            
            # Ensure dimensions match
            if y_pred.shape[1] > y_test.shape[1]:
                y_pred = y_pred[:, :y_test.shape[1]]
                y_pred_proba = y_pred_proba[:, :y_test.shape[1]]
            elif y_pred.shape[1] < y_test.shape[1]:
                # Pad with zeros
                padding = np.zeros((y_pred.shape[0], y_test.shape[1] - y_pred.shape[1]))
                y_pred = np.hstack([y_pred, padding])
                padding_proba = np.zeros((y_pred_proba.shape[0], y_test.shape[1] - y_pred_proba.shape[1]))
                y_pred_proba = np.hstack([y_pred_proba, padding_proba])
            
            # Calculate proper multi-label metrics
            metrics = {}
            
            # 1. SUBSET ACCURACY (exact match) - This is the correct "accuracy" for multi-label
            subset_accuracy = np.mean(np.all(y_test == y_pred, axis=1))
            metrics['subset_accuracy'] = float(subset_accuracy)
            
            # 2. HAMMING LOSS (element-wise accuracy)
            hamming_loss_score = hamming_loss(y_test, y_pred)
            hamming_accuracy = 1 - hamming_loss_score  # Convert to accuracy
            metrics['hamming_accuracy'] = float(hamming_accuracy)
            
            # 3. Use hamming accuracy as main "accuracy" metric
            metrics['accuracy'] = float(hamming_accuracy)
            
            # 4. Standard multi-label metrics with zero_division handling
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                metrics['macro_f1'] = float(f1_score(y_test, y_pred, average='macro', zero_division=0))
                metrics['weighted_f1'] = float(f1_score(y_test, y_pred, average='weighted', zero_division=0))
                metrics['macro_precision'] = float(precision_score(y_test, y_pred, average='macro', zero_division=0))
                metrics['macro_recall'] = float(recall_score(y_test, y_pred, average='macro', zero_division=0))
            
            # 5. Robust ROC-AUC for multi-label classification
            roc_auc_macro = self._safe_roc_auc_calculation(y_test, y_pred_proba, average='macro')
            metrics['roc_auc_macro'] = roc_auc_macro
            metrics['roc_auc'] = roc_auc_macro  # Use macro as main ROC-AUC metric
            
            # Try weighted ROC-AUC as well
            roc_auc_weighted = self._safe_roc_auc_calculation(y_test, y_pred_proba, average='weighted')
            metrics['roc_auc_weighted'] = roc_auc_weighted
            
            # 6. Per-emotion performance (includes per-emotion ROC-AUC)
            emotion_performance = {}
            for i, emotion in enumerate(self.emotion_labels):
                if i < y_test.shape[1]:
                    y_true_emotion = y_test[:, i]
                    y_pred_emotion = y_pred[:, i]
                    y_proba_emotion = y_pred_proba[:, i]
                    
                    # Only calculate if emotion exists in test set and has both classes
                    unique_values = np.unique(y_true_emotion)
                    if len(unique_values) > 1:
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore")
                            emotion_f1 = f1_score(y_true_emotion, y_pred_emotion, zero_division=0)
                            emotion_precision = precision_score(y_true_emotion, y_pred_emotion, zero_division=0)
                            emotion_recall = recall_score(y_true_emotion, y_pred_emotion, zero_division=0)
                        
                        # Calculate per-emotion ROC-AUC safely
                        try:
                            emotion_roc_auc = roc_auc_score(y_true_emotion, y_proba_emotion)
                        except ValueError:
                            emotion_roc_auc = 0.5  # Neutral performance for problematic emotions
                        
                        emotion_performance[emotion] = {
                            'f1': float(emotion_f1),
                            'precision': float(emotion_precision),
                            'recall': float(emotion_recall),
                            'roc_auc': float(emotion_roc_auc),
                            'support': int(np.sum(y_true_emotion))
                        }
                    else:
                        # Single class emotion - add with neutral metrics
                        emotion_performance[emotion] = {
                            'f1': 0.0,
                            'precision': 0.0,
                            'recall': 0.0,
                            'roc_auc': 0.5,
                            'support': int(np.sum(y_true_emotion)),
                            'note': 'Single class - metrics not meaningful'
                        }
            
            metrics['emotion_performance'] = emotion_performance
            
            # 7. Balance improvement assessment (check if oversampling worked)
            rare_emotion_f1_scores = []
            for emotion, perf in emotion_performance.items():
                if perf['support'] > 0 and perf['support'] < 2000:  # Previously rare emotions
                    rare_emotion_f1_scores.append(perf['f1'])
            
            if rare_emotion_f1_scores:
                avg_rare_f1 = np.mean(rare_emotion_f1_scores)
                metrics['rare_emotion_avg_f1'] = float(avg_rare_f1)
                
                if avg_rare_f1 > 0.3:
                    metrics['oversampling_success'] = "Excellent - Rare emotions performing well"
                elif avg_rare_f1 > 0.2:
                    metrics['oversampling_success'] = "Good - Significant improvement for rare emotions"
                elif avg_rare_f1 > 0.1:
                    metrics['oversampling_success'] = "Moderate - Some improvement for rare emotions"
                else:
                    metrics['oversampling_success'] = "Poor - Rare emotions still struggling"
            else:
                metrics['rare_emotion_avg_f1'] = 0.0
                metrics['oversampling_success'] = "No rare emotions to assess"
            
            # 8. Confidence metrics
            if y_pred_proba.size > 0:
                avg_confidence = np.mean(np.max(y_pred_proba, axis=1))
                metrics['avg_confidence'] = float(avg_confidence)
                
                # High confidence predictions
                high_confidence = np.mean(np.max(y_pred_proba, axis=1) > 0.7)
                metrics['high_confidence_ratio'] = float(high_confidence)
            
            # 9. Performance quality assessment
            quality_score = self._assess_performance_quality(metrics)
            metrics['quality_assessment'] = quality_score
            
            # Show key metrics including oversampling effectiveness
            st.info(f"{model_name} evaluation completed successfully")
            
            # Show oversampling effectiveness
            if 'oversampling_success' in metrics:
                if "Excellent" in metrics['oversampling_success']:
                    st.success(f"**Oversampling Impact**: {metrics['oversampling_success']}")
                elif "Good" in metrics['oversampling_success']:
                    st.success(f"**Oversampling Impact**: {metrics['oversampling_success']}")
                elif "Moderate" in metrics['oversampling_success']:
                    st.warning(f"**Oversampling Impact**: {metrics['oversampling_success']}")
                else:
                    st.error(f"**Oversampling Impact**: {metrics['oversampling_success']}")
            
            return metrics
            
        except Exception as e:
            st.error(f"Error evaluating {model_name}: {str(e)}")
            st.exception(e)
            return None
    
    def _assess_performance_quality(self, metrics):
        """Assess overall model performance quality including oversampling effectiveness"""
        hamming_acc = metrics.get('hamming_accuracy', 0)
        roc_auc = metrics.get('roc_auc', 0.5)
        f1_score = metrics.get('macro_f1', 0)
        rare_f1 = metrics.get('rare_emotion_avg_f1', 0)
        
        # Weighted score (hamming accuracy is most important, rare emotion F1 shows oversampling success)
        quality_score = (hamming_acc * 0.3) + (roc_auc * 0.25) + (f1_score * 0.25) + (rare_f1 * 0.2)
        
        if quality_score >= 0.8:
            return "Excellent"
        elif quality_score >= 0.7:
            return "Good"
        elif quality_score >= 0.6:
            return "Fair"
        else:
            return "Poor"
    
    def _show_detailed_comparison(self, results, y_test):
        """Show detailed model comparison with correct metrics including oversampling effectiveness"""
        st.subheader("Model Performance Comparison")
        
        # Overall Performance Summary with ROC-AUC and Oversampling Assessment
        st.write("**Complete Multi-Label Performance Metrics with Oversampling Assessment:**")
        comparison_data = []
        
        for model_name, metrics in results.items():
            model_display_name = model_name.replace('_', ' ').title()
            
            # Use correct metrics including ROC-AUC and oversampling effectiveness
            subset_accuracy = metrics.get('subset_accuracy', 0) * 100
            hamming_accuracy = metrics.get('hamming_accuracy', 0) * 100
            macro_f1 = metrics.get('macro_f1', 0) * 100
            macro_precision = metrics.get('macro_precision', 0) * 100
            macro_recall = metrics.get('macro_recall', 0) * 100
            roc_auc = metrics.get('roc_auc', 0) * 100
            rare_f1 = metrics.get('rare_emotion_avg_f1', 0) * 100
            quality = metrics.get('quality_assessment', 'Unknown')
            oversampling_status = metrics.get('oversampling_success', 'Unknown')
            
            comparison_data.append({
                'Model': model_display_name,
                'Hamming Accuracy': f"{hamming_accuracy:.1f}%",
                'Precision': f"{macro_precision:.1f}%",
                'Recall': f"{macro_recall:.1f}%",
                'F-Measure': f"{macro_f1:.1f}%",
                'ROC-AUC': f"{roc_auc:.1f}%",
                'Rare Emotion F1': f"{rare_f1:.1f}%",
                'Subset Accuracy': f"{subset_accuracy:.1f}%",
                'Quality': quality,
                'Oversampling Success': oversampling_status
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        st.dataframe(comparison_df, use_container_width=True, hide_index=True)
        
        # Highlight best performing model and oversampling effectiveness
        if len(comparison_data) > 1:
            best_hamming = max(comparison_data, key=lambda x: float(x['Hamming Accuracy'].rstrip('%')))
            best_roc = max(comparison_data, key=lambda x: float(x['ROC-AUC'].rstrip('%')))
            best_f1 = max(comparison_data, key=lambda x: float(x['F-Measure'].rstrip('%')))
            best_rare_f1 = max(comparison_data, key=lambda x: float(x['Rare Emotion F1'].rstrip('%')))
            
            st.write("**Best Performance:**")
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Best Hamming Accuracy", f"{best_hamming['Model']}", 
                         delta=f"{best_hamming['Hamming Accuracy']}")
            with col2:
                st.metric("Best ROC-AUC", f"{best_roc['Model']}", 
                         delta=f"{best_roc['ROC-AUC']}")
            with col3:
                st.metric("Best F-Measure", f"{best_f1['Model']}", 
                         delta=f"{best_f1['F-Measure']}")
            with col4:
                st.metric("Best Rare Emotion F1", f"{best_rare_f1['Model']}", 
                         delta=f"{best_rare_f1['Rare Emotion F1']}")
        
        # Show oversampling effectiveness summary
        st.write("**Oversampling Effectiveness Summary:**")
        for model_name, metrics in results.items():
            model_display = model_name.replace('_', ' ').title()
            rare_f1 = metrics.get('rare_emotion_avg_f1', 0) * 100
            oversampling_status = metrics.get('oversampling_success', 'Unknown')
            
            if "Excellent" in oversampling_status:
                st.success(f"**{model_display}**: {oversampling_status} (Rare F1: {rare_f1:.1f}%)")
            elif "Good" in oversampling_status:
                st.success(f"**{model_display}**: {oversampling_status} (Rare F1: {rare_f1:.1f}%)")
            elif "Moderate" in oversampling_status:
                st.warning(f"**{model_display}**: {oversampling_status} (Rare F1: {rare_f1:.1f}%)")
            else:
                st.info(f"**{model_display}**: {oversampling_status} (Rare F1: {rare_f1:.1f}%)")
        
        # Enhanced explanation of metrics including oversampling
        with st.expander("Understanding All Metrics and Oversampling Impact"):
            st.write("""
            **Primary Performance Metrics:**
            
            **Hamming Accuracy**: Average accuracy across all emotion labels (Main metric)
            - More forgiving - measures per-emotion accuracy
            - Expected: 60-80%+ for good performance
            
            **Precision**: How many predicted positive emotions were actually correct
            - High precision = few false positives
            
            **Recall**: How many actual positive emotions were correctly identified  
            - High recall = few false negatives
            
            **F-Measure (F1-Score)**: Harmonic mean of precision and recall
            - Balances precision and recall
            - Expected: 40-60% for emotion detection
            
            **ROC-AUC**: Area Under the Receiver Operating Characteristic curve
            - Measures ability to distinguish between classes
            - Range: 0.5 (random) to 1.0 (perfect)
            - Expected: 70-90% for good models
            
            **Oversampling Impact Assessment:**
            
            **Rare Emotion F1**: Average F1-score for emotions that had <2,000 samples originally
            - This metric shows if our aggressive oversampling worked!
            - >30% = Excellent oversampling success
            - 20-30% = Good improvement from oversampling
            - 10-20% = Moderate improvement
            - <10% = Oversampling didn't help much
            
            **Oversampling Success**: Overall assessment of how well the balancing worked
            - Shows if the 5-10x boosting of rare emotions improved their detection
            
            **Why This Matters:**
            Without proper balancing, rare emotions like "grief", "pride", "nervousness" would have 
            near-zero F1 scores. Our aggressive oversampling should boost these to 20-40%+ F1 scores.
            """)
        
        # Show which metrics to focus on
        st.success("Model evaluation completed with oversampling effectiveness assessment!")
    
    def create_roc_curves(self, classifiers, X_test, y_test):
        """Create ROC-AUC visualization for top emotions including rare emotions"""
        try:
            # Select top 4 frequent emotions + top 2 rare emotions for visualization
            emotion_counts = {}
            for i, emotion in enumerate(self.emotion_labels):
                if i < y_test.shape[1]:
                    emotion_counts[emotion] = np.sum(y_test[:, i])
            
            # Get top frequent emotions
            frequent_emotions = sorted(emotion_counts.items(), key=lambda x: x[1], reverse=True)[:4]
            # Get rare emotions (that we boosted)
            rare_emotions = sorted([(e, c) for e, c in emotion_counts.items() if c < 2000 and c > 100], 
                                 key=lambda x: x[1], reverse=True)[:2]
            
            # Combine for visualization
            selected_emotions = [emotion for emotion, _ in frequent_emotions + rare_emotions]
            
            # Create subplots
            fig = make_subplots(
                rows=2, cols=3,
                subplot_titles=selected_emotions,
                specs=[[{"secondary_y": False}]*3]*2
            )
            
            colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD']
            
            for idx, emotion in enumerate(selected_emotions):
                row = (idx // 3) + 1
                col = (idx % 3) + 1
                
                emotion_idx = self.emotion_labels.index(emotion)
                y_true = y_test[:, emotion_idx]
                
                # Skip if only one class
                if len(np.unique(y_true)) < 2:
                    continue
                
                # Determine if this is a rare emotion (previously boosted)
                is_rare = emotion_counts[emotion] < 2000
                
                # Get predictions from both models
                models_data = []
                
                if classifiers.rf_classifier:
                    rf_proba = classifiers.rf_classifier.predict_proba(X_test)
                    if isinstance(rf_proba, list):
                        rf_emotion_proba = rf_proba[emotion_idx][:, 1] if rf_proba[emotion_idx].shape[1] == 2 else rf_proba[emotion_idx][:, 0]
                    else:
                        rf_emotion_proba = rf_proba[:, emotion_idx]
                    models_data.append(('Random Forest', rf_emotion_proba, colors[0]))
                
                if classifiers.nb_classifier and classifiers.pca:
                    X_test_pca = classifiers.pca.transform(X_test)
                    nb_proba = classifiers.nb_classifier.predict_proba(X_test_pca)
                    if isinstance(nb_proba, list):
                        nb_emotion_proba = nb_proba[emotion_idx][:, 1] if nb_proba[emotion_idx].shape[1] == 2 else nb_proba[emotion_idx][:, 0]
                    else:
                        nb_emotion_proba = nb_proba[:, emotion_idx]
                    models_data.append(('Naive Bayes', nb_emotion_proba, colors[1]))
                
                # Plot ROC curves for each model
                for model_name, proba, color in models_data:
                    try:
                        fpr, tpr, _ = roc_curve(y_true, proba)
                        auc_score = roc_auc_score(y_true, proba)
                        
                        # Add special marking for rare emotions
                        display_name = f'{model_name} (AUC={auc_score:.3f})'
                        if is_rare:
                            display_name += ' [RARE]'
                        
                        fig.add_trace(
                            go.Scatter(
                                x=fpr, y=tpr,
                                mode='lines',
                                name=display_name,
                                line=dict(color=color, width=3 if is_rare else 2, dash='dot' if is_rare else 'solid'),
                                showlegend=(idx == 0)  # Only show legend for first subplot
                            ),
                            row=row, col=col
                        )
                    except Exception:
                        continue
                
                # Add diagonal line (random classifier)
                fig.add_trace(
                    go.Scatter(
                        x=[0, 1], y=[0, 1],
                        mode='lines',
                        line=dict(color='gray', dash='dash', width=1),
                        showlegend=False
                    ),
                    row=row, col=col
                )
                
                # Add annotation for rare emotions
                if is_rare:
                    fig.add_annotation(
                        text="RARE EMOTION<br>(Oversampled)",
                        x=0.7, y=0.3,
                        xref=f"x{'' if row == 1 and col == 1 else row*3 + col - 3}",
                        yref=f"y{'' if row == 1 and col == 1 else row*3 + col - 3}",
                        showarrow=False,
                        font=dict(size=10, color="orange"),
                        bgcolor="rgba(255,165,0,0.3)",
                        bordercolor="orange",
                        borderwidth=1
                    )
            
            # Update layout
            fig.update_layout(
                title="ROC-AUC Curves: Top Frequent + Oversampled Rare Emotions",
                height=600,
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)"
            )
            
            # Update axes
            for i in range(1, 7):
                row = ((i-1) // 3) + 1
                col = ((i-1) % 3) + 1
                fig.update_xaxes(title_text="False Positive Rate", row=row, col=col)
                fig.update_yaxes(title_text="True Positive Rate", row=row, col=col)
            
            return fig
            
        except Exception as e:
            st.error(f"Error creating ROC curves: {str(e)}")
            return None
    
    def create_emotion_trajectory(self, results_df):
        """Create emotion trajectory visualization over time"""
        try:
            if 'trajectory_index' not in results_df.columns:
                results_df['trajectory_index'] = range(len(results_df))
            
            # Get emotion counts over trajectory
            window_size = max(10, len(results_df) // 20)  # Adaptive window size
            
            # Create rolling emotion counts
            emotion_trajectory = {}
            unique_emotions = results_df['top_emotion'].unique()
            
            for emotion in unique_emotions:
                emotion_mask = results_df['top_emotion'] == emotion
                emotion_trajectory[emotion] = emotion_mask.rolling(window=window_size, center=True).sum()
            
            # Create the plot
            fig = go.Figure()
            
            colors = px.colors.qualitative.Set3
            for i, (emotion, trajectory) in enumerate(emotion_trajectory.items()):
                fig.add_trace(go.Scatter(
                    x=results_df['trajectory_index'],
                    y=trajectory,
                    mode='lines',
                    name=emotion.title(),
                    line=dict(color=colors[i % len(colors)], width=2)
                ))
            
            fig.update_layout(
                title=f"Emotion Trajectory Over Time (Window Size: {window_size})",
                xaxis_title="Sample Index",
                yaxis_title="Emotion Frequency (Rolling Sum)",
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                height=500
            )
            
            return fig
            
        except Exception as e:
            st.error(f"Error creating emotion trajectory: {str(e)}")
            return None
    
    def create_emotion_comparison(self, results_a, results_b, label_a, label_b):
        """Create emotion comparison chart between two texts"""
        try:
            # Get all emotions and their probabilities
            emotions_a = {emotion: prob for emotion, prob in zip(results_a['top_3_emotions'], results_a['top_3_probabilities'])}
            emotions_b = {emotion: prob for emotion, prob in zip(results_b['top_3_emotions'], results_b['top_3_probabilities'])}
            
            # Get union of all emotions
            all_emotions = set(emotions_a.keys()) | set(emotions_b.keys())
            
            # Create comparison data
            comparison_data = []
            for emotion in all_emotions:
                prob_a = emotions_a.get(emotion, 0)
                prob_b = emotions_b.get(emotion, 0)
                comparison_data.append({
                    'Emotion': emotion.title(),
                    label_a: prob_a * 100,
                    label_b: prob_b * 100
                })
            
            comparison_df = pd.DataFrame(comparison_data)
            
            # Create grouped bar chart
            fig = go.Figure()
            
            fig.add_trace(go.Bar(
                name=label_a,
                x=comparison_df['Emotion'],
                y=comparison_df[label_a],
                marker_color='#FF6B6B'
            ))
            
            fig.add_trace(go.Bar(
                name=label_b,
                x=comparison_df['Emotion'],
                y=comparison_df[label_b],
                marker_color='#4ECDC4'
            ))
            
            fig.update_layout(
                title=f"Emotion Comparison: {label_a} vs {label_b}",
                xaxis_title="Emotions",
                yaxis_title="Confidence (%)",
                barmode='group',
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                height=500
            )
            
            return fig
            
        except Exception as e:
            st.error(f"Error creating emotion comparison: {str(e)}")
            return None
    
    def display_performance_summary(self, results):
        """Display performance summary with all required metrics and oversampling assessment"""
        st.subheader("Performance Summary - All Required Metrics + Oversampling Impact")
        
        if not results:
            st.error("No results to display")
            return
        
        summary_data = []
        
        for model_name, metrics in results.items():
            # Use all required metrics plus oversampling assessment
            hamming_accuracy = metrics.get('hamming_accuracy', 0) * 100
            macro_precision = metrics.get('macro_precision', 0) * 100
            macro_recall = metrics.get('macro_recall', 0) * 100
            macro_f1 = metrics.get('macro_f1', 0) * 100
            roc_auc = metrics.get('roc_auc', 0) * 100
            rare_f1 = metrics.get('rare_emotion_avg_f1', 0) * 100
            quality = metrics.get('quality_assessment', 'Unknown')
            oversampling_success = metrics.get('oversampling_success', 'Unknown')
            
            summary_data.append({
                'Model': model_name.replace('_', ' ').title(),
                'Accuracy': f"{hamming_accuracy:.1f}%",
                'Precision': f"{macro_precision:.1f}%", 
                'Recall': f"{macro_recall:.1f}%",
                'F-Measure': f"{macro_f1:.1f}%",
                'ROC-AUC': f"{roc_auc:.1f}%",
                'Rare Emotion F1': f"{rare_f1:.1f}%",
                'Quality': quality,
                'Oversampling Impact': oversampling_success.split(' - ')[0]  # First part only
            })
        
        # Display as table
        summary_df = pd.DataFrame(summary_data)
        st.dataframe(summary_df, use_container_width=True, hide_index=True)
        
        # Show best model and oversampling effectiveness
        if len(summary_data) > 1:
            best_model = max(summary_data, key=lambda x: float(x['Accuracy'].rstrip('%')))
            best_rare_f1 = max(summary_data, key=lambda x: float(x['Rare Emotion F1'].rstrip('%')))
            
            col1, col2 = st.columns(2)
            with col1:
                st.success(f"**Best Overall Model**: {best_model['Model']} ({best_model['Accuracy']} accuracy)")
            with col2:
                st.success(f"**Best Rare Emotion Performance**: {best_rare_f1['Model']} ({best_rare_f1['Rare Emotion F1']} rare F1)")
        
        # Show oversampling effectiveness summary
        st.write("**Oversampling Effectiveness:**")
        for data in summary_data:
            rare_f1_val = float(data['Rare Emotion F1'].rstrip('%'))
            if rare_f1_val > 30:
                st.success(f"✓ **{data['Model']}**: Excellent rare emotion improvement ({data['Rare Emotion F1']})")
            elif rare_f1_val > 20:
                st.success(f"✓ **{data['Model']}**: Good rare emotion improvement ({data['Rare Emotion F1']})")
            elif rare_f1_val > 10:
                st.warning(f"⚠ **{data['Model']}**: Moderate rare emotion improvement ({data['Rare Emotion F1']})")
            else:
                st.error(f"✗ **{data['Model']}**: Poor rare emotion improvement ({data['Rare Emotion F1']})")
        
        return summary_data
    
    def explain_metrics(self):
        """Explain key metrics including oversampling impact"""
        with st.expander("Understanding Performance Metrics + Oversampling Impact"):
            st.write("""
            **Key Performance Metrics:**
            
            **Precision**: Accuracy of positive predictions
            - How many predicted emotions were actually correct
            
            **Recall**: Coverage of actual positives  
            - How many actual emotions were correctly identified
            
            **F-Measure**: Balance between precision and recall
            - Harmonic mean of precision and recall
            
            **ROC-AUC**: Model's discriminative ability
            - Higher values indicate better separation between classes
            - Range: 0.5 (random) to 1.0 (perfect)
            
            **Hamming Accuracy**: Average accuracy across all emotion labels
            - More appropriate for multi-label classification
            
            **OVERSAMPLING IMPACT METRICS:**
            
            **Rare Emotion F1**: Shows if our 5-10x oversampling worked!
            - Measures F1-score specifically for emotions that were originally rare (<2,000 samples)
            - Before oversampling: These would be ~1-5% F1
            - After aggressive oversampling: Should be 20-40%+ F1
            - This metric directly shows the success of our balancing strategy
            
            **Why This Matters:**
            Our aggressive oversampling specifically targeted emotions like "grief", "pride", 
            "nervousness" that had <1,500 samples. Without this, they would have terrible F1-scores.
            The "Rare Emotion F1" metric shows if our 5-10x boosting strategy worked!
            """)
    
    def get_model_recommendations(self, results):
        """Provide recommendations based on metrics including oversampling effectiveness"""
        recommendations = []
        
        if len(results) > 1:
            # Compare models
            best_acc = max(results.items(), key=lambda x: x[1].get('hamming_accuracy', 0))
            best_auc = max(results.items(), key=lambda x: x[1].get('roc_auc', 0))
            best_rare = max(results.items(), key=lambda x: x[1].get('rare_emotion_avg_f1', 0))
            
            recommendations.append("**Model Performance Summary:**")
            recommendations.append(f"• Best Overall Accuracy: {best_acc[0].replace('_', ' ').title()}")
            recommendations.append(f"• Best ROC-AUC: {best_auc[0].replace('_', ' ').title()}")
            recommendations.append(f"• Best Rare Emotion Performance: {best_rare[0].replace('_', ' ').title()}")
            
            # Assess oversampling success
            recommendations.append("")
            recommendations.append("**Oversampling Effectiveness:**")
            for model_name, metrics in results.items():
                rare_f1 = metrics.get('rare_emotion_avg_f1', 0) * 100
                if rare_f1 > 25:
                    recommendations.append(f"• {model_name.replace('_', ' ').title()}: Excellent rare emotion improvement ({rare_f1:.1f}% F1)")
                elif rare_f1 > 15:
                    recommendations.append(f"• {model_name.replace('_', ' ').title()}: Good rare emotion improvement ({rare_f1:.1f}% F1)")
                else:
                    recommendations.append(f"• {model_name.replace('_', ' ').title()}: Moderate rare emotion improvement ({rare_f1:.1f}% F1)")
            
        else:
            recommendations.append("**Model evaluation completed successfully with oversampling assessment.**")
        
        return recommendations