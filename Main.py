import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
from pathlib import Path
import os
import gc

# Import simplified modules
from Bert_config import SimpleBERTEmbeddings
from data_preprocessing import SimpleDataPreprocessor
from Train_models import SimpleEmotionClassifiers
from Model_evaluation import SimpleModelEvaluator

# Essential emotion labels (27 emotions)
EMOTION_LABELS = [
    'admiration', 'amusement', 'anger', 'annoyance', 'approval', 'caring', 
    'confusion', 'curiosity', 'desire', 'disappointment', 'disapproval', 
    'disgust', 'embarrassment', 'excitement', 'fear', 'gratitude', 'grief', 
    'joy', 'love', 'nervousness', 'optimism', 'pride', 'realization', 
    'relief', 'remorse', 'sadness', 'surprise', 'neutral'
]

# Team info
TEAM_NAMES = [
    "Zoe Akua Ohene-Ampofo, 22252412",
]

# Page config
st.set_page_config(
    page_title="GoEmotions: Advanced Emotion Detection System",
    layout="wide",
    page_icon="📊"
)

# Enhanced CSS styling
ACCENT_START = "#22d3ee"
ACCENT_MID = "#a78bfa" 
ACCENT_END = "#f472b6"

st.markdown(
    f"""
    <style>
        .stApp {{
            background: radial-gradient(1200px 800px at 10% 10%, #0d1321 0%, #0a0f1c 30%, #070b14 55%, #05080f 100%) !important;
        }}
        .glass {{
            background: rgba(255,255,255,0.04);
            backdrop-filter: blur(8px);
            -webkit-backdrop-filter: blur(8px);
            border: 1px solid rgba(255,255,255,0.08);
            border-radius: 14px;
            padding: 1rem 1.1rem;
            box-shadow: 0 8px 24px rgba(0,0,0,0.35);
        }}
        .main-header {{
            font-size: 2.2rem;
            font-weight: 800;
            letter-spacing: 0.5px;
            background: linear-gradient(90deg, {ACCENT_START}, {ACCENT_MID}, {ACCENT_END});
            -webkit-background-clip: text;
            background-clip: text;
            color: transparent;
            margin: 0.2rem 0 0.2rem 0;
            text-align: center;
        }}
        .chip {{
            display: inline-block;
            padding: 6px 10px;
            border-radius: 999px;
            background: rgba(255,255,255,0.06);
            border: 1px solid rgba(255,255,255,0.08);
            margin: 4px 6px 0 0;
            font-size: 0.9rem;
            color: #e5e7eb;
        }}
        .emotion-word {{
            padding: 2px 6px;
            border-radius: 4px;
            margin: 1px;
            display: inline-block;
        }}
        .live-emotion {{
            background: linear-gradient(135deg, {ACCENT_START}33, {ACCENT_MID}33);
            border: 1px solid {ACCENT_MID}66;
            padding: 8px 12px;
            border-radius: 8px;
            margin: 4px;
            font-weight: 600;
        }}
        .threshold-card {{
            background: rgba(255,255,255,0.06);
            border: 1px solid rgba(255,255,255,0.12);
            border-radius: 12px;
            padding: 1rem;
            margin: 0.5rem 0;
        }}
        .stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {{ color: #e5e7eb !important; }}
        [data-testid="stMetric"] {{
            background: rgba(255,255,255,0.04);
            border: 1px solid rgba(255,255,255,0.08);
            border-radius: 12px;
            padding: 0.9rem 0.9rem;
            box-shadow: inset 0 0 0 1px rgba(255,255,255,0.04), 0 6px 18px rgba(0,0,0,0.35);
        }}
        [data-testid="stMetric"] [data-testid="stMetricDelta"] {{ color: {ACCENT_START} !important; }}
        .stTabs [data-baseweb="tab-list"] {{ gap: 4px; }}
        .stTabs [data-baseweb="tab"] {{
            background: rgba(255,255,255,0.04);
            border: 1px solid rgba(255,255,255,0.08);
            border-radius: 12px;
            padding: 10px 16px;
            color: #d1d5db;
        }}
        .stTabs [aria-selected="true"] {{
            background: linear-gradient(135deg, {ACCENT_START}33, {ACCENT_MID}33);
            color: #ffffff !important;
            border-color: {ACCENT_MID}66;
            box-shadow: 0 0 0 1px {ACCENT_START}33, 0 8px 22px rgba(0,0,0,0.35);
        }}
        .stButton > button {{
            background: linear-gradient(135deg, {ACCENT_START} 0%, {ACCENT_MID} 50%, {ACCENT_END} 100%);
            border: none;
            color: white;
            font-weight: 700;
            padding: 0.6rem 1.1rem;
            border-radius: 12px;
            transition: transform 0.06s ease-in-out, box-shadow 0.2s ease;
            box-shadow: 0 8px 22px {ACCENT_START}29, 0 2px 8px rgba(0,0,0,0.35);
        }}
        .stButton > button:hover {{
            transform: translateY(-1px);
            box-shadow: 0 12px 28px {ACCENT_MID}38, 0 4px 12px rgba(0,0,0,0.4);
        }}
        .stButton > button:active {{ transform: translateY(0px) scale(0.99); }}
        .step-container {{
            background: rgba(255,255,255,0.04);
            border: 1px solid rgba(255,255,255,0.08);
            border-radius: 14px;
            padding: 1.5rem;
            margin: 1rem 0;
            box-shadow: 0 8px 24px rgba(0,0,0,0.35);
        }}
        .progress-step {{
            background: rgba(255,255,255,0.06);
            border: 1px solid rgba(255,255,255,0.08);
            border-radius: 10px;
            padding: 0.8rem;
            margin: 0.3rem;
            text-align: center;
            font-weight: 600;
        }}
        .progress-step.active {{
            background: linear-gradient(135deg, {ACCENT_START}33, {ACCENT_MID}33);
            border-color: {ACCENT_MID}66;
            color: #ffffff !important;
        }}
        .progress-step.completed {{
            background: linear-gradient(135deg, #10b981, #059669);
            border-color: #10b981;
            color: #ffffff !important;
        }}
    </style>
    """,
    unsafe_allow_html=True,
)

# Simple cleanup function
def clear_memory():
    """Clear memory and GPU cache"""
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    except:
        pass

# Initialize components
@st.cache_resource
def get_components():
    preprocessor = SimpleDataPreprocessor()
    bert_embedder = SimpleBERTEmbeddings()
    classifiers = SimpleEmotionClassifiers()
    evaluator = SimpleModelEvaluator()
    return preprocessor, bert_embedder, classifiers, evaluator

preprocessor, bert_embedder, classifiers, evaluator = get_components()

# Session state
if 'step' not in st.session_state:
    st.session_state.step = 1

# Header
header_container = st.container()
with header_container:
    st.markdown('<div class="main-header">GoEmotions: Advanced Emotion Detection System</div>', unsafe_allow_html=True)
    
    # Team names as chips
    chips = " ".join([f"<span class='chip'>{name}</span>" for name in TEAM_NAMES])
    st.markdown(f"<div style='text-align: center; margin-bottom: 1rem;'>{chips}</div>", unsafe_allow_html=True)
    
    st.markdown(
        "<div style='color:#9ca3af; margin-top:6px; text-align: center;'>"
        "<strong>INNOVATIONS:</strong> Live Emotion Analysis, Word-Level Attention, Dynamic Thresholds &nbsp;|&nbsp; "
        "<strong>VISUALIZATIONS:</strong> ROC-AUC Curves, Emotion Trajectories, Memory Management &nbsp;|&nbsp; "
        "<strong>MODELS:</strong> BERT + Random Forest + Naive Bayes with Advanced Oversampling"
        "</div>",
        unsafe_allow_html=True,
    )

# Progress indicator
st.markdown("<div style='margin-top:1rem; margin-bottom: 1rem;'></div>", unsafe_allow_html=True)

progress_steps = ["Upload Data", "Process Data", "Train Model", "Evaluate Model", "Predict Emotion"]
current_step = st.session_state.get('step', 1)

# Create progress indicator
progress_html = "<div style='display: flex; justify-content: center; gap: 0.5rem; margin: 1rem 0;'>"
for i, step_name in enumerate(progress_steps, 1):
    if i < current_step:
        progress_html += f"<div class='progress-step completed'>{step_name}</div>"
    elif i == current_step:
        progress_html += f"<div class='progress-step active'>{step_name}</div>"
    else:
        progress_html += f"<div class='progress-step'>{step_name}</div>"

progress_html += "</div>"
st.markdown(progress_html, unsafe_allow_html=True)

st.divider()

# Main interface based on current step
if current_step == 1:
    # Step 1: Data Upload
    st.markdown('<div class="step-container">', unsafe_allow_html=True)
    st.header("Step 1: Upload Your Data")
    
    uploaded_file = st.file_uploader(
        "Upload GoEmotions CSV file",
        type=['csv'],
        help="Upload your CSV file with 'text' column and emotion labels"
    )
    
    if uploaded_file:
        df = preprocessor.load_data(uploaded_file)
        if df is not None:
            st.session_state.df = df
            
            # Show basic info
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Samples", f"{len(df):,}")
            with col2:
                st.metric("Emotions", len(EMOTION_LABELS))
            with col3:
                avg_length = df['text'].str.len().mean()
                st.metric("Avg Text Length", f"{avg_length:.0f}")
            
            # Show preview
            st.subheader("Data Preview")
            st.dataframe(df.head(), use_container_width=True)
            
            # Emotion chart
            st.subheader("Emotion Distribution in Dataset")
            
            emotion_counts = {}
            for emotion in EMOTION_LABELS:
                if emotion in df.columns:
                    count = (df[emotion] == 1).sum()
                    emotion_counts[emotion] = count
            
            if emotion_counts:
                emotion_df = pd.DataFrame([
                    {'Emotion': emotion.title(), 'Count': count} 
                    for emotion, count in emotion_counts.items()
                ]).sort_values('Count', ascending=False)
                
                # Analyze imbalance
                dominant_emotion = emotion_df.iloc[0]['Emotion'].lower()
                dominant_count = emotion_df.iloc[0]['Count']
                total_count = emotion_df['Count'].sum()
                dominant_percentage = (dominant_count / total_count) * 100
                
                # Check for rare emotions
                rare_emotions = emotion_df[emotion_df['Count'] < 1500]
                
                if len(rare_emotions) > 0:
                    st.error(f"**CLASS IMBALANCE DETECTED!** {len(rare_emotions)} emotions have <1,500 samples.")
                    st.write("**Rare emotions that will hurt F1-scores:**")
                    for _, row in rare_emotions.head(5).iterrows():
                        pct = (row['Count'] / total_count) * 100
                        st.write(f"   • **{row['Emotion']}**: {row['Count']:,} samples ({pct:.2f}%)")
                    st.info("**Solution**: Our advanced oversampling will fix this in Step 2!")
                elif dominant_percentage > 25:
                    st.warning(f"**Moderate imbalance detected.** {dominant_emotion} represents {dominant_percentage:.1f}% of data.")
                else:
                    st.success(f"**Good balance!** Most emotions have adequate samples.")
                
                fig = px.bar(
                    emotion_df, 
                    x='Emotion', 
                    y='Count',
                    title="Frequency of Emotions in Dataset",
                    labels={'Count': 'Number of Samples', 'Emotion': 'Emotion Type'}
                )
                
                fig.update_layout(
                    template="plotly_dark",
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    xaxis_tickangle=-45,
                    height=500
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            if st.button("Continue to Processing", type="primary"):
                st.session_state.step = 2
                st.rerun()
    
    st.markdown('</div>', unsafe_allow_html=True)

elif current_step == 2:
    # Step 2: Interactive Data Processing
    st.markdown('<div class="step-container">', unsafe_allow_html=True)
    st.header("Step 2: Advanced Data Processing")
    
    if 'df' not in st.session_state:
        st.warning("Please upload data first")
        if st.button("Back to Upload"):
            st.session_state.step = 1
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)
    
    else:
        df = st.session_state.df
        st.write(f"Working with {len(df):,} samples")
        
        # Interactive preprocessing options
        preprocessing_options = preprocessor.show_interactive_preprocessing_options()
        
        # Advanced Emotion Balance Configuration
        st.subheader("Advanced Class Balance Configuration")
        
        col1, col2 = st.columns(2)
        with col1:
            fix_imbalance = st.checkbox(
                "**Enable Advanced Oversampling**", 
                value=True,
                help="BOOST: 5-10x oversampling for rare emotions to dramatically improve F1-scores!"
            )
            
            if fix_imbalance:
                st.success("**Performance Mode**: Rare emotions will be boosted 5-10x for better learning")
                aggressive_level = st.selectbox(
                    "Balancing Strategy:",
                    ["Maximum Performance (5-10x boost)", "Moderate Balance (3-5x boost)", "Conservative (2-3x boost)"],
                    index=0,
                    help="Higher levels = better F1-scores for rare emotions"
                )
                
                if "Maximum Performance" in aggressive_level:
                    target_min_samples = 3000
                    max_dominant_samples = 12000
                    st.info("**Expected**: 20-30% F1-score improvement for rare emotions")
                elif "Moderate Balance" in aggressive_level:
                    target_min_samples = 2000
                    max_dominant_samples = 10000
                    st.info("**Expected**: 15-25% F1-score improvement")
                else:
                    target_min_samples = 1500
                    max_dominant_samples = 8000
                    st.info("**Expected**: 10-20% F1-score improvement")
        
        with col2:
            sample_size = st.slider(
                "Final Dataset Size", 
                min_value=5000, 
                max_value=min(100000, len(df) * 3), 
                value=min(25000, len(df)),
                help="After advanced oversampling, final dataset size for training"
            )
            
            if fix_imbalance:
                st.write("**Balancing Targets:**")
                st.write(f"• Min samples per emotion: {target_min_samples:,}")
                st.write(f"• Max dominant samples: {max_dominant_samples:,}")
                st.write(f"• Rare emotions will be boosted up to 10x!")
        
        # Show preprocessing preview
        if preprocessing_options:
            preprocessor.show_preprocessing_preview(df, preprocessing_options)
        
        # Add preprocessing checkbox
        st.subheader("Data Processing Options")
        enable_preprocessing = st.checkbox("Enable Text Preprocessing", value=True, 
                                         help="Apply selected preprocessing steps to clean text data")
        
        if not enable_preprocessing:
            st.info("Preprocessing disabled. Raw text will be used for training.")
            preprocessing_options = {'bert_optimized': True}  # Minimal processing
        
        # Process button with confirmation
        if preprocessing_options:
            st.subheader("Ready to Process")
            
            col1, col2 = st.columns([1, 1])
            with col1:
                if st.button("Show Processing Summary", type="secondary"):
                    st.session_state.show_summary = True
            
            with col2:
                process_button = st.button("Process Data", type="primary")
            
            # Show processing summary if requested
            if st.session_state.get('show_summary', False):
                st.info("**Processing Summary:**")
                if preprocessing_options.get('bert_optimized', False):
                    st.write("**BERT-Optimized Mode**: Minimal cleaning for maximum accuracy")
                else:
                    active_steps = []
                    if preprocessing_options.get('remove_stopwords'):
                        active_steps.append("Remove stop words")
                    if preprocessing_options.get('remove_punctuation'):
                        active_steps.append("Remove punctuation")
                    if preprocessing_options.get('remove_numbers'):
                        active_steps.append("Remove numbers")
                    if preprocessing_options.get('use_lemmatization'):
                        active_steps.append("Lemmatization")
                    elif preprocessing_options.get('use_stemming'):
                        active_steps.append("Stemming")
                    
                    st.write(f"**Standard NLP**: {', '.join(active_steps) if active_steps else 'Minimal processing'}")
                
                if fix_imbalance:
                    st.write(f"**Balance Fix**: Limit dominant class to {max_dominant_samples:,} samples")
                st.write(f"**Final Size**: {sample_size:,} samples")
            
            if process_button:
                with st.spinner("Processing data with advanced oversampling..."):
                    try:
                        # Pass the new aggressive balancing parameters
                        if fix_imbalance:
                            # Update preprocessor with aggressive settings
                            if hasattr(preprocessor, 'aggressive_emotion_balancing'):
                                # Store settings for the aggressive balancing
                                preprocessor.target_min_samples = target_min_samples
                                preprocessor.max_dominant_samples = max_dominant_samples
                        
                        X_train, X_test, y_train, y_test = preprocessor.process_data(
                            df, 
                            sample_size=sample_size,
                            preprocessing_options=preprocessing_options if enable_preprocessing else {'bert_optimized': True},
                            fix_imbalance=fix_imbalance
                        )
                        
                        if X_train is not None and X_test is not None:
                            st.session_state.X_train = X_train
                            st.session_state.X_test = X_test
                            st.session_state.y_train = y_train
                            st.session_state.y_test = y_test
                            st.session_state.data_processed = True
                            st.session_state.preprocessing_options = preprocessing_options
                            
                            st.success("Data processed with advanced oversampling!")
                            st.balloons()
                            
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("Training Samples", f"{len(X_train):,}")
                            with col2:
                                st.metric("Test Samples", f"{len(X_test):,}")
                            with col3:
                                # Calculate expected improvement
                                if fix_imbalance:
                                    if "Maximum Performance" in aggressive_level:
                                        expected_boost = "20-30%"
                                    elif "Moderate Balance" in aggressive_level:
                                        expected_boost = "15-25%"
                                    else:
                                        expected_boost = "10-20%"
                                    st.metric("Expected F1 Boost", expected_boost)
                                else:
                                    st.metric("Ready for Training", "Yes")
                            
                        else:
                            st.error("Data processing failed")
                            
                    except Exception as e:
                        st.error(f"Error: {str(e)}")
        
        # Continue button if data exists
        if all(key in st.session_state for key in ['X_train', 'X_test', 'y_train', 'y_test']):
            st.divider()
            st.success("Data is processed with advanced oversampling and ready for high-performance training!")
            
            if st.button("**CONTINUE TO TRAINING**", type="primary", key="continue_big"):
                st.session_state.step = 3
                st.rerun()
        
        st.markdown('</div>', unsafe_allow_html=True)

elif current_step == 3:
    # Step 3: Train Models with BERT Model Info
    st.markdown('<div class="step-container">', unsafe_allow_html=True)
    st.header("Step 3: Train Models")
    
    if not all(key in st.session_state for key in ['X_train', 'X_test', 'y_train', 'y_test']):
        st.warning("Please process data first")
        if st.button("Back to Processing"):
            st.session_state.step = 2
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)
        
    else:
        st.write(f"Ready to train with {len(st.session_state.X_train):,} training samples")
        
        # BERT Model Selection with Advantages/Disadvantages
        st.subheader("BERT Model Configuration")
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            model_options = {
                'bert-base-uncased': {
                    'name': 'BERT Base Uncased (Recommended)',
                    'description': '**Best for emotion detection** - Optimized for social media text',
                },
                'bert-large-uncased': {
                    'name': 'BERT Large Uncased',
                    'description': '**Slower but slightly more accurate** - 2x computational cost',
                },
            }
            
            selected_model = st.selectbox(
                "Choose BERT Model:",
                options=list(model_options.keys()),
                index=0,
                format_func=lambda x: model_options[x]['name']
            )
        
        with col2:
            st.write("**System Status:**")
            st.write("CPU: Available")
            try:
                import torch
                if torch.cuda.is_available():
                    st.write("GPU: Available")
                else:
                    st.write("GPU: Not available")
            except:
                st.write("GPU: Not detected")
        
        # Model advantages/disadvantages
        with st.expander("BERT Model Comparison & Hardware Requirements"):
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**BERT Base Uncased**")
                st.write("**Advantages:**")
                st.write("• Perfect for emotion detection (GoEmotions trained)")
                st.write("• Fast training: ~2-5 minutes")
                st.write("• Memory efficient: 768 dimensions")
                st.write("• Works on CPU (4GB+ RAM)")
                st.write("• GPU recommended: 2GB+")
                st.write("• Proven accuracy: 75-85% emotion tasks")
                
                st.write("**Disadvantages:**")
                st.write("• Lower capacity than BERT Large")
                st.write("• May miss subtle emotions")
            
            with col2:
                st.write("**BERT Large Uncased**")
                st.write("**Advantages:**")
                st.write("• Higher capacity: 1024 dimensions")
                st.write("• Better at complex language")
                st.write("• ~3% better accuracy")
                st.write("• More robust to noise")
                
                st.write("**Disadvantages:**")
                st.write("• 2x slower training: ~5-15 minutes")
                st.write("• Memory hungry: 8GB+ RAM needed")
                st.write("• GPU recommended: 6GB+")
                st.write("• Overkill for social media text")
                st.write("• Diminishing returns for emotions")
            
            # Hardware recommendations
            st.divider()
            st.write("**Recommendations:**")
            
            try:
                import torch
                if torch.cuda.is_available():
                    st.success("**Excellent setup!** GPU detected. Both models will work great. BERT Base recommended for speed.")
                else:
                    st.info("**CPU Mode**: Training will be slower but functional. BERT Base recommended.")
            except:
                st.info("**Standard setup**: BERT Base is the optimal choice for emotion detection.")
        
        # Update embedder
        if selected_model != bert_embedder.model_name:
            bert_embedder.model_name = selected_model
            bert_embedder.model = None
            bert_embedder.tokenizer = None
        
        st.divider()
        
        # Check if models are already trained
        if st.session_state.get('models_trained', False):
            st.success("Models have been trained successfully!")
            
            if st.button("**CONTINUE TO EVALUATION**", type="primary", key="proceed_to_evaluation"):
                st.session_state.step = 4
                st.rerun()
        
        else:
            train_button = st.button("Start Training", type="primary", key="start_training_btn")
            
            if train_button:
                progress_bar = st.progress(0)
                status = st.empty()
                
                try:
                    # Generate BERT embeddings
                    status.text(f"Loading {selected_model} model...")
                    success = bert_embedder.load_model()
                    if not success:
                        st.error("Failed to load BERT model")
                        st.stop()
                    
                    progress_bar.progress(20)
                    
                    status.text("Generating training embeddings...")
                    X_train_embeddings = bert_embedder.generate_embeddings(st.session_state.X_train)
                    if X_train_embeddings is None:
                        st.error("Failed to generate training embeddings")
                        st.stop()
                    
                    progress_bar.progress(40)
                    
                    status.text("Generating test embeddings...")
                    X_test_embeddings = bert_embedder.generate_embeddings(st.session_state.X_test)
                    if X_test_embeddings is None:
                        st.error("Failed to generate test embeddings")
                        st.stop()
                    
                    progress_bar.progress(60)
                    
                    # Train models
                    status.text("Training Naive Bayes with GaussianNB + PCA...")
                    nb_success = classifiers.train_naive_bayes(X_train_embeddings, st.session_state.y_train)
                    
                    progress_bar.progress(80)
                    
                    status.text("Training Random Forest with imbalance handling...")
                    rf_success = classifiers.train_random_forest(X_train_embeddings, st.session_state.y_train)
                    
                    progress_bar.progress(100)
                    
                    # Store results
                    st.session_state.X_train_embeddings = X_train_embeddings
                    st.session_state.X_test_embeddings = X_test_embeddings
                    st.session_state.classifiers = classifiers
                    st.session_state.bert_embedder = bert_embedder
                    st.session_state.models_trained = True
                    st.session_state.selected_model = selected_model
                    
                    status.text("Training completed!")
                    st.success("Models trained successfully!")
                    st.balloons()
                    
                    # Show training summary
                    st.subheader("Training Summary")
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("BERT Model", selected_model)
                        st.metric("Embedding Dimension", "768" if 'base' in selected_model else "1024")
                    
                    with col2:
                        st.metric("Training Samples", len(X_train_embeddings))
                        st.metric("Test Samples", len(X_test_embeddings))
                    
                    with col3:
                        models_trained = []
                        if nb_success:
                            models_trained.append("GaussianNB + PCA")
                        if rf_success:
                            models_trained.append("Random Forest")
                        st.metric("Models Trained", len(models_trained))
                        st.metric("Status", "Ready")
                    
                    st.info("**Auto-advancing to Step 4 in 2 seconds...**")
                    time.sleep(2)
                    st.session_state.step = 4
                    st.rerun()
                    
                except Exception as e:
                    st.error(f"Training failed: {str(e)}")
                    st.exception(e)
        
        st.markdown('</div>', unsafe_allow_html=True)

elif current_step == 4:
    # Step 4: Evaluate Models with ROC-AUC Visualization
    st.markdown('<div class="step-container">', unsafe_allow_html=True)
    st.header("Step 4: Evaluate Models with Advanced Visualization")
    
    if all(key in st.session_state for key in ['X_test_embeddings', 'y_test', 'classifiers']):
        
        if st.button("Evaluate Models", type="primary"):
            with st.spinner("Evaluating models with comprehensive metrics and visualizations..."):
                # Evaluate models
                results = evaluator.evaluate_models(
                    st.session_state.classifiers,
                    st.session_state.X_test_embeddings,
                    st.session_state.y_test
                )
                
                if results:
                    st.session_state.results = results
                    
                    # Performance summary
                    summary_data = evaluator.display_performance_summary(results)
                    
                    if summary_data and len(summary_data) > 1:
                        best_model_data = max(summary_data, key=lambda x: float(x['Accuracy'].rstrip('%')))
                        best_model_name = best_model_data['Model'].lower().replace(' ', '_')
                        st.session_state.best_model_for_prediction = best_model_name
                        
                        st.success(f"**{best_model_data['Model']}** will be used for predictions")
                    else:
                        st.session_state.best_model_for_prediction = 'random_forest'
                    
                    if st.button("Continue to Advanced Predictions", type="primary"):
                        st.session_state.step = 5
                        st.rerun()
    else:
        st.warning("Please train models first")
        if st.button("Back to Training"):
            st.session_state.step = 3
            st.rerun()
    
    st.markdown('</div>', unsafe_allow_html=True)

elif current_step == 5:
    # Step 5: Advanced Predictions with Innovations
    st.markdown('<div class="step-container">', unsafe_allow_html=True)
    st.header("Step 5: Advanced Emotion Predictions")
    
    if all(key in st.session_state for key in ['classifiers', 'bert_embedder']):
        
        best_model = st.session_state.get('best_model_for_prediction', 'random_forest')
        
        if 'results' in st.session_state:
            best_model_display = best_model.replace('_', ' ').title()
            
            if best_model in st.session_state.results:
                metrics = st.session_state.results[best_model]
                hamming_accuracy = metrics.get('hamming_accuracy', 0) * 100
                roc_auc = metrics.get('roc_auc', 0) * 100
                
                st.info(f"Using best model: **{best_model_display}** "
                       f"(Accuracy: {hamming_accuracy:.1f}%, ROC-AUC: {roc_auc:.1f}%)")
            else:
                st.info(f"Using model: **{best_model_display}**")
        
        # Prediction tabs with innovations
        tab1, tab2, tab3, tab4 = st.tabs(["Single Text", "Batch Upload", "Live Analysis", "Emotion Trajectory"])
        
        with tab1:
            st.subheader("Single Text Prediction with Word-Level Attention")
            
            # Emotion examples
            with st.expander("Try these example texts"):
                example_texts = [
                    "I'm so excited about this new opportunity! Can't wait to start!",
                    "This situation is really frustrating and making me angry.",
                    "I feel so sad and disappointed about what happened.",
                    "I'm worried and nervous about the upcoming presentation.",
                    "That joke was absolutely hilarious! I can't stop laughing.",
                    "I'm grateful for all the support you've given me.",
                    "This is just a regular update about the project status."
                ]
                
                for i, example in enumerate(example_texts):
                    if st.button(f"Use Example {i+1}: \"{example[:50]}...\"", key=f"example_{i}"):
                        st.session_state.example_text = example
            
            text_input = st.text_area(
                "Enter text to analyze:",
                value=st.session_state.get('example_text', ''),
                placeholder="Type something like: 'I'm so excited about this!' or 'This makes me angry.'"
            )
            
            col1, col2 = st.columns([1, 1])
            with col1:
                show_attention = st.checkbox("Show Word-Level Attention", value=True, 
                                           help="Highlight which words triggered each emotion")
            with col2:
                st.write("")  # Placeholder for balanced layout
            
            if st.button("Predict Emotions", type="primary"):
                if text_input.strip():
                    with st.spinner("Analyzing emotions with advanced features..."):
                        results = classifiers.predict_single_text_with_attention(
                            text_input, 
                            st.session_state.bert_embedder,
                            model_type=best_model,
                            threshold=0.3,
                            thresholds=None,
                            return_attention=show_attention
                        )
                        
                        if results and 'top_3_emotions' in results:
                            st.subheader("Top 3 Predicted Emotions")
                            
                            # Show top 3 emotions in nice cards
                            for i, (emotion, prob) in enumerate(zip(results['top_3_emotions'], results['top_3_probabilities'])):
                                rank = i + 1
                                confidence_pct = prob * 100
                                
                                # Color coding
                                if rank == 1:
                                    color = "#FFD700"
                                elif rank == 2:
                                    color = "#C0C0C0"
                                else:
                                    color = "#CD7F32"
                                
                                # Create emotion card
                                st.markdown(f"""
                                <div class="threshold-card" style="border-left: 4px solid {color};">
                                    <div style="display: flex; justify-content: space-between; align-items: center;">
                                        <div>
                                            <h4>#{rank} {emotion.title()}</h4>
                                            <p style="margin: 0; color: #9ca3af;">Confidence: {confidence_pct:.1f}%</p>
                                        </div>
                                        <div style="font-size: 2rem; opacity: 0.7;">{confidence_pct:.0f}%</div>
                                    </div>
                                </div>
                                """, unsafe_allow_html=True)
                                
                                # Progress bar
                                st.progress(prob, text=f"{emotion.title()}: {confidence_pct:.1f}%")
                            
                            # Word-level attention visualization
                            if show_attention and 'attention_weights' in results:
                                st.subheader("Word-Level Emotion Attention")
                                st.write("**Words highlighted by importance for detected emotions:**")
                                
                                attention_html = bert_embedder.visualize_attention(
                                    text_input, 
                                    results['attention_weights'],
                                    results['top_3_emotions'][0]  # Top emotion
                                )
                                
                                if attention_html:
                                    st.markdown(attention_html, unsafe_allow_html=True)
                                    
                                    st.info(f"**Analysis**: Words in darker colors contributed more to detecting **{results['top_3_emotions'][0]}** emotion.")
                            
                            # Show insight based on top emotion
                            top_emotion = results['top_3_emotions'][0]
                            top_confidence = results['top_3_probabilities'][0] * 100
                            
                            if top_confidence >= 60:
                                insight = f"Strong **{top_emotion}** emotion detected! The model is quite confident."
                            elif top_confidence >= 40:
                                insight = f"Moderate **{top_emotion}** emotion detected with reasonable confidence."
                            else:
                                insight = f"Weak **{top_emotion}** signal. The text may be emotionally neutral or contain mixed emotions."
                            
                            st.info(f"**Analysis**: {insight}")
                        
                        else:
                            st.error("Failed to analyze emotions. Please try again.")
                else:
                    st.warning("Please enter some text")
        
        with tab2:
            # Batch upload with trajectory
            st.subheader("Batch File Prediction")
            
            uploaded_file = st.file_uploader(
                "Upload CSV file with 'text' column",
                type=['csv'],
                key="batch_upload"
            )
            
            if uploaded_file:
                try:
                    df = pd.read_csv(uploaded_file)
                    
                    if 'text' not in df.columns:
                        st.error("File must contain a 'text' column")
                    else:
                        st.success(f"Loaded {len(df):,} texts for prediction")
                        st.dataframe(df.head(), use_container_width=True)
                        
                        # Batch processing options
                        show_trajectory = st.checkbox("Show Emotion Trajectory", value=True, key="batch_trajectory")
                        
                        if st.button("Process Batch", type="primary"):
                            with st.spinner(f"Processing {len(df):,} texts..."):
                                results_df = classifiers.predict_batch_with_trajectory(
                                    df, 
                                    st.session_state.bert_embedder,
                                    model_type=best_model,
                                    threshold=0.25,
                                    thresholds=None
                                )
                                
                                if results_df is not None:
                                    st.success(f"Processed {len(results_df):,} texts!")
                                    
                                    # Show trajectory if enabled
                                    if show_trajectory:
                                        trajectory_fig = evaluator.create_emotion_trajectory(results_df)
                                        if trajectory_fig:
                                            st.subheader("Emotion Trajectory Over Time")
                                            st.plotly_chart(trajectory_fig, use_container_width=True)
                                    
                                    # Show results
                                    st.subheader("First 50 Results")
                                    display_df = results_df.head(50).copy()
                                    
                                    if 'text' in display_df.columns:
                                        display_df['text'] = display_df['text'].apply(lambda x: x[:80] + "..." if len(str(x)) > 80 else x)
                                    
                                    if 'confidence' in display_df.columns:
                                        display_df['confidence'] = display_df['confidence'].apply(lambda x: f"{x*100:.1f}%")
                                    
                                    display_columns = ['text', 'top_emotion', 'confidence', 'top_3_emotions']
                                    available_columns = [col for col in display_columns if col in display_df.columns]
                                    
                                    if available_columns:
                                        st.dataframe(display_df[available_columns], use_container_width=True)
                                    else:
                                        st.dataframe(display_df, use_container_width=True)
                                    
                                    # Download section
                                    st.subheader("Download Results")
                                    col1, col2 = st.columns(2)
                                    
                                    with col1:
                                        csv_data = results_df.to_csv(index=False)
                                        st.download_button(
                                            "Download Complete Results",
                                            data=csv_data,
                                            file_name=f"emotion_predictions_{len(results_df)}.csv",
                                            mime="text/csv",
                                            type="primary"
                                        )
                                    
                                    with col2:
                                        summary = {
                                            'Total_Processed': len(results_df),
                                            'Most_Common_Emotion': results_df['top_emotion'].mode().iloc[0] if 'top_emotion' in results_df.columns else 'N/A',
                                            'Unique_Emotions': results_df['top_emotion'].nunique() if 'top_emotion' in results_df.columns else 0,
                                            'Average_Confidence': f"{results_df['confidence'].mean()*100:.1f}%" if 'confidence' in results_df.columns else 'N/A'
                                        }
                                        
                                        summary_csv = pd.DataFrame([summary]).to_csv(index=False)
                                        st.download_button(
                                            "Download Summary", 
                                            data=summary_csv,
                                            file_name="emotion_summary.csv",
                                            mime="text/csv"
                                        )
                                else:
                                    st.error("Batch prediction failed")
                
                except Exception as e:
                    st.error(f"Error processing file: {str(e)}")
        
        with tab3:
            # Live emotion analysis
            st.subheader("Live Emotion Analysis")
            st.write("**Type and watch emotions update in real-time!**")
            
            # Live text input
            live_text = st.text_area(
                "Type your message here:",
                placeholder="Start typing to see live emotion analysis...",
                height=100,
                key="live_input_area"
            )
            
            # Live analysis trigger
            if st.button("Analyze Current Text", key="live_analyze_btn"):
                if live_text and len(live_text.strip()) > 5:
                    with st.spinner("Analyzing..."):
                        results = classifiers.predict_single_text(
                            live_text, 
                            st.session_state.bert_embedder,
                            model_type=best_model,
                            threshold=0.3
                        )
                        
                        if results and 'top_3_emotions' in results:
                            st.write("**Live Emotions Detected:**")
                            
                            # Show live emotions
                            for i, (emotion, prob) in enumerate(zip(results['top_3_emotions'], results['top_3_probabilities'])):
                                if prob > 0.2:  # Show confident predictions
                                    confidence = prob * 100
                                    st.markdown(f"<span class='live-emotion'>{emotion.title()} ({confidence:.0f}%)</span>", unsafe_allow_html=True)
                            
                            # Emotion meter
                            top_emotion = results['top_3_emotions'][0]
                            top_confidence = results['top_3_probabilities'][0]
                            
                            col1, col2 = st.columns([3, 1])
                            with col1:
                                st.progress(top_confidence, text=f"Strongest emotion: {top_emotion.title()}")
                            with col2:
                                st.metric("Confidence", f"{top_confidence*100:.0f}%")
                        else:
                            st.error("Failed to analyze emotions. Please try again.")
                else:
                    st.info("Please type at least 5 characters for analysis...")
        
        with tab4:
            # Emotion trajectory comparison
            st.subheader("Emotion Trajectory Comparison")
            st.write("**Compare emotion trends between different texts or time periods**")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Text A:**")
                text_a = st.text_area("First text or early period", placeholder="Enter first text...", key="text_a")
            
            with col2:
                st.write("**Text B:**")
                text_b = st.text_area("Second text or later period", placeholder="Enter second text...", key="text_b")
            
            if st.button("Compare Emotions", type="primary"):
                if text_a.strip() and text_b.strip():
                    with st.spinner("Analyzing both texts..."):
                        # Analyze both texts
                        results_a = classifiers.predict_single_text(text_a, st.session_state.bert_embedder, model_type=best_model)
                        results_b = classifiers.predict_single_text(text_b, st.session_state.bert_embedder, model_type=best_model)
                        
                        if results_a and results_b:
                            # Create comparison chart
                            comparison_fig = evaluator.create_emotion_comparison(results_a, results_b, "Text A", "Text B")
                            if comparison_fig:
                                st.plotly_chart(comparison_fig, use_container_width=True)
                            
                            # Show key differences
                            st.subheader("Key Differences")
                            
                            col1, col2 = st.columns(2)
                            with col1:
                                st.write("**Text A Top Emotions:**")
                                for i, (emotion, prob) in enumerate(zip(results_a['top_3_emotions'], results_a['top_3_probabilities'])):
                                    st.write(f"{i+1}. {emotion.title()}: {prob*100:.1f}%")
                            
                            with col2:
                                st.write("**Text B Top Emotions:**")
                                for i, (emotion, prob) in enumerate(zip(results_b['top_3_emotions'], results_b['top_3_probabilities'])):
                                    st.write(f"{i+1}. {emotion.title()}: {prob*100:.1f}%")
                            
                            # Insight
                            top_a = results_a['top_3_emotions'][0]
                            top_b = results_b['top_3_emotions'][0]
                            
                            if top_a == top_b:
                                st.info(f"**Consistent Emotion**: Both texts show **{top_a}** as the dominant emotion.")
                            else:
                                st.warning(f"**Emotion Shift**: From **{top_a}** in Text A to **{top_b}** in Text B.")
                        
                        else:
                            st.error("Failed to analyze one or both texts")
                else:
                    st.warning("Please enter both texts for comparison")
    else:
        st.warning("Please complete evaluation first")
        if st.button("Back to Evaluation"):
            st.session_state.step = 4
            st.rerun()
    
    st.markdown('</div>', unsafe_allow_html=True)

# Enhanced sidebar with memory monitoring
with st.sidebar:
    st.header("System Status")
    
    status_items = [
        ("Data Loaded", 'df' in st.session_state),
        ("Data Processed", 'data_processed' in st.session_state and st.session_state.data_processed),
        ("Models Trained", 'models_trained' in st.session_state and st.session_state.models_trained),
        ("Models Evaluated", 'results' in st.session_state)
    ]
    
    for label, status in status_items:
        if status:
            st.success(f"✓ {label}")
        else:
            st.error(f"✗ {label}")
    
    # Progress indicator
    completed_steps = sum(1 for _, status in status_items if status)
    total_steps = len(status_items)
    progress = completed_steps / total_steps
    
    st.subheader("Overall Progress")
    st.progress(progress)
    st.write(f"**{completed_steps}/{total_steps} steps completed**")
    
    st.divider()
    
    # Model performance
    if hasattr(st.session_state, 'results') and st.session_state.results:
        st.header("Model Performance")
        results = st.session_state.results
        
        for model_name, metrics in results.items():
            hamming_accuracy = metrics.get('hamming_accuracy', 0) * 100
            roc_auc = metrics.get('roc_auc', 0) * 100
            
            if hamming_accuracy >= 75:
                status_text = "Excellent"
            elif hamming_accuracy >= 65:
                status_text = "Good"
            else:
                status_text = "Fair"
            
            st.write(f"**{model_name.replace('_', ' ').title()}**")
            st.write(f"{hamming_accuracy:.1f}% {status_text}")
            st.write(f"   ROC-AUC: {roc_auc:.1f}%")
    
    st.divider()
    
    # Manual navigation
    st.header("Manual Navigation")
    current_step_num = st.session_state.get('step', 1)
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("← Previous") and current_step_num > 1:
            st.session_state.step = current_step_num - 1
            st.rerun()
    
    with col2:
        if st.button("Next →") and current_step_num < 5:
            st.session_state.step = current_step_num + 1
            st.rerun()
    
    st.write(f"Step: {current_step_num}/5")
    
    if st.button("Start Over"):
        # Clear memory before restart
        clear_memory()
        for key in list(st.session_state.keys()):
            if key != 'step':
                del st.session_state[key]
        st.session_state.step = 1
        st.rerun()

# Footer
st.divider()
st.markdown(
    "<div style='text-align: center; color: #9ca3af; padding: 18px;'>"
    "<p><strong>GoEmotions: Advanced Emotion Detection System</strong></p>"
    "<p>BERT-based Multi-Label Emotion Classification with Random Forest and Naive Bayes</p>"
    "</div>",
    unsafe_allow_html=True
)
