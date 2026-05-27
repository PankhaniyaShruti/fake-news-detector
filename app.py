"""
Fake News Detection Web Application
Main Streamlit application with modern dashboard UI
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import time
import os

# Import custom modules
from utils import (
    clean_text, extract_features, analyze_news_article,
    load_model_and_vectorizer, get_model_info, generate_sample_text
)
from model import predict_news
from database import DatabaseManager


# ==================== PAGE CONFIGURATION ====================
st.set_page_config(
    page_title="Fake News Detector",
    page_icon="🕵️",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ==================== CUSTOM CSS ====================
def load_custom_css():
    """Load custom CSS for modern UI"""
    st.markdown("""
    <style>
    .main {
        padding: 20px;
    }
    
    .app-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 30px;
        border-radius: 15px;
        color: white;
        margin-bottom: 30px;
        text-align: center;
    }
    
    .app-header h1 {
        font-size: 2.5rem;
        margin-bottom: 10px;
    }
    
    .app-header p {
        font-size: 1.2rem;
        opacity: 0.9;
    }
    
    .card {
        background: white;
        padding: 25px;
        border-radius: 15px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        margin-bottom: 20px;
    }
    
    .prediction-real {
        background: linear-gradient(135deg, #11998e, #38ef7d);
        padding: 30px;
        border-radius: 15px;
        color: white;
        text-align: center;
        font-size: 28px;
        font-weight: bold;
        margin: 20px 0;
        animation: fadeIn 0.5s ease;
    }
    
    .prediction-fake {
        background: linear-gradient(135deg, #eb3349, #f45c43);
        padding: 30px;
        border-radius: 15px;
        color: white;
        text-align: center;
        font-size: 28px;
        font-weight: bold;
        margin: 20px 0;
        animation: fadeIn 0.5s ease;
    }
    
    .stButton > button {
        border-radius: 10px;
        font-weight: 600;
        transition: all 0.3s ease;
        padding: 10px 25px;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(0,0,0,0.2);
    }
    
    .stTextArea > div > div > textarea {
        border-radius: 10px;
        border: 2px solid #e0e0e0;
        font-size: 16px;
        padding: 15px;
    }
    
    .stTextArea > div > div > textarea:focus {
        border-color: #667eea;
        box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.2);
    }
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
        background-color: #f8f9fa;
        border-radius: 12px;
        padding: 5px;
    }
    
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        padding: 10px 20px;
        font-weight: 600;
    }
    
    [data-testid="stMetricValue"] {
        font-size: 2rem;
        font-weight: bold;
    }
    
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #f8f9fa 0%, #e9ecef 100%);
    }
    
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    @keyframes slideIn {
        from { opacity: 0; transform: translateX(-20px); }
        to { opacity: 1; transform: translateX(0); }
    }
    
    .animate-fade-in {
        animation: fadeIn 0.5s ease;
    }
    
    .animate-slide-in {
        animation: slideIn 0.5s ease;
    }
    
    .stProgress > div > div > div > div {
        border-radius: 10px;
    }
    
    [data-testid="stDataFrame"] {
        border-radius: 10px;
        overflow: hidden;
    }
    
    .streamlit-expanderHeader {
        font-weight: 600;
        font-size: 1.1rem;
    }
    </style>
    """, unsafe_allow_html=True)


# ==================== INITIALIZATION ====================
@st.cache_resource
def init_resources():
    """Initialize model, vectorizer, and database"""
    model, vectorizer = load_model_and_vectorizer()
    db = DatabaseManager()
    return model, vectorizer, db


def load_model():
    """Load or train model if not exists"""
    model, vectorizer, _ = init_resources()
    
    if model is None or vectorizer is None:
        st.warning("Model not found! Training new model automatically...")
        
        try:
            from preprocessor import load_and_prepare_data
            from model import train_model
            from utils import save_model_and_vectorizer
            
            with st.spinner("Training model... This may take a minute..."):
                X_train, X_test, y_train, y_test = load_and_prepare_data()
                model, vectorizer, metrics = train_model(
                    X_train, X_test, y_train, y_test
                )
                save_model_and_vectorizer(model, vectorizer)
            
            st.success(f"Model trained! Accuracy: {metrics['test_accuracy']:.1%}")
            st.rerun()
            
        except Exception as e:
            st.error(f"Model training failed: {str(e)}")
            return None, None
    
    return model, vectorizer


# ==================== SIDEBAR ====================
def render_sidebar():
    """Render sidebar with information and statistics"""
    with st.sidebar:
        st.markdown("""
        <div style="text-align: center; padding: 20px 0;">
            <h2 style="color: #667eea;">Fake News Detector</h2>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        with st.expander("Model Information", expanded=True):
            model_info = get_model_info()
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Status", model_info['status'])
                st.metric("Type", model_info['model_type'])
            with col2:
                st.metric("Features", model_info['features'])
                st.metric("Vectorizer", "TF-IDF")
        
        st.markdown("---")
        
        with st.expander("Prediction Statistics", expanded=True):
            try:
                _, _, db = init_resources()
                stats = db.get_statistics()
                
                if stats['total'] > 0:
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Total", stats['total'])
                        st.metric("Real News", stats['real_count'])
                    with col2:
                        st.metric("Today", stats['today_count'])
                        st.metric("Fake News", stats['fake_count'])
                    
                    st.metric("Avg Confidence", f"{stats['avg_confidence']:.1f}%")
                else:
                    st.info("No predictions yet. Start analyzing!")
                    
            except Exception as e:
                st.warning("Statistics not available")
        
        st.markdown("---")
        
        with st.expander("How It Works", expanded=False):
            st.markdown("""
            **Detection Process:**
            
            1. **Text Input** - Paste your news article
            2. **Preprocessing** - Clean and normalize text
            3. **Feature Extraction** - TF-IDF vectorization
            4. **Prediction** - ML model analysis
            5. **Result** - Real or Fake verdict
            """)
        
        st.markdown("---")
        
        with st.expander("Tips for Better Results", expanded=False):
            st.info("""
            - Use at least 50+ words for accuracy
            - Paste the full article when possible
            - Watch for sensational language
            - Check for multiple sources
            - Be cautious of all-caps words
            - Look for excessive punctuation
            """)
        
        st.markdown("---")
        
        st.markdown("""
        <div style="text-align: center; color: #666; font-size: 0.8rem; padding: 10px;">
            <p>Made with ❤ for Education</p>
            <p>v1.0.0 | MIT License</p>
        </div>
        """, unsafe_allow_html=True)


# ==================== DETECTION TAB ====================
def render_detection_tab(model, vectorizer, db):
    """Render the main detection interface"""
    
    st.markdown("### Enter News Article Text")
    st.markdown("*Paste or type the news article you want to analyze*")
    
    text_input = st.text_area(
        "News Article Text",
        height=200,
        placeholder="Paste your news article here...",
        label_visibility="collapsed",
        key="news_text_input"
    )
    
    if text_input:
        word_count = len(text_input.split())
        st.caption(f"Word count: {word_count} | Characters: {len(text_input)}")
    
    col1, col2, col3, col4 = st.columns([1, 1, 1, 1])
    
    with col1:
        analyze_btn = st.button(
            "Analyze Text",
            type="primary",
            use_container_width=True,
            disabled=not text_input
        )
    
    with col2:
        if st.button("Load Sample", use_container_width=True):
            samples = generate_sample_text()
            import random
            sample_text = random.choice([samples['real_news'], samples['fake_news']])
            st.session_state.news_text_input = sample_text
            st.rerun()
    
    with col3:
        if st.button("Load Real News", use_container_width=True):
            samples = generate_sample_text()
            st.session_state.news_text_input = samples['real_news']
            st.rerun()
    
    with col4:
        if st.button("Load Fake News", use_container_width=True):
            samples = generate_sample_text()
            st.session_state.news_text_input = samples['fake_news']
            st.rerun()
    
    if analyze_btn and text_input:
        if len(text_input.split()) < 10:
            st.warning("Text is too short! Please enter at least 10 words for better accuracy.")
            return
        
        with st.spinner("Analyzing text with Machine Learning..."):
            progress_bar = st.progress(0)
            for i in range(100):
                time.sleep(0.01)
                progress_bar.progress(i + 1)
            progress_bar.empty()
            
            prediction, confidence = predict_news(text_input, model, vectorizer)
            analysis = analyze_news_article(text_input)
            
            try:
                db.save_prediction(
                    text=text_input,
                    prediction=prediction,
                    confidence=confidence,
                    word_count=analysis['word_count']
                )
            except Exception as e:
                st.warning(f"Could not save to history: {e}")
            
            st.session_state.last_prediction = {
                'prediction': prediction,
                'confidence': confidence,
                'analysis': analysis,
                'text': text_input,
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
    
    if 'last_prediction' in st.session_state:
        display_prediction_results(st.session_state.last_prediction)


def display_prediction_results(result):
    """Display the prediction results"""
    
    st.markdown("---")
    st.markdown("## Analysis Results")
    
    if result['prediction'] == "Real News":
        st.markdown(f"""
        <div class="prediction-real">
            <h2 style="margin:0;">{result['prediction']}</h2>
            <p style="font-size: 20px; margin-top: 10px;">
                Confidence: {result['confidence']:.1f}%
            </p>
            <p style="font-size: 14px; opacity: 0.9;">
                Analyzed at: {result['timestamp']}
            </p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="prediction-fake">
            <h2 style="margin:0;">{result['prediction']}</h2>
            <p style="font-size: 20px; margin-top: 10px;">
                Confidence: {result['confidence']:.1f}%
            </p>
            <p style="font-size: 14px; opacity: 0.9;">
                Analyzed at: {result['timestamp']}
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("### Confidence Meter")
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.progress(result['confidence'] / 100)
        confidence_color = "green" if result['prediction'] == "Real News" else "red"
        st.markdown(f"""
        <div style="text-align: center; font-size: 1.2rem; font-weight: bold; color: {confidence_color};">
            {result['confidence']:.1f}% Confidence
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown("### Detailed Text Analysis")
    
    analysis = result['analysis']
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Word Count", analysis['word_count'])
    
    with col2:
        st.metric("Unique Words", f"{analysis['unique_word_ratio']:.0%}")
    
    with col3:
        st.metric("Avg Word Length", f"{analysis['avg_word_length']:.1f}")
    
    with col4:
        suspicious = analysis['suspicious_score']
        if suspicious > 0.5:
            st.metric("Suspicious Score", f"{suspicious:.0%}", delta="High")
        else:
            st.metric("Suspicious Score", f"{suspicious:.0%}")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Exclamation Marks", analysis['exclamation_count'])
    
    with col2:
        st.metric("Question Marks", analysis['question_count'])
    
    with col3:
        st.metric("Uppercase Ratio", f"{analysis['uppercase_ratio']:.1%}")
    
    with col4:
        st.metric("Characters", analysis['char_count'])
    
    st.markdown("---")
    st.markdown("### Sensational Language Detection")
    
    sens = analysis['sensational_words']
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if sens['breaking']:
            st.warning("'Breaking' detected")
        else:
            st.success("No 'Breaking'")
    
    with col2:
        if sens['exclusive']:
            st.warning("'Exclusive' detected")
        else:
            st.success("No 'Exclusive'")
    
    with col3:
        if sens['shocking']:
            st.warning("'Shocking' detected")
        else:
            st.success("No 'Shocking'")
    
    with col4:
        if sens['urgent']:
            st.warning("'Urgent' detected")
        else:
            st.success("No 'Urgent'")
    
    sensational_count = sum(sens.values())
    if sensational_count > 0:
        st.warning(f"{sensational_count} sensational word(s) detected! This is often a sign of fake news. Be cautious and verify from other sources.")
    
    with st.expander("View Analyzed Text", expanded=False):
        st.text_area(
            "Analyzed Text",
            value=result['text'],
            height=150,
            disabled=True,
            label_visibility="collapsed"
        )


# ==================== HISTORY TAB ====================
def render_history_tab(db):
    """Render prediction history"""
    
    st.markdown("### Prediction History")
    
    predictions = db.get_all_predictions(limit=50)
    
    if not predictions:
        st.info("No predictions yet! Go to the Detection tab to analyze news articles.")
        return
    
    df = pd.DataFrame(predictions)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['date'] = df['timestamp'].dt.strftime('%b %d, %Y %H:%M')
    
    display_df = df[['date', 'text_preview', 'prediction', 'confidence', 'word_count']].copy()
    display_df.columns = ['Date', 'Text Preview', 'Result', 'Confidence %', 'Words']
    display_df['Confidence %'] = display_df['Confidence %'].round(1)
    display_df = display_df.sort_values('Date', ascending=False)
    
    def style_result(val):
        if val == 'Real News':
            return 'background-color: #d4edda; color: #155724; font-weight: bold;'
        else:
            return 'background-color: #f8d7da; color: #721c24; font-weight: bold;'
    
    # ============ FIX: applymap → map ============
    st.dataframe(
        display_df.style.map(style_result, subset=['Result']),
        use_container_width=True,
        height=500,
        hide_index=True
    )
    

# ==================== ANALYTICS TAB ====================
def render_analytics_tab(db):
    """Render analytics dashboard"""
    
    st.markdown("### Analytics Dashboard")
    
    predictions = db.get_all_predictions(limit=10000)
    
    if not predictions:
        st.info("No data available for analytics. Start making predictions first!")
        return
    
    df = pd.DataFrame(predictions)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Predictions", len(df))
    
    with col2:
        real_pct = (df['prediction'] == 'Real News').mean() * 100
        st.metric("Real News %", f"{real_pct:.1f}%")
    
    with col3:
        avg_conf = df['confidence'].mean()
        st.metric("Avg Confidence", f"{avg_conf:.1f}%")
    
    with col4:
        avg_words = df['word_count'].mean()
        st.metric("Avg Words", f"{avg_words:.0f}")
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        real_count = len(df[df['prediction'] == 'Real News'])
        fake_count = len(df[df['prediction'] == 'Fake News'])
        
        fig = go.Figure(data=[go.Pie(
            labels=['Real News', 'Fake News'],
            values=[real_count, fake_count],
            hole=0.5,
            marker=dict(colors=['#38ef7d', '#f45c43']),
            textinfo='label+percent',
            textfont=dict(size=14)
        )])
        
        fig.update_layout(
            title={'text': 'Prediction Distribution', 'x': 0.5, 'xanchor': 'center', 'font': {'size': 18}},
            height=400,
            showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5)
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = px.histogram(
            df,
            x='confidence',
            color='prediction',
            nbins=15,
            color_discrete_map={'Real News': '#38ef7d', 'Fake News': '#f45c43'},
            labels={'confidence': 'Confidence %', 'prediction': 'Result'},
            opacity=0.8
        )
        
        fig.update_layout(
            title={'text': 'Confidence Score Distribution', 'x': 0.5, 'xanchor': 'center', 'font': {'size': 18}},
            height=400,
            showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5),
            bargap=0.1
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    if len(df) >= 2:
        st.markdown("---")
        st.markdown("### Prediction Timeline")
        
        df['date'] = df['timestamp'].dt.date
        daily = df.groupby(['date', 'prediction']).size().reset_index(name='count')
        
        fig = px.bar(
            daily,
            x='date',
            y='count',
            color='prediction',
            color_discrete_map={'Real News': '#38ef7d', 'Fake News': '#f45c43'},
            labels={'date': 'Date', 'count': 'Number of Predictions', 'prediction': 'Result'},
            barmode='stack'
        )
        
        fig.update_layout(
            title={'text': 'Daily Prediction Activity', 'x': 0.5, 'xanchor': 'center', 'font': {'size': 18}},
            height=400,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5)
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    st.markdown("### Word Count Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig = px.box(
            df,
            x='prediction',
            y='word_count',
            color='prediction',
            color_discrete_map={'Real News': '#38ef7d', 'Fake News': '#f45c43'},
            labels={'prediction': 'Result', 'word_count': 'Word Count'},
            points='outliers'
        )
        
        fig.update_layout(height=400, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = px.scatter(
            df,
            x='word_count',
            y='confidence',
            color='prediction',
            color_discrete_map={'Real News': '#38ef7d', 'Fake News': '#f45c43'},
            labels={'word_count': 'Word Count', 'confidence': 'Confidence %', 'prediction': 'Result'},
            opacity=0.7,
            size_max=10
        )
        
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)


# ==================== ABOUT TAB ====================
def render_about_tab():
    """Render about page"""
    
    st.markdown("## About This Project")
    
    st.markdown("""
    <div style="background: linear-gradient(135deg, #667eea, #764ba2); padding: 30px; border-radius: 15px; color: white; text-align: center; margin-bottom: 30px;">
        <h2>Fake News Detection System</h2>
        <p style="font-size: 1.2rem; opacity: 0.9;">Using Machine Learning & NLP to Combat Misinformation</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### Project Overview
        
        This Fake News Detection System uses advanced Machine Learning 
        and Natural Language Processing (NLP) techniques to analyze news 
        articles and predict whether they contain real or fake information.
        
        ### Technology Stack
        
        - **Python** - Core programming language
        - **Streamlit** - Web application framework
        - **Scikit-learn** - Machine learning algorithms
        - **NLTK** - Natural Language Processing
        - **Pandas** - Data manipulation & analysis
        - **Plotly** - Interactive visualizations
        - **SQLite** - Local database storage
        
        ### Educational Purpose
        
        This project was created for educational purposes to demonstrate:
        - Text classification using ML
        - NLP preprocessing techniques
        - Web app development with Streamlit
        - Data visualization best practices
        """)
    
    with col2:
        st.markdown("""
        ### How Detection Works
        
        **1. Text Preprocessing**
        - Lowercase conversion
        - URL & HTML tag removal
        - Special character removal
        - Stopword removal
        - Word stemming
        
        **2. Feature Extraction**
        - TF-IDF vectorization
        - 5000 most frequent terms
        - Unigrams & bigrams
        - Sublinear TF scaling
        
        **3. Machine Learning**
        - Logistic Regression classifier
        - Probability-based prediction
        - Confidence score calculation
        
        ### Model Performance
        
        - Algorithm: Logistic Regression
        - Training Accuracy: ~92-95%
        - Test Accuracy: ~85-90%
        - Cross-Validation: 5-fold
        - Features: 5000 TF-IDF terms
        """)
    
    st.markdown("---")
    st.markdown("### Important Limitations")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.warning("""
        **Accuracy Limitations**
        
        - Model accuracy is not 100%
        - Depends on training data quality
        - May miss sophisticated fake news
        - Limited to English language
        """)
    
    with col2:
        st.info("""
        **Usage Guidelines**
        
        - Use as an assistant tool only
        - Not a replacement for human judgment
        - Verify from multiple sources
        - Consider context and nuances
        """)
    
    with col3:
        st.error("""
        **When to be Cautious**
        
        - Very short text (< 20 words)
        - Mixed language content
        - Highly technical articles
        - Satirical content
        """)
    
    st.markdown("---")
    st.warning("""
    ### Disclaimer
    
    This tool is provided for educational and research purposes only. 
    The predictions made by this system should not be considered as absolute 
    truth or used as the sole basis for making important decisions.
    
    **Always:**
    - Verify information from multiple reliable sources
    - Use critical thinking when evaluating news
    - Consider the context and source of information
    - Report actual misinformation to appropriate authorities
    """)
    
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; padding: 20px;">
        <h3>Built for Learning</h3>
        <p>This project demonstrates practical application of:</p>
        <p>
            Python Programming | Machine Learning | Data Science | 
            Web Development | Natural Language Processing
        </p>
        <br>
        <p style="color: #666;">Made with ❤ | MIT License | v1.0.0</p>
    </div>
    """, unsafe_allow_html=True)


# ==================== MAIN APPLICATION ====================
def main():
    """Main application entry point"""
    
    load_custom_css()
    
    model, vectorizer, db = init_resources()
    
    model, vectorizer = load_model()
    
    render_sidebar()
    
    st.markdown("""
    <div class="app-header animate-fade-in">
        <h1>Fake News Detection System</h1>
        <p>Detect misinformation using Machine Learning & Natural Language Processing</p>
    </div>
    """, unsafe_allow_html=True)
    
    tab1, tab2, tab3, tab4 = st.tabs([
        "Detection", 
        "History", 
        "Analytics", 
        "About"
    ])
    
    with tab1:
        if model and vectorizer:
            render_detection_tab(model, vectorizer, db)
        else:
            st.error("Model Not Available. Run: python train_model.py")
    
    with tab2:
        render_history_tab(db)
    
    with tab3:
        render_analytics_tab(db)
    
    with tab4:
        render_about_tab()


if __name__ == "__main__":
    main()