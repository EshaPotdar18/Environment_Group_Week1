"""
Modern AQI Monitoring Web App with Authentication
------------------------------------------------
A sophisticated Streamlit application for air quality analysis, visualization, 
and AQI prediction using machine learning with user authentication.
"""

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import joblib
import config
import hashlib
import json
from datetime import datetime, timedelta
import time

try:
    from xgboost import XGBClassifier
    xgb_available = True
except ImportError:
    xgb_available = False

# Configure page
st.set_page_config(
    page_title="Esha AirWatch Pro - Air Quality Monitoring",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern styling
st.markdown("""
<style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@400;700&family=Source+Sans+Pro:wght@300;400;600&display=swap');
    
    /* Global styling */
    .main-header {
        font-family: 'Playfair Display', serif;
        font-weight: 700;
        color: #be123c;
        text-align: center;
        margin-bottom: 2rem;
        font-size: 3rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    
    .sub-header {
        font-family: 'Playfair Display', serif;
        font-weight: 400;
        color: #be123c;
        margin-bottom: 1rem;
        font-size: 1.8rem;
    }
    
    .metric-card {
        background: linear-gradient(135deg, #fdf2f8 0%, #ffffff 100%);
        padding: 1.5rem;
        border-radius: 12px;
        border: 1px solid #e5e7eb;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);
        margin-bottom: 1rem;
        transition: transform 0.2s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 15px rgba(0, 0, 0, 0.1);
    }
    
    .login-container {
        max-width: 400px;
        margin: 0 auto;
        padding: 2rem;
        background: linear-gradient(135deg, #fdf2f8 0%, #ffffff 100%);
        border-radius: 16px;
        box-shadow: 0 10px 25px rgba(0, 0, 0, 0.1);
        border: 1px solid #e5e7eb;
    }
    
    .sidebar-content {
        font-family: 'Source Sans Pro', sans-serif;
    }
    
    .success-message {
        background: linear-gradient(135deg, #dcfce7 0%, #f0fdf4 100%);
        color: #166534;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #22c55e;
        margin: 1rem 0;
    }
    
    .warning-message {
        background: linear-gradient(135deg, #fef3c7 0%, #fffbeb 100%);
        color: #92400e;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #f59e0b;
        margin: 1rem 0;
    }
    
    .info-card {
        background: linear-gradient(135deg, #dbeafe 0%, #f0f9ff 100%);
        color: #1e40af;
        padding: 1.5rem;
        border-radius: 12px;
        border: 1px solid #93c5fd;
        margin: 1rem 0;
    }
    
    .feature-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
        gap: 1.5rem;
        margin: 2rem 0;
    }
    
    .stButton > button {
        background: linear-gradient(135deg, #be123c 0%, #ec4899 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.75rem 2rem;
        font-weight: 600;
        font-family: 'Source Sans Pro', sans-serif;
        transition: all 0.3s ease;
        box-shadow: 0 4px 6px rgba(190, 18, 60, 0.2);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 15px rgba(190, 18, 60, 0.3);
    }
    
    .nav-header {
        background: linear-gradient(135deg, #be123c 0%, #ec4899 100%);
        color: white;
        padding: 1rem;
        border-radius: 8px;
        margin-bottom: 1rem;
        text-align: center;
        font-family: 'Playfair Display', serif;
        font-weight: 700;
    }
</style>
""", unsafe_allow_html=True)

# Authentication functions
def hash_password(password):
    """Hash password using SHA-256"""
    return hashlib.sha256(password.encode()).hexdigest()

def load_users():
    """Load users from JSON file"""
    try:
        with open('users.json', 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return {}

def save_users(users):
    """Save users to JSON file"""
    with open('users.json', 'w') as f:
        json.dump(users, f)

def authenticate_user(username, password):
    """Authenticate user credentials"""
    users = load_users()
    if username in users:
        return users[username]['password'] == hash_password(password)
    return False

def register_user(username, password, email):
    """Register a new user"""
    users = load_users()
    if username in users:
        return False, "Username already exists"
    
    users[username] = {
        'password': hash_password(password),
        'email': email,
        'created_at': datetime.now().isoformat(),
        'last_login': None
    }
    save_users(users)
    return True, "User registered successfully"

def update_last_login(username):
    """Update user's last login time"""
    users = load_users()
    if username in users:
        users[username]['last_login'] = datetime.now().isoformat()
        save_users(users)

# Login/Registration UI
def show_auth_page():
    """Display authentication page"""
    st.markdown('<div class="main-header">🌍 Esha AirWatch Pro</div>', unsafe_allow_html=True)
    st.markdown('<div style="text-align: center; color: #475569; font-size: 1.2rem; margin-bottom: 3rem;">Advanced Air Quality Monitoring & Prediction Platform</div>', unsafe_allow_html=True)
    
    # Create tabs for login and registration
    tab1, tab2 = st.tabs(["🔐 Login", "📝 Register"])
    
    with tab1:
        # st.markdown('<div class="login-container">', unsafe_allow_html=True)
        st.markdown("### Welcome Back")
        st.markdown("Please sign in to access your dashboard")
        
        with st.form("login_form"):
            username = st.text_input("Username", placeholder="Enter your username")
            password = st.text_input("Password", type="password", placeholder="Enter your password")
            login_button = st.form_submit_button("Sign In", use_container_width=True)
            
            if login_button:
                if username and password:
                    if authenticate_user(username, password):
                        st.session_state.authenticated = True
                        st.session_state.username = username
                        update_last_login(username)
                        st.success("Login successful! Redirecting...")
                        time.sleep(1)
                        st.rerun()
                    else:
                        st.error("Invalid username or password")
                else:
                    st.warning("Please enter both username and password")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    with tab2:
        # st.markdown('<div class="login-container">', unsafe_allow_html=True)
        st.markdown("### Create Account")
        st.markdown("Join Esha AirWatch Pro to start monitoring air quality")
        
        with st.form("register_form"):
            new_username = st.text_input("Username", placeholder="Choose a username")
            new_email = st.text_input("Email", placeholder="Enter your email")
            new_password = st.text_input("Password", type="password", placeholder="Create a password")
            confirm_password = st.text_input("Confirm Password", type="password", placeholder="Confirm your password")
            register_button = st.form_submit_button("Create Account", use_container_width=True)
            
            if register_button:
                if new_username and new_email and new_password and confirm_password:
                    if new_password == confirm_password:
                        success, message = register_user(new_username, new_password, new_email)
                        if success:
                            st.success(message)
                            st.info("Please switch to the Login tab to sign in")
                        else:
                            st.error(message)
                    else:
                        st.error("Passwords do not match")
                else:
                    st.warning("Please fill in all fields")
        
        st.markdown('</div>', unsafe_allow_html=True)

# Main application
def show_main_app():
    """Display the main application after authentication"""
    
    # Header with user info
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        st.markdown('<div class="main-header">🌍 Esha AirWatch Pro Dashboard</div>', unsafe_allow_html=True)
    with col3:
        if st.button("🚪 Logout", key="logout"):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()
    
    # Welcome message
    st.markdown(f'<div class="info-card">Welcome back, <strong>{st.session_state.username}</strong>! Ready to analyze air quality data?</div>', unsafe_allow_html=True)
    
    # Sidebar navigation
    st.sidebar.markdown('<div class="nav-header">Navigation</div>', unsafe_allow_html=True)
    
    # Sidebar logo and info
    st.sidebar.markdown("""
    <div style='text-align: center; margin: 1rem 0;'>
        <div style='font-size: 3rem;'>🌍</div>
        <h3 style='color: #be123c; margin: 0.5rem 0;'>Esha AirWatch Pro</h3>
        <p style='color: #475569; font-size: 0.9rem;'>Advanced Air Quality Analytics</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Navigation menu
    page = st.sidebar.selectbox(
        "Select Page",
        ["📊 Dashboard", "🔬 Model Training", "🎯 Predictions", "📈 Analytics", "⚙️ Settings"],
        key="navigation"
    )
    
    # Page routing
    if page == "📊 Dashboard":
        show_dashboard()
    elif page == "🔬 Model Training":
        show_model_training()
    elif page == "🎯 Predictions":
        show_predictions()
    elif page == "📈 Analytics":
        show_analytics()
    elif page == "⚙️ Settings":
        show_settings()

def show_dashboard():
    """Display the main dashboard"""
    st.markdown('<div class="sub-header">📊 Air Quality Dashboard</div>', unsafe_allow_html=True)
    
    # Load data
    try:
        df = pd.read_csv("data/processed_aqi.csv")
        
        # Quick stats
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown("""
            <div class="metric-card">
                <h3 style="color: #be123c; margin: 0;">Total Records</h3>
                <h2 style="color: #475569; margin: 0.5rem 0;">{:,}</h2>
                <p style="color: #6b7280; margin: 0;">Data points analyzed</p>
            </div>
            """.format(len(df)), unsafe_allow_html=True)
        
        with col2:
            unique_cities = df['City'].nunique() if 'City' in df.columns else 0
            st.markdown("""
            <div class="metric-card">
                <h3 style="color: #be123c; margin: 0;">Cities Monitored</h3>
                <h2 style="color: #475569; margin: 0.5rem 0;">{}</h2>
                <p style="color: #6b7280; margin: 0;">Locations covered</p>
            </div>
            """.format(unique_cities), unsafe_allow_html=True)
        
        with col3:
            aqi_categories = df['AQI_Bucket'].nunique() if 'AQI_Bucket' in df.columns else 0
            st.markdown("""
            <div class="metric-card">
                <h3 style="color: #be123c; margin: 0;">AQI Categories</h3>
                <h2 style="color: #475569; margin: 0.5rem 0;">{}</h2>
                <p style="color: #6b7280; margin: 0;">Quality levels</p>
            </div>
            """.format(aqi_categories), unsafe_allow_html=True)
        
        with col4:
            avg_aqi = df['AQI'].mean() if 'AQI' in df.columns else 0
            st.markdown("""
            <div class="metric-card">
                <h3 style="color: #be123c; margin: 0;">Average AQI</h3>
                <h2 style="color: #475569; margin: 0.5rem 0;">{:.1f}</h2>
                <p style="color: #6b7280; margin: 0;">Overall air quality</p>
            </div>
            """.format(avg_aqi), unsafe_allow_html=True)
        
        # Data preview
        st.markdown('<div class="sub-header">📋 Data Overview</div>', unsafe_allow_html=True)
        st.dataframe(df.head(10), use_container_width=True)
        
        # Quick visualization
        if 'AQI_Bucket' in df.columns:
            st.markdown('<div class="sub-header">📊 AQI Distribution</div>', unsafe_allow_html=True)
            fig = px.histogram(df, x='AQI_Bucket', title="Distribution of AQI Categories",
                             color_discrete_sequence=['#be123c'])
            fig.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font_family="Source Sans Pro"
            )
            st.plotly_chart(fig, use_container_width=True)
            
    except FileNotFoundError:
        st.error("Data file not found. Please upload your data first.")

def show_model_training():
    """Display model training interface"""
    st.markdown('<div class="sub-header">🔬 Machine Learning Model Training</div>', unsafe_allow_html=True)
    
    # Data upload section
    st.markdown("### 📁 Data Upload")
    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"], help="Upload your air quality data in CSV format.")
    
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.success(f"Data uploaded successfully! {len(df)} records loaded.")
    else:
        try:
            df = pd.read_csv("data/processed_aqi.csv")
            st.info("Using default processed_aqi.csv")
        except FileNotFoundError:
            st.error("No data available. Please upload a CSV file.")
            return
    
    # Preprocessing options
    st.markdown("### ⚙️ Preprocessing Options")
    col1, col2 = st.columns(2)
    
    with col1:
        missing_strategy = st.selectbox(
            "Handle missing values:",
            ["Fill with mean", "Fill with median", "Drop rows"],
            help="Choose how to handle missing values in your data."
        )
    
    with col2:
        remove_outliers = st.checkbox("Remove outliers (Z-score > 3)")
    
    # Data preprocessing (same logic as original)
    df = df.dropna(subset=["AQI_Bucket"])
    df_numeric = df.select_dtypes(include=[np.number]).columns.tolist()
    
    if missing_strategy == "Fill with mean":
        df[df_numeric] = df[df_numeric].fillna(df[df_numeric].mean())
    elif missing_strategy == "Fill with median":
        df[df_numeric] = df[df_numeric].fillna(df[df_numeric].median())
    else:
        df = df.dropna()
    
    if remove_outliers:
        z_scores = np.abs((df[df_numeric] - df[df_numeric].mean()) / df[df_numeric].std(ddof=0))
        outlier_mask = (z_scores > 3).any(axis=1)
        n_outliers = outlier_mask.sum()
        df = df[~outlier_mask]
        st.info(f"Removed {n_outliers} outlier rows.")
    
    # Feature engineering
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        df["Year"] = df["Date"].dt.year
        df["Month"] = df["Date"].dt.month
        df["Day"] = df["Date"].dt.day
        df["Season"] = df["Date"].dt.month % 12 // 3 + 1
    
    # Encode categorical features
    for col in ["City", "Station"]:
        if col in df.columns:
            df[col] = LabelEncoder().fit_transform(df[col].astype(str))
    
    # Encode target
    label_enc = LabelEncoder()
    df["AQI_Bucket"] = label_enc.fit_transform(df["AQI_Bucket"].astype(str))
    
    # Feature selection
    st.markdown("### 🎯 Feature Selection")
    all_features = [c for c in df.columns if c not in ["AQI_Bucket", "Date"]]
    selected_features = st.multiselect(
        "Select features for modeling:",
        all_features,
        default=all_features,
        help="Choose which features to use for model training."
    )
    
    if not selected_features:
        st.warning("Please select at least one feature.")
        return
    
    X = df[selected_features]
    y = df["AQI_Bucket"]
    
    # Remove classes with <2 samples
    class_counts = y.value_counts()
    valid_classes = class_counts[class_counts >= 2].index
    mask = y.isin(valid_classes)
    X = X[mask]
    y = y[mask]
    
    if X.shape[0] == 0:
        st.error("No data available for training after filtering.")
        return
    
    # Model configuration
    st.markdown("### 🤖 Model Configuration")
    col1, col2 = st.columns(2)
    
    with col1:
        model_name = st.selectbox(
            "Choose model:",
            ["Random Forest", "Logistic Regression"] + (["XGBoost"] if xgb_available else []),
            help="Select a machine learning model for AQI prediction."
        )
    
    with col2:
        test_size = st.slider("Test size (%)", 10, 50, 20)
    
    # Feature scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=test_size/100, stratify=y, random_state=42
    )
    
    # Model selection
    if model_name == "Random Forest":
        model = RandomForestClassifier(n_estimators=200, random_state=42)
    elif model_name == "Logistic Regression":
        model = LogisticRegression(max_iter=1000, random_state=42)
    else:
        num_classes = len(np.unique(y))
        model = XGBClassifier(
            use_label_encoder=False,
            eval_metric='mlogloss',
            objective='multi:softprob',
            num_class=num_classes,
            random_state=42
        )
    
    # Training button
    if st.button("🚀 Train Model", type="primary"):
        with st.spinner("Training model..."):
            # Train model
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            
            # Store in session state
            st.session_state['model'] = model
            st.session_state['scaler'] = scaler
            st.session_state['selected_features'] = selected_features
            st.session_state['label_encoder'] = label_enc
            
            # Save model
            try:
                joblib.dump(model, config.MODEL_PATH)
                joblib.dump(scaler, config.SCALER_PATH)
                st.success("Model and scaler saved successfully!")
            except Exception as e:
                st.error(f"Error saving model: {e}")
            
            # Show results
            st.markdown("### 📊 Training Results")
            
            # Classification report
            unique_labels = np.unique(np.concatenate([y_test, y_pred]))
            class_names = label_enc.inverse_transform(unique_labels)
            
            st.text("Classification Report:")
            st.text(classification_report(y_test, y_pred, labels=unique_labels, target_names=class_names))
            
            # Confusion matrix
            cm = confusion_matrix(y_test, y_pred, labels=unique_labels)
            fig = px.imshow(cm, 
                          x=class_names, 
                          y=class_names,
                          color_continuous_scale='Blues',
                          title="Confusion Matrix")
            fig.update_layout(
                xaxis_title="Predicted",
                yaxis_title="Actual",
                font_family="Source Sans Pro"
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Feature importance
            if hasattr(model, "feature_importances_"):
                importances = model.feature_importances_
                if len(importances) == len(selected_features):
                    feat_imp = pd.DataFrame({
                        'feature': selected_features,
                        'importance': importances
                    }).sort_values('importance', ascending=True)
                    
                    fig = px.bar(feat_imp, x='importance', y='feature', 
                               orientation='h', title="Feature Importance",
                               color_discrete_sequence=['#be123c'])
                    fig.update_layout(
                        plot_bgcolor='rgba(0,0,0,0)',
                        paper_bgcolor='rgba(0,0,0,0)',
                        font_family="Source Sans Pro"
                    )
                    st.plotly_chart(fig, use_container_width=True)
            
            st.balloons()

def show_predictions():
    """Display prediction interface"""
    st.markdown('<div class="sub-header">🎯 AQI Predictions</div>', unsafe_allow_html=True)
    
    # Check if model is available
    if 'model' not in st.session_state:
        st.warning("No trained model found. Please train a model first or load an existing one.")
        
        if st.button("📂 Load Saved Model"):
            try:
                model = joblib.load(config.MODEL_PATH)
                scaler = joblib.load(config.SCALER_PATH)
                st.session_state['model'] = model
                st.session_state['scaler'] = scaler
                st.success("Model loaded successfully!")
                st.rerun()
            except Exception as e:
                st.error(f"Error loading model: {e}")
        return
    
    st.markdown("### 🔮 Make Predictions")
    st.info("Enter the environmental parameters below to predict the AQI category.")
    
    # Get feature names
    if 'selected_features' in st.session_state:
        features = st.session_state['selected_features']
    else:
        st.error("Feature information not available. Please retrain the model.")
        return
    
    # Create input form
    input_data = {}
    
    # Organize inputs in columns
    num_cols = min(3, len(features))
    cols = st.columns(num_cols)
    
    for i, feature in enumerate(features):
        col_idx = i % num_cols
        with cols[col_idx]:
            # Set reasonable defaults based on feature name
            if 'PM2.5' in feature:
                default_val = 25.0
                help_text = "PM2.5 concentration (μg/m³)"
            elif 'PM10' in feature:
                default_val = 50.0
                help_text = "PM10 concentration (μg/m³)"
            elif 'NO2' in feature:
                default_val = 40.0
                help_text = "NO2 concentration (μg/m³)"
            elif 'SO2' in feature:
                default_val = 20.0
                help_text = "SO2 concentration (μg/m³)"
            elif 'CO' in feature:
                default_val = 1.0
                help_text = "CO concentration (mg/m³)"
            elif 'O3' in feature:
                default_val = 100.0
                help_text = "O3 concentration (μg/m³)"
            else:
                default_val = 0.0
                help_text = f"Enter value for {feature}"
            
            input_data[feature] = st.number_input(
                feature,
                value=default_val,
                help=help_text,
                key=f"input_{feature}"
            )
    
    # Prediction button
    if st.button("🎯 Predict AQI Category", type="primary"):
        try:
            # Prepare input data
            input_df = pd.DataFrame([input_data])
            input_scaled = st.session_state['scaler'].transform(input_df)
            
            # Make prediction
            prediction = st.session_state['model'].predict(input_scaled)[0]
            prediction_proba = st.session_state['model'].predict_proba(input_scaled)[0]
            
            # Get class name
            if 'label_encoder' in st.session_state:
                class_name = st.session_state['label_encoder'].inverse_transform([prediction])[0]
            else:
                class_name = f"Category {prediction}"
            
            # Display result
            st.markdown("### 📊 Prediction Result")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown(f"""
                <div class="metric-card" style="text-align: center;">
                    <h3 style="color: #be123c; margin: 0;">Predicted AQI Category</h3>
                    <h1 style="color: #475569; margin: 1rem 0; font-size: 2.5rem;">{class_name}</h1>
                    <p style="color: #6b7280; margin: 0;">Based on input parameters</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                # Show confidence scores
                if 'label_encoder' in st.session_state:
                    classes = st.session_state['label_encoder'].classes_
                    proba_df = pd.DataFrame({
                        'Category': classes,
                        'Probability': prediction_proba
                    }).sort_values('Probability', ascending=True)
                    
                    fig = px.bar(proba_df, x='Probability', y='Category',
                               orientation='h', title="Prediction Confidence",
                               color_discrete_sequence=['#ec4899'])
                    fig.update_layout(
                        plot_bgcolor='rgba(0,0,0,0)',
                        paper_bgcolor='rgba(0,0,0,0)',
                        font_family="Source Sans Pro"
                    )
                    st.plotly_chart(fig, use_container_width=True)
            
        except Exception as e:
            st.error(f"Error making prediction: {e}")

def show_analytics():
    """Display analytics and visualizations"""
    st.markdown('<div class="sub-header">📈 Advanced Analytics</div>', unsafe_allow_html=True)
    
    try:
        df = pd.read_csv("data/processed_aqi.csv")
        
        # Time series analysis if Date column exists
        if 'Date' in df.columns:
            st.markdown("### 📅 Time Series Analysis")
            df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
            
            if 'AQI' in df.columns:
                # Group by date and calculate mean AQI
                daily_aqi = df.groupby('Date')['AQI'].mean().reset_index()
                
                fig = px.line(daily_aqi, x='Date', y='AQI', 
                            title="AQI Trend Over Time",
                            color_discrete_sequence=['#be123c'])
                fig.update_layout(
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    font_family="Source Sans Pro"
                )
                st.plotly_chart(fig, use_container_width=True)
        
        # Correlation analysis
        st.markdown("### 🔗 Correlation Analysis")
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 1:
            corr_matrix = df[numeric_cols].corr()
            
            fig = px.imshow(corr_matrix,
                          color_continuous_scale='RdBu',
                          title="Feature Correlation Matrix")
            fig.update_layout(
                font_family="Source Sans Pro"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Distribution analysis
        st.markdown("### 📊 Distribution Analysis")
        
        if 'AQI' in df.columns:
            col1, col2 = st.columns(2)
            
            with col1:
                fig = px.histogram(df, x='AQI', nbins=30,
                                 title="AQI Distribution",
                                 color_discrete_sequence=['#be123c'])
                fig.update_layout(
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    font_family="Source Sans Pro"
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                fig = px.box(df, y='AQI',
                           title="AQI Box Plot",
                           color_discrete_sequence=['#ec4899'])
                fig.update_layout(
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    font_family="Source Sans Pro"
                )
                st.plotly_chart(fig, use_container_width=True)
        
        # City-wise analysis if City column exists
        if 'City' in df.columns and 'AQI' in df.columns:
            st.markdown("### 🏙️ City-wise Analysis")
            city_aqi = df.groupby('City')['AQI'].agg(['mean', 'count']).reset_index()
            city_aqi = city_aqi[city_aqi['count'] >= 10]  # Filter cities with enough data
            
            if not city_aqi.empty:
                fig = px.bar(city_aqi.head(10), x='City', y='mean',
                           title="Average AQI by City (Top 10)",
                           color_discrete_sequence=['#be123c'])
                fig.update_layout(
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    font_family="Source Sans Pro",
                    xaxis_tickangle=-45
                )
                st.plotly_chart(fig, use_container_width=True)
        
    except FileNotFoundError:
        st.error("Data file not found. Please upload your data first.")
    except Exception as e:
        st.error(f"Error in analytics: {e}")

def show_settings():
    """Display settings page"""
    st.markdown('<div class="sub-header">⚙️ Settings</div>', unsafe_allow_html=True)
    
    # User profile section
    st.markdown("### 👤 User Profile")
    users = load_users()
    current_user = users.get(st.session_state.username, {})
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.info(f"**Username:** {st.session_state.username}")
        st.info(f"**Email:** {current_user.get('email', 'Not provided')}")
        
    with col2:
        created_at = current_user.get('created_at')
        if created_at:
            created_date = datetime.fromisoformat(created_at).strftime("%B %d, %Y")
            st.info(f"**Member since:** {created_date}")
        
        last_login = current_user.get('last_login')
        if last_login:
            login_date = datetime.fromisoformat(last_login).strftime("%B %d, %Y at %I:%M %p")
            st.info(f"**Last login:** {login_date}")
    
    # Application settings
    st.markdown("### 🔧 Application Settings")
    
    # Model management
    st.markdown("#### 🤖 Model Management")
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("📂 Load Saved Model"):
            try:
                model = joblib.load(config.MODEL_PATH)
                scaler = joblib.load(config.SCALER_PATH)
                st.session_state['model'] = model
                st.session_state['scaler'] = scaler
                st.success("Model loaded successfully!")
            except Exception as e:
                st.error(f"Error loading model: {e}")
    
    with col2:
        if st.button("🗑️ Clear Session Data"):
            keys_to_keep = ['authenticated', 'username']
            keys_to_remove = [key for key in st.session_state.keys() if key not in keys_to_keep]
            for key in keys_to_remove:
                del st.session_state[key]
            st.success("Session data cleared!")
    
    # Data management
    st.markdown("#### 📁 Data Management")
    
    # Show current data info
    try:
        df = pd.read_csv("data/processed_aqi.csv")
        st.info(f"Current dataset: {len(df)} records, {len(df.columns)} columns")
        
        if st.button("📊 Show Data Summary"):
            st.markdown("**Data Summary:**")
            st.dataframe(df.describe(), use_container_width=True)
            
    except FileNotFoundError:
        st.warning("No data file found.")
    
    # About section
    st.markdown("### ℹ️ About Esha AirWatch Pro")
    st.markdown("""
    <div class="info-card">
        <h4>Esha AirWatch Pro v2.0</h4>
        <p>Advanced Air Quality Monitoring & Prediction Platform</p>
        <ul>
            <li>🔐 Secure user authentication</li>
            <li>📊 Interactive data visualization</li>
            <li>🤖 Machine learning predictions</li>
            <li>📈 Advanced analytics</li>
            <li>🌍 Multi-city monitoring</li>
        </ul>
        <p><strong>Built with:</strong> Streamlit, Scikit-learn, Plotly, Pandas</p>
    </div>
    """, unsafe_allow_html=True)

# Main application logic
def main():
    """Main application entry point"""
    
    # Initialize session state
    if 'authenticated' not in st.session_state:
        st.session_state.authenticated = False
    
    # Show appropriate page based on authentication status
    if st.session_state.authenticated:
        show_main_app()
    else:
        show_auth_page()

if __name__ == "__main__":
    main()
