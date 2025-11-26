import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import hashlib
import time
from datetime import datetime, timedelta

# Suppress warnings
import warnings
warnings.filterwarnings("ignore")

# -----------------------------------------------------------------------------
# UTILITIES & MOCK DATA GENERATORS
# -----------------------------------------------------------------------------

def generate_maintenance_data(n_samples=1000):
    np.random.seed(42)
    data = {
        'Temperature': np.random.normal(80, 15, n_samples),
        'Pressure': np.random.normal(35, 5, n_samples),
        'Vibration': np.random.normal(0.5, 0.2, n_samples),
        'UsageHours': np.random.randint(100, 5000, n_samples),
        'DaysSinceMaintenance': np.random.randint(1, 180, n_samples),
        'ComponentHealth': np.random.uniform(0.5, 1.0, n_samples),
        'ComponentID': np.random.choice([f'C-{i}' for i in range(1, 21)], n_samples)
    }
    df = pd.DataFrame(data)
    # Simulate failure based on conditions
    df['Failure'] = (
        (df['Temperature'] > 100) | 
        (df['Pressure'] > 45) | 
        (df['Vibration'] > 0.8) | 
        (df['ComponentHealth'] < 0.6)
    ).astype(int)
    # Add some noise
    flip_indices = np.random.choice(n_samples, size=int(n_samples*0.05), replace=False)
    df.loc[flip_indices, 'Failure'] = 1 - df.loc[flip_indices, 'Failure']
    return df

def generate_inventory_data():
    dates = pd.date_range(start='2022-01-01', periods=104, freq='W')
    data = []
    base_demand = 100
    for date in dates:
        # Seasonal pattern + trend
        seasonal = 20 * np.sin(2 * np.pi * date.week / 52)
        trend = 0.5 * (date.year - 2022) * 52 + date.week * 0.1
        noise = np.random.normal(0, 10)
        demand = int(max(0, base_demand + seasonal + trend + noise))
        data.append({'date': date, 'spare_part': demand})
    return pd.DataFrame(data).set_index('date')

def generate_sales_data():
    dates = pd.date_range(start='2020-01-01', periods=48, freq='M')
    data = {
        'date': dates,
        'sales': np.random.randint(200, 500, 48) + np.arange(48) * 5,
        'price': np.random.uniform(20000, 35000, 48),
        'marketing_spend': np.random.uniform(5000, 20000, 48),
        'festival_season': [1 if m in [10, 11, 12] else 0 for m in dates.month],
        'competitor_launches': np.random.randint(0, 3, 48)
    }
    df = pd.DataFrame(data)
    df['month'] = df['date'].dt.month
    df['year'] = df['date'].dt.year
    return df

def generate_sentiment_data():
    reviews = [
        "Great service, very quick!", "The engine noise is still there.", "Love the new model!",
        "Maintenance costs are too high.", "Smooth drive and great mileage.", "Dealership staff was rude.",
        "Excellent build quality.", "Battery drained too fast.", "Best purchase I've made.",
        "Spare parts are hard to find."
    ]
    # Fixed: reduced list to 2 items to match the size of probability array p=[0.6, 0.4]
    sentiments = ["Positive", "Negative"]
    dates = pd.date_range(end=datetime.now(), periods=100).tolist()
    
    data = {
        'Date': np.random.choice(dates, 500),
        'Review': np.random.choice(reviews, 500),
        'Sentiment': np.random.choice(sentiments, 500, p=[0.6, 0.4]), # Slightly biased towards positive
        'Car_Model': np.random.choice(['Model X', 'Model Y', 'Civic Type Z', 'Raptor F'], 500)
    }
    return pd.DataFrame(data)

def generate_geo_data():
    # Generate random points around a central lat/long (e.g., USA center)
    base_lat, base_lon = 39.8283, -98.5795
    n_dealers = 50
    data = {
        'lat': np.random.normal(base_lat, 5, n_dealers),
        'lon': np.random.normal(base_lon, 10, n_dealers),
        'Sales_Volume': np.random.randint(50, 500, n_dealers),
        'Region': np.random.choice(['North', 'South', 'East', 'West'], n_dealers)
    }
    return pd.DataFrame(data)

# -----------------------------------------------------------------------------
# AUTHENTICATION
# -----------------------------------------------------------------------------

USERS = {
    "admin": hashlib.sha256("admin123".encode()).hexdigest(),
    "user1": hashlib.sha256("password123".encode()).hexdigest(),
}

def verify_password(username, password):
    if username in USERS:
        return USERS[username] == hashlib.sha256(password.encode()).hexdigest()
    return False

def login_page():
    # Custom CSS for a professional look
    st.markdown("""
        <style>
        .login-header {
            font-size: 2.2rem;
            font-weight: 700;
            color: #1E3A8A; 
            margin-bottom: 0.5rem;
        }
        .login-subheader {
            font-size: 1.1rem;
            color: #4B5563;
            margin-bottom: 1.5rem;
        }
        .feature-item {
            display: flex;
            align-items: center;
            margin-bottom: 0.8rem;
            font-size: 1rem;
            color: #374151;
        }
        .feature-icon {
            margin-right: 12px;
            font-size: 1.2rem;
        }
        </style>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns([1.5, 1])

    with col1:
        st.markdown('<div class="login-header">MotorMinds</div>', unsafe_allow_html=True)
        st.markdown('<div class="login-subheader">Next-Gen Automotive Intelligence Platform</div>', unsafe_allow_html=True)
        
        # Professional Automotive Image
        st.image("https://images.unsplash.com/photo-1492144534655-ae79c964c9d7?ixlib=rb-4.0.3&auto=format&fit=crop&w=1000&q=80", 
                 caption="Data-Driven Performance", use_container_width=True)
        
        st.markdown("""
        <div style="margin-top: 20px;">
            <div class="feature-item">
                <span class="feature-icon">🔍</span>
                <span><b>Predictive Maintenance</b>: AI-driven failure forecasting.</span>
            </div>
            <div class="feature-item">
                <span class="feature-icon">📊</span>
                <span><b>Sales Analytics</b>: Real-time market insights.</span>
            </div>
            <div class="feature-item">
                <span class="feature-icon">⚡</span>
                <span><b>Live Telemetry</b>: Real-time IoT monitoring.</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("<br><br>", unsafe_allow_html=True) # Vertical spacer
        
        with st.container(border=True):
            st.markdown("### 🔐 Member Login")
            st.markdown("Welcome back! Please enter your details.")
            
            with st.form("login_form"):
                username = st.text_input("Username", placeholder="e.g. admin")
                password = st.text_input("Password", type="password", placeholder="••••••••")
                
                submitted = st.form_submit_button("Sign In", use_container_width=True, type="primary")
                
                if submitted:
                    if verify_password(username, password):
                        st.session_state['logged_in'] = True
                        st.session_state['username'] = username
                        st.rerun()
                    else:
                        st.error("Invalid username or password")

            st.markdown("""
                <div style="text-align: center; margin-top: 15px; font-size: 0.85em;">
                    <a href="#" style="color: #666; text-decoration: none; margin-right: 10px;">Forgot Password?</a>
                    <span style="color: #ccc;">|</span>
                    <a href="#" style="color: #666; text-decoration: none; margin-left: 10px;">Contact Support</a>
                </div>
            """, unsafe_allow_html=True)
            
            st.divider()
            st.info("💡 **Demo Access:**\n\nAdmin: `admin` / `admin123`\nUser: `user1` / `password123`")

# -----------------------------------------------------------------------------
# MAIN APP LOGIC
# -----------------------------------------------------------------------------

def main():
    if 'logged_in' not in st.session_state:
        st.session_state['logged_in'] = False

    if not st.session_state['logged_in']:
        login_page()
        return
    
    # Check Role
    username = st.session_state.get('username', 'User')
    is_admin = username == 'admin'
    role_name = "Administrator" if is_admin else "Standard User"
    
    # Sidebar
    with st.sidebar:
        st.image("https://img.icons8.com/color/96/000000/engine.png", width=60)
        st.title("MotorMinds")
        
        # Profile Badge
        with st.container(border=True):
            st.markdown(f"👤 **{username}**")
            st.caption(f"Role: {role_name}")
        
        if st.button("Logout", type="secondary", use_container_width=True):
            st.session_state['logged_in'] = False
            st.rerun()
        
        st.markdown("---")
        
        # Menu Permissions
        if is_admin:
            menu = [
                "Home", 
                "Predictive Maintenance", 
                "Spare Parts Forecasting", 
                "Car Sales Prediction",
                "Customer Sentiment", 
                "Geospatial Insights",
                "Real-time Simulator"
            ]
        else:
            # Restricted menu for standard users
            menu = [
                "Home",
                "Predictive Maintenance",
                "Real-time Simulator"
            ]
            
        choice = st.radio("Navigation", menu)
        
        if not is_admin:
            st.info("🔒 Advanced modules are restricted to Admin users.")

        st.markdown("---")
        st.caption("© 2024 MotorMinds Analytics")

    # Routing
    if choice == "Home":
        show_home_page(is_admin)
    elif choice == "Predictive Maintenance":
        run_predictive_maintenance()
    elif choice == "Spare Parts Forecasting":
        if is_admin: run_spare_parts_forecasting()
    elif choice == "Car Sales Prediction":
        if is_admin: run_car_sales_prediction()
    elif choice == "Customer Sentiment":
        if is_admin: run_sentiment_analysis()
    elif choice == "Geospatial Insights":
        if is_admin: run_geo_analysis()
    elif choice == "Real-time Simulator":
        run_realtime_simulator()

# -----------------------------------------------------------------------------
# PAGE FUNCTIONS
# -----------------------------------------------------------------------------

def show_home_page(is_admin):
    st.title("🚀 Welcome to MotorMinds")
    
    if is_admin:
        st.markdown("### Executive Dashboard")
        st.markdown("_Overview of strategic KPIs and system-wide performance._")
    else:
        st.markdown("### Operational Dashboard")
        st.markdown("_Overview of assigned vehicle monitoring and real-time status._")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        if is_admin:
            st.success("✅ **System Integrity**: All regional servers operational.")
            st.write("""
            **Admin Actions:**
            * Review **Sales Predictions** for Q4 strategy.
            * Check **Spare Parts Inventory** alerts.
            * Analyze **Customer Sentiment** trends.
            """)
        else:
            st.info("ℹ️ **Shift Update**: You are monitoring Sector 7.")
            st.write("""
            **User Actions:**
            * Monitor **Real-time Telemetry** for Vehicle V-X99.
            * Run **Predictive Maintenance** checks on reported sensors.
            """)
        
        st.subheader("Platform Capabilities")
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("""
            * **Predictive Maintenance**
            * **Real-time Simulator**
            """)
        with c2:
            if is_admin:
                st.markdown("""
                * **Sales & Inventory AI**
                * **Geospatial Analytics**
                """)

    with col2:
        # Metrics differ by role
        with st.container(border=True):
            if is_admin:
                st.metric(label="Total Revenue (YTD)", value="$4.2M", delta="+12%")
                st.metric(label="Global Fleet Health", value="98.4%", delta="+0.2%")
                st.metric(label="Active Alerts", value="3", delta="Normal", delta_color="off")
            else:
                st.metric(label="Vehicles Online", value="14", delta="Sector 7")
                st.metric(label="Pending Inspections", value="2", delta="-1")
                st.metric(label="Shift Efficiency", value="94%", delta="+2%")

def run_predictive_maintenance():
    st.header("🔧 Predictive Maintenance Analysis")
    st.markdown("Analyze sensor data to predict component failures using Machine Learning.")

    tabs = st.tabs(["Data & Model", "Visualizations", "Live Scoring"])

    with tabs[0]:
        col1, col2 = st.columns([1, 2])
        with col1:
            st.markdown("#### Data Source")
            data_source = st.radio("Select Source", ["Use Synthetic Data", "Upload CSV"])
        
        df = None
        if data_source == "Upload CSV":
            uploaded_file = st.file_uploader("Upload sensor data", type=["csv"])
            if uploaded_file:
                df = pd.read_csv(uploaded_file)
        else:
            df = generate_maintenance_data()
            st.success("Generated 1000 synthetic sensor records.")

        if df is not None:
            with col2:
                st.dataframe(df.head(5), use_container_width=True)

            # Modeling Logic
            feature_cols = ["Temperature", "Pressure", "Vibration", "UsageHours", "DaysSinceMaintenance", "ComponentHealth"]
            
            # Simple check if columns exist
            if not all(col in df.columns for col in feature_cols):
                st.error(f"Dataset must contain: {feature_cols}")
                return

            X = df[feature_cols].values
            y = df["Failure"].values
            
            # Train/Test Split
            from sklearn.model_selection import train_test_split
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
            
            # Model Training
            from sklearn.ensemble import RandomForestClassifier
            model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
            model.fit(X_train, y_train)
            
            y_pred = model.predict(X_test)
            
            # Metrics
            from sklearn.metrics import accuracy_score, f1_score
            acc = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            
            st.divider()
            m1, m2, m3 = st.columns(3)
            m1.metric("Model Accuracy", f"{acc*100:.1f}%")
            m2.metric("F1 Score", f"{f1:.3f}")
            m3.metric("Test Samples", len(X_test))

    with tabs[1]:
        if df is not None:
            st.subheader("Feature Correlations")
            fig, ax = plt.subplots(figsize=(10, 4))
            
            # Fix: Select only numeric columns for correlation matrix
            numeric_df = df.select_dtypes(include=[np.number])
            sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', ax=ax, fmt=".2f")
            
            st.pyplot(fig)
            
            st.subheader("Failure Distribution by Temperature")
            fig2, ax2 = plt.subplots()
            sns.histplot(data=df, x="Temperature", hue="Failure", kde=True, ax=ax2)
            st.pyplot(fig2)

    with tabs[2]:
        st.subheader("Single Component Scorer")
        with st.form("prediction_form"):
            c1, c2, c3 = st.columns(3)
            temp = c1.number_input("Temperature", 50, 150, 85)
            pres = c2.number_input("Pressure", 20, 60, 35)
            vib = c3.number_input("Vibration", 0.0, 1.0, 0.4)
            hours = c1.number_input("Usage Hours", 0, 10000, 2000)
            days = c2.number_input("Days Since Maint.", 0, 365, 45)
            health = c3.slider("Component Health Index", 0.0, 1.0, 0.9)
            
            if st.form_submit_button("Predict Failure Risk"):
                # Mock prediction usage
                input_data = np.array([[temp, pres, vib, hours, days, health]])
                if df is not None:
                    prob = model.predict_proba(input_data)[0][1]
                    pred = model.predict(input_data)[0]
                    
                    if pred == 1:
                        st.error(f"⚠️ High Failure Risk Detected! (Probability: {prob:.2f})")
                        st.warning("Recommendation: Schedule maintenance immediately.")
                    else:
                        st.success(f"✅ Component Healthy. (Failure Probability: {prob:.2f})")

def run_spare_parts_forecasting():
    st.header("📦 Spare Parts Demand Forecasting")
    
    data = generate_inventory_data()
    
    col1, col2 = st.columns([3, 1])
    with col1:
        st.line_chart(data)
    with col2:
        st.write("### Data Stats")
        st.write(data.describe())

    st.subheader("Forecast Settings")
    periods = st.slider("Weeks to Forecast", 1, 20, 10)
    
    # Simple Exponential Smoothing (Mocked for stability without complex statsmodels deps if needed)
    # Using simple moving average + trend for demo purposes if statsmodels fails, 
    # but here we assume standard environment.
    try:
        from statsmodels.tsa.holtwinters import ExponentialSmoothing
        model = ExponentialSmoothing(data['spare_part'], trend='add', seasonal='add', seasonal_periods=12).fit()
        forecast = model.forecast(periods)
        
        forecast_df = pd.DataFrame({'Forecast': forecast})
        
        st.subheader("Forecast Results")
        st.line_chart(pd.concat([data['spare_part'], forecast_df['Forecast']]))
        
        with st.expander("View Forecast Data"):
            st.dataframe(forecast_df)
            
    except Exception as e:
        st.error(f"Forecasting Error: {e}")
        st.info("Ensure statsmodels is installed.")

def run_car_sales_prediction():
    st.header("📈 Car Sales Prediction")
    
    df = generate_sales_data()
    
    st.subheader("Historical Sales Data")
    st.dataframe(df.head())
    
    # Correlation with Sales
    st.subheader("Impact Factors on Sales")
    corr = df[['sales', 'price', 'marketing_spend', 'competitor_launches']].corr()['sales']
    st.bar_chart(corr.drop('sales'))

    st.subheader("Future Sales Projector")
    with st.form("sales_form"):
        spend = st.slider("Marketing Spend ($)", 5000, 50000, 15000)
        price = st.slider("Average Car Price ($)", 20000, 40000, 28000)
        season = st.checkbox("Is Festival Season?")
        
        if st.form_submit_button("Estimate Sales"):
            # Simple linear formula mock
            base = 300
            spend_effect = (spend - 10000) / 100
            price_effect = (30000 - price) / 100
            season_effect = 50 if season else 0
            
            est_sales = base + spend_effect + price_effect + season_effect
            st.metric("Estimated Monthly Sales", f"{int(est_sales)} Units")

def run_sentiment_analysis():
    st.header("🗣️ Customer Sentiment Analysis")
    st.markdown("Analyze customer feedback to improve services and products.")
    
    df = generate_sentiment_data()
    
    c1, c2, c3 = st.columns(3)
    c1.metric("Total Reviews", len(df))
    c2.metric("Positive Sentiment", f"{len(df[df['Sentiment']=='Positive'])/len(df)*100:.1f}%")
    c3.metric("Negative Sentiment", f"{len(df[df['Sentiment']=='Negative'])/len(df)*100:.1f}%")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Sentiment Distribution")
        fig, ax = plt.subplots()
        df['Sentiment'].value_counts().plot.pie(autopct='%1.1f%%', ax=ax, colors=['#66b3ff', '#ff9999'])
        st.pyplot(fig)
        
    with col2:
        st.subheader("Recent Reviews")
        st.dataframe(df[['Review', 'Sentiment']].head(10), hide_index=True)
        
    st.subheader("Filter by Car Model")
    model_filter = st.selectbox("Select Model", df['Car_Model'].unique())
    filtered_df = df[df['Car_Model'] == model_filter]
    
    st.bar_chart(filtered_df['Sentiment'].value_counts())

def run_geo_analysis():
    st.header("🌍 Geospatial Sales Insights")
    st.markdown("Visualize dealership performance across different regions.")
    
    df = generate_geo_data()
    
    # Map Visualization
    st.subheader("Dealership Locations & Sales Volume")
    
    # Size points by sales volume
    st.map(df, latitude='lat', longitude='lon', size='Sales_Volume', color='#FF0000')
    
    st.subheader("Regional Performance")
    avg_sales = df.groupby('Region')['Sales_Volume'].mean()
    st.bar_chart(avg_sales)
    
    with st.expander("View Raw Geo Data"):
        st.dataframe(df)

def run_realtime_simulator():
    st.header("⚡ Real-time Vehicle Telemetry")
    st.markdown("Simulating live data stream from a connected vehicle engine (Vehicle ID: **V-X99**).")
    
    col1, col2, col3 = st.columns(3)
    placeholder1 = col1.empty()
    placeholder2 = col2.empty()
    placeholder3 = col3.empty()
    
    chart_placeholder = st.empty()
    
    data_container = []
    
    if st.button("Start Simulation (10s)"):
        for i in range(20): # Simulate 20 data points
            # Generate random live data
            rpm = np.random.randint(2000, 4000)
            temp = np.random.normal(90, 2)
            speed = np.random.randint(60, 120)
            
            # Update metrics
            placeholder1.metric("Engine RPM", f"{rpm}", delta=np.random.randint(-50, 50))
            placeholder2.metric("Engine Temp (°C)", f"{temp:.1f}", delta=f"{np.random.uniform(-0.5, 0.5):.1f}")
            placeholder3.metric("Speed (km/h)", f"{speed}", delta=np.random.randint(-2, 2))
            
            # Update chart
            data_container.append({"Time": i, "RPM": rpm, "Temp": temp})
            chart_df = pd.DataFrame(data_container)
            
            with chart_placeholder:
                st.line_chart(chart_df.set_index("Time")[["RPM", "Temp"]])
            
            time.sleep(0.5)
        st.success("Simulation Complete")

# -----------------------------------------------------------------------------
# RUN ENTRY POINT
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    st.set_page_config(
        page_title="MotorMinds", 
        layout="wide",
        page_icon="🚗"
    )
    main()