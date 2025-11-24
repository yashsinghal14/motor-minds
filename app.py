import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Suppress warnings
import warnings
warnings.filterwarnings("ignore")

import hashlib

# User database
USERS = {
    "admin": hashlib.sha256("admin123".encode()).hexdigest(),
    "user1": hashlib.sha256("password123".encode()).hexdigest(),
}

def verify_password(username, password):
    if username in USERS:
        return USERS[username] == hashlib.sha256(password.encode()).hexdigest()
    return False

def login_page():
    st.title("🔐 MotorMinds Login")
    with st.form("login_form"):
        st.subheader("Please Login to Continue")
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        if st.form_submit_button("Login"):
            if verify_password(username, password):
                st.session_state['logged_in'] = True
                st.session_state['username'] = username
                st.rerun()
            else:
                st.error("Invalid credentials")
    st.info("**Demo:** Username: `admin` | Password: `admin123`")

# Initialize session state
if 'logged_in' not in st.session_state:
    st.session_state['logged_in'] = False

# Set up the main function
def main():
    if not st.session_state['logged_in']:
        login_page()
        return
    
    # Add logout button in sidebar
    with st.sidebar:
        st.markdown("---")
        st.write(f"👤 **{st.session_state['username']}**")
        if st.button("Logout"):
            st.session_state['logged_in'] = False
            st.rerun()
    
    # Your existing code continues here...
    st.set_page_config(page_title="MotorMinds", layout="wide")
    st.set_page_config(page_title="MotorMinds", layout="wide")
    st.title("MotorMinds - Automotive Smart Insights Platform")

    # Sidebar menu
    menu = ["Home", "Predictive Maintenance", "Spare Parts Demand Forecasting", "Car Sales Prediction"]
    choice = st.sidebar.selectbox("Select Analysis", menu)

    if choice == "Home":
        show_home_page()
    elif choice == "Predictive Maintenance":
        run_predictive_maintenance()
    elif choice == "Spare Parts Demand Forecasting":
        run_spare_parts_forecasting()
    elif choice == "Car Sales Prediction":
        run_car_sales_prediction()

def show_home_page():
    st.write("""
    ## Welcome to MotorMinds

    **MotorMinds** is an Automotive Smart Insights Platform that leverages advanced data analytics and machine learning to provide accurate forecasts and predictions for the automotive industry. Our platform helps automotive companies optimize inventory, improve customer satisfaction, and increase operational efficiency.

    ### Key Features

    - **Vehicle Sales Forecasting**: Predict future vehicle sales using time series analysis and machine learning models.
    - **Predictive Maintenance**: Anticipate maintenance needs and potential component failures before they occur.
    - **Spare Parts Demand Prediction**: Accurately forecast spare parts demand to optimize inventory levels.

    ### Benefits

    - **Optimize Inventory Management**: Reduce holding costs and prevent stockouts by maintaining optimal inventory levels.
    - **Enhance Vehicle Reliability**: Minimize unexpected breakdowns and extend vehicle lifespan through predictive maintenance.
    - **Adapt to Market Changes**: Stay ahead of market shifts by adapting quickly based on accurate sales forecasts.

    ### How to Use This App

    - Navigate through the app using the sidebar menu.
    - Upload your own data or adjust parameters in each section to see customized insights.
    - Visualize results through interactive graphs and charts.

    **Get started by selecting an analysis from the sidebar!**
    """)

# Predictive Maintenance Analysis
def run_predictive_maintenance():
    st.subheader("Predictive Maintenance Analysis")

    # Upload data
    uploaded_file = st.file_uploader("Upload your predictive maintenance data CSV file", type=["csv"])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
    else:
        st.info("Using default dataset.")
        df = pd.read_csv("data/predictive_maintenance_data.csv")

    # Data preprocessing
    df["Failure"] = df["Failure"].astype(int)

    # Feature selection
    feature_cols = ["Temperature", "Pressure", "Vibration", "UsageHours", "DaysSinceMaintenance", "ComponentHealth"]
    X = df[feature_cols].values
    y = df["Failure"].values

    # Parameter adjustments
    st.sidebar.subheader("Model Parameters")
    max_depth = st.sidebar.slider("Decision Tree Max Depth", 1, 10, 6)
    min_samples_split = st.sidebar.slider("Decision Tree Min Samples Split", 2, 100, 50)
    min_samples_leaf = st.sidebar.slider("Decision Tree Min Samples Leaf", 1, 50, 20)
    lr_C = st.sidebar.slider("Logistic Regression Inverse Regularization Strength (C)", 0.001, 1.0, 0.01)

    # Function to split data by components to prevent data leakage
    def split_by_component(df, test_size=0.3):
        components = df["ComponentID"].unique()
        np.random.shuffle(components)
        split_point = int(len(components) * (1 - test_size))
        train_components = components[:split_point]
        test_components = components[split_point:]
        
        train_df = df[df["ComponentID"].isin(train_components)]
        test_df = df[df["ComponentID"].isin(test_components)]
        
        X_train = train_df[feature_cols].values
        y_train = train_df["Failure"].values.astype(int)
        X_test = test_df[feature_cols].values
        y_test = test_df["Failure"].values.astype(int)
        
        return X_train, X_test, y_train, y_test, train_components, test_components

    # Split the data
    X_train, X_test, y_train, y_test, train_components, test_components = split_by_component(df, test_size=0.3)

    # Scale the features
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Implement Decision Tree and Logistic Regression
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.linear_model import LogisticRegression

    # Train Decision Tree
    tree = DecisionTreeClassifier(max_depth=max_depth, min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf)
    tree.fit(X_train_scaled, y_train)

    # Get leaf nodes
    train_regions = tree.apply(X_train_scaled)
    test_regions = tree.apply(X_test_scaled)

    # One-hot encode the leaf nodes
    from sklearn.preprocessing import OneHotEncoder
    enc = OneHotEncoder()
    X_train_regions = enc.fit_transform(train_regions.reshape(-1, 1)).toarray()
    X_test_regions = enc.transform(test_regions.reshape(-1, 1)).toarray()

    # Combine features
    X_train_hybrid = np.hstack((X_train_scaled, X_train_regions))
    X_test_hybrid = np.hstack((X_test_scaled, X_test_regions))

    # Train Logistic Regression with class weights
    class_weights = {0: 1.0, 1: np.sum(y_train == 0) / np.sum(y_train == 1)}
    lr = LogisticRegression(C=lr_C, max_iter=1000, class_weight=class_weights)
    lr.fit(X_train_hybrid, y_train)

    # Make predictions with optimized threshold
    y_pred_prob = lr.predict_proba(X_test_hybrid)[:, 1]
    # Find optimal threshold using F1 score
    from sklearn.metrics import f1_score
    thresholds = np.linspace(0.1, 0.9, 50)
    f1_scores = []
    for threshold in thresholds:
        y_pred = (y_pred_prob >= threshold).astype(int)
        f1_scores.append(f1_score(y_test, y_pred))
    optimal_threshold = thresholds[np.argmax(f1_scores)]

    y_pred = (y_pred_prob >= optimal_threshold).astype(int)

    # Evaluate and plot results
    from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, confusion_matrix
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_prob)
    conf_matrix = confusion_matrix(y_test, y_pred)

    # Display Metrics
    st.write("### Model Performance Metrics:")
    st.write(f"**Accuracy:** {accuracy * 100:.2f}%")
    st.write(f"**Precision:** {precision:.4f}")
    st.write(f"**Recall:** {recall:.4f}")
    st.write(f"**F1-Score:** {f1:.4f}")
    st.write(f"**ROC-AUC:** {roc_auc:.4f}")

    # Display Confusion Matrix
    st.write("### Confusion Matrix:")
    fig, ax = plt.subplots()
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                xticklabels=['No Failure', 'Failure'],
                yticklabels=['No Failure', 'Failure'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    st.pyplot(fig)

    # ROC Curve
    st.write("### ROC Curve:")
    from sklearn.metrics import roc_curve
    fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
    fig2, ax2 = plt.subplots()
    ax2.plot(fpr, tpr, label=f'ROC (AUC = {roc_auc:.3f})')
    ax2.plot([0, 1], [0, 1], 'k--')
    ax2.set_xlabel('False Positive Rate')
    ax2.set_ylabel('True Positive Rate')
    ax2.legend()
    st.pyplot(fig2)

# Spare Parts Demand Forecasting
def run_spare_parts_forecasting():
    st.subheader("Spare Parts Demand Forecasting")

    # Upload data
    uploaded_file = st.file_uploader("Upload your inventory data CSV file", type=["csv"])
    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
    else:
        st.info("Using default dataset.")
        data = pd.read_csv("data/inventory.csv")

    # Data preprocessing
    data = data[pd.notnull(data.invoice_line_text)].reset_index(drop=True)
    data = data[data.current_km_reading <= 100000].reset_index(drop=True)

    # Dropping redundant columns
    data = data[['job_card_date', 'vehicle_model', 'invoice_line_text']]

    # Data cleaning (simplified for brevity)
    data['invoice_line_text'] = data['invoice_line_text'].str.replace('BULB ', 'BULB')
    data['invoice_line_text'] = data['invoice_line_text'].str.replace('OVERHUAL', 'OVERHAUL')

    # Dropping rows related to services
    service_related_tokens = [
        'OVERHAUL', 'WELDING', 'SERVICE', 'WORK', 'PUNCHER', 'REBORE',
        'DENT', 'RC CARD', 'TAX', 'ENGINE WORK', 'CHECK', 'LABOUR',
        'CHARGE', 'FEES', 'PAYMENT', 'STICKERS', 'ADJUSTMENT', 'REGISTOR',
        'INSURANCE', 'ADJUSTMENT', 'REMOVAL', 'THREADING', 'CLEANING',
    ]
    data = data[~data['invoice_line_text'].isin(service_related_tokens)].reset_index(drop=True)

    # Renaming columns
    data.rename(columns={"job_card_date": "date", "invoice_line_text": "spare_part"}, inplace=True)
    data['date'] = pd.to_datetime(data['date'], format='%d-%m-%y')

    # Setting date as index
    data_indexed = data.set_index('date')

    # Resampling
    weekly_data_indexed = data_indexed[['spare_part']].resample('W').count()

    # Time Series Decomposition
    from statsmodels.tsa.seasonal import seasonal_decompose
    result = seasonal_decompose(weekly_data_indexed['spare_part'], model='mul', period=4)

    # Plot decomposition
    st.write("### Time Series Decomposition:")
    fig3 = result.plot()
    st.pyplot(fig3)

    # Train-Test Split
    st.sidebar.subheader("Forecasting Parameters")
    split_point = st.sidebar.slider("Number of Weeks for Testing", 4, 52, 16)
    train_data = weekly_data_indexed[:-split_point]
    test_data = weekly_data_indexed[-split_point:]

    # Exponential Smoothing Model
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    fitted_model = ExponentialSmoothing(train_data['spare_part'], trend='mul', seasonal='add', seasonal_periods=26).fit()
    test_predictions = fitted_model.forecast(len(test_data))

    # Plotting
    st.write("### Actual vs Predicted Spare Parts Demand:")
    fig4, ax = plt.subplots(figsize=(12, 6))
    train_data['spare_part'].plot(legend=True, label='TRAIN DATA', ax=ax)
    test_data['spare_part'].plot(legend=True, label='TEST DATA', ax=ax)
    test_predictions.plot(legend=True, label='PREDICTION', ax=ax)
    st.pyplot(fig4)

    # Error Metrics
    from sklearn.metrics import mean_absolute_error, mean_squared_error
    mae_error = mean_absolute_error(test_data['spare_part'], test_predictions)
    mse_error = mean_squared_error(test_data['spare_part'], test_predictions)
    st.write(f"**Mean Absolute Error:** {mae_error:.2f}")
    st.write(f"**Mean Squared Error:** {mse_error:.2f}")

# Car Sales Prediction
def run_car_sales_prediction():
    st.subheader("Car Sales Prediction")

    # Upload data
    uploaded_file = st.file_uploader("Upload your car sales data CSV file", type=["csv"])
    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
    else:
        st.info("Using default dataset.")
        data = pd.read_csv("data/car_sales_data.csv")

    # Preprocess data
    data['date'] = pd.to_datetime(data['date'])
    data['month'] = data['date'].dt.month
    data['year'] = data['date'].dt.year

    # One-hot encoding for car models
    car_dummies = pd.get_dummies(data['car_model'], prefix='car')
    data = pd.concat([data, car_dummies], axis=1)

    # Features and target
    features = ['price', 'marketing_spend', 'festival_season', 'competitor_launches', 'month']
    X = data[features]
    y = data['sales']

    # Parameter adjustments
    st.sidebar.subheader("Prediction Parameters")
    future_months = st.sidebar.slider('Select number of future months to predict:', 1, 12, 3)

    # Train-Test Split
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Linear Regression Model
    from sklearn.linear_model import LinearRegression
    lr = LinearRegression()
    lr.fit(X_train, y_train)

    # Predictions
    y_pred = lr.predict(X_test)

    # Evaluation Metrics
    from sklearn.metrics import mean_squared_error, r2_score
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    st.write(f"**Mean Squared Error:** {mse:.2f}")
    st.write(f"**R² Score:** {r2:.4f}")

    # Plot Actual vs Predicted
    st.write("### Actual vs Predicted Sales:")
    fig5, ax = plt.subplots()
    ax.scatter(y_test, y_pred)
    ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
    ax.set_xlabel('Actual')
    ax.set_ylabel('Predicted')
    st.pyplot(fig5)

    # Future Predictions
    st.write("### Future Sales Predictions:")
    last_features = X.iloc[-1].copy()
    future_predictions = []
    future_dates = []
    for i in range(future_months):
        last_features['month'] = ((last_features['month'] + 1) % 12) or 12
        pred = lr.predict([last_features.values])[0]
        future_predictions.append(pred)
        future_dates.append(pd.to_datetime(f"{int(data['year'].max())}-{int(last_features['month'])}-01"))

    # Create a DataFrame for future predictions
    future_df = pd.DataFrame({
        'Date': future_dates,
        'Predicted Sales': future_predictions
    })

    st.table(future_df)

    # Plot Future Predictions
    st.write("### Sales Forecast:")
    fig6, ax = plt.subplots()
    ax.plot(data['date'], data['sales'], label='Historical Sales')
    ax.plot(future_df['Date'], future_df['Predicted Sales'], label='Predicted Sales', linestyle='--', marker='o')
    ax.set_xlabel('Date')
    ax.set_ylabel('Sales')
    ax.legend()
    st.pyplot(fig6)

# Run the main function
if __name__ == "__main__":
    main()