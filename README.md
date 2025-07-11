# MotorMinds - Automotive Smart Insights Platform

MotorMinds is an advanced Automotive Smart Insights Platform that leverages machine learning and data analytics to provide accurate forecasts and predictions for the automotive industry. The platform helps automotive companies optimize inventory, improve customer satisfaction, and increase operational efficiency.

## 🚗 Features

### 1. **Predictive Maintenance**
- Anticipate maintenance needs and potential component failures before they occur
- Uses hybrid Decision Tree + Logistic Regression model for high accuracy
- Prevents unexpected breakdowns and extends vehicle lifespan
- Real-time performance metrics and ROC curve analysis

### 2. **Spare Parts Demand Forecasting**
- Accurately forecast spare parts demand to optimize inventory levels
- Time series decomposition and analysis
- Exponential Smoothing model with trend and seasonal components
- Reduces holding costs and prevents stockouts

### 3. **Car Sales Prediction**
- Predict future vehicle sales using machine learning models
- Multiple feature analysis including price, marketing spend, seasonal trends
- Future sales forecasting with customizable time horizons
- Interactive visualizations for trend analysis

## 🛠️ Technology Stack

- **Backend**: Python 3.8+
- **Frontend**: Streamlit
- **Data Processing**: Pandas, NumPy
- **Machine Learning**: Scikit-learn
- **Time Series Analysis**: Statsmodels
- **Visualization**: Matplotlib, Seaborn

## 📦 Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Setup Instructions

1. **Clone the repository**
   ```bash
   git clone https://github.com/yashsinghal14/motor-minds.git
   cd motor-minds
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv myenv
   ```

3. **Activate the virtual environment**
   - On Windows:
     ```bash
     myenv\Scripts\activate
     ```
   - On macOS/Linux:
     ```bash
     source myenv/bin/activate
     ```

4. **Install required packages**
   ```bash
   pip install -r requirements.txt
   ```

## 🚀 Usage

### 🌐 Live Demo
Try the live application: **[MotorMinds Live Demo](https://motorminds.streamlit.app/)**

### 💻 Local Development

1. **Start the application**
   ```bash
   streamlit run app.py
   ```

2. **Access the web interface**
   - Open your browser and navigate to `http://localhost:8501`

3. **Navigate through the platform**
   - Use the sidebar menu to switch between different analysis modules
   - Upload your own datasets or use the provided sample data
   - Adjust model parameters using the sidebar controls
   - View interactive visualizations and predictions

## 📊 Data Requirements

### Predictive Maintenance Data (`predictive_maintenance_data.csv`)
Required columns:
- `ComponentID`: Unique identifier for components
- `Temperature`: Operating temperature
- `Pressure`: Operating pressure
- `Vibration`: Vibration measurements
- `UsageHours`: Hours of component usage
- `DaysSinceMaintenance`: Days since last maintenance
- `ComponentHealth`: Health score of component
- `Failure`: Binary target variable (0/1)

### Inventory Data (`inventory.csv`)
Required columns:
- `job_card_date`: Date of service (DD-MM-YY format)
- `vehicle_model`: Model of the vehicle
- `invoice_line_text`: Description of spare parts
- `current_km_reading`: Current kilometer reading

### Car Sales Data (`car_sales_data.csv`)
Required columns:
- `date`: Date of sales record
- `car_model`: Model of the car
- `sales`: Number of units sold
- `price`: Price of the vehicle
- `marketing_spend`: Marketing expenditure
- `festival_season`: Binary indicator for festival season
- `competitor_launches`: Number of competitor launches

## 🔧 Model Configuration

### Predictive Maintenance
- **Decision Tree Parameters**: Max depth, min samples split, min samples leaf
- **Logistic Regression**: Regularization strength (C parameter)
- **Threshold Optimization**: Automatic F1-score based threshold selection

### Spare Parts Forecasting
- **Time Series Decomposition**: Multiplicative model with customizable seasonality
- **Exponential Smoothing**: Trend and seasonal components
- **Testing Period**: Configurable test size (4-52 weeks)

### Car Sales Prediction
- **Feature Engineering**: One-hot encoding for categorical variables
- **Linear Regression**: Multiple feature analysis
- **Forecasting**: Customizable prediction horizon (1-12 months)

## 📈 Performance Metrics

The platform provides comprehensive performance evaluation:

- **Classification Metrics**: Accuracy, Precision, Recall, F1-Score, ROC-AUC
- **Regression Metrics**: Mean Squared Error, R² Score
- **Time Series Metrics**: Mean Absolute Error, Mean Squared Error
- **Visualization**: Confusion matrices, ROC curves, forecast plots

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📞 Support

For support and questions:
- Create an issue in the repository
- Contact the development team

## 🔮 Future Enhancements

- [ ] Real-time data integration
- [ ] Advanced deep learning models
- [ ] Multi-brand vehicle support
- [ ] API endpoints for external integration
- [ ] Dashboard customization
- [ ] Mobile responsive design
- [ ] User authentication and role management

## 📋 Changelog

### Version 1.0.0
- Initial release with core functionality
- Predictive maintenance analysis
- Spare parts demand forecasting
- Car sales prediction
- Interactive Streamlit interface

---

**MotorMinds** - Driving the future of automotive analytics with intelligent insights.
