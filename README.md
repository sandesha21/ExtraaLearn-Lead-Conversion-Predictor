# ExtraaLearn-Lead-Conversion-Predictor

A machine learning project to predict lead conversion probability for an EdTech startup using customer interaction data and behavioral patterns.

## 🎯 Project Overview

This project develops a predictive model to help ExtraaLearn identify which leads are most likely to convert to paid customers, enabling optimized resource allocation and improved conversion rates.

**Key Results:**
- 🎯 **85% Recall** achieved with Random Forest model
- 📊 **83-86% Accuracy** on test data
- 🔍 **4,612 leads** analyzed across 15 features
- 💡 **Clear conversion patterns** identified for business strategy

## 📁 Repository Structure

```
├── README.md                           # Project overview and setup guide
├── PROJECT_DESCRIPTION.md              # Detailed project documentation
├── ExtraaLearn.csv                     # Dataset with lead information
└── extraalearn_lead_conversion_prediction.ipynb # Complete analysis notebook
```

## 🚀 Quick Start

### Prerequisites
```bash
Python 3.7+
Jupyter Notebook
```

### Required Libraries
```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

### Running the Analysis
1. Clone this repository
2. Open `extraalearn_lead_conversion_prediction.ipynb`
3. Run all cells to reproduce the analysis

## 📊 Dataset Overview

**Source:** ExtraaLearn lead interaction data  
**Size:** 4,612 records × 15 features  
**Target:** Binary classification (converted/not converted)

### Key Features:
- **Demographics:** Age, occupation
- **Behavior:** Website visits, time spent, page views
- **Interactions:** First contact method, last activity
- **Marketing:** Print media, digital media, referrals

## 🔬 Methodology

### 1. Data Exploration
- Comprehensive EDA with visualizations
- Missing value analysis (no missing data found)
- Feature distribution and correlation analysis

### 2. Model Development
- **Decision Tree Classifier** (baseline)
- **Random Forest Classifier** (best performer)
- Hyperparameter tuning with GridSearchCV

### 3. Evaluation Strategy
- **Primary Metric:** Recall (minimize false negatives)
- **Secondary Metrics:** Precision, F1-score, Accuracy
- Train/test split for unbiased evaluation

## 📈 Key Findings

### Top Conversion Predictors:
1. **Time spent on website** (most important)
2. **First interaction channel** (website vs mobile)
3. **Profile completion level**
4. **Age** (45-55 age group converts best)
5. **Last activity type**

### Customer Insights:
- 🌐 **Website users** convert better than mobile app users
- ⏱️ **Higher engagement time** = higher conversion probability
- 👥 **Referral leads** have significantly higher conversion rates
- 📝 **Complete profiles** strongly correlate with conversion
- 🎂 **Age 45-55** demographic shows peak conversion rates

## 💼 Business Impact

### Recommendations:
1. **Prioritize high-engagement leads** (long website sessions)
2. **Focus on referral programs** for quality lead generation
3. **Target 45-55 age demographic** with specialized campaigns
4. **Optimize website experience** over mobile app development
5. **Incentivize profile completion** to increase conversion likelihood

### Expected Outcomes:
- 📉 **Reduced customer acquisition costs**
- 📈 **Improved conversion rates**
- 🎯 **Better resource allocation**
- ⚡ **Faster lead qualification process**

## 🛠️ Technical Details

### Model Performance:
```
Random Forest Classifier (Tuned)
├── Recall: 85%
├── Precision: 68%
├── F1-Score: 76%
└── Accuracy: 83%
```

### Technologies Used:
- **Python** - Core programming language
- **Pandas** - Data manipulation and analysis
- **Scikit-learn** - Machine learning algorithms
- **Matplotlib/Seaborn** - Data visualization
- **Jupyter Notebook** - Interactive development environment

## 📋 Usage Examples

### Load and Explore Data:
```python
import pandas as pd
df = pd.read_csv('ExtraaLearn.csv')
print(df.head())
print(df.info())
```

### Quick Model Training:
```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Prepare data (see notebook for full preprocessing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Predict
predictions = rf_model.predict(X_test)
```

## 📚 Documentation

- **[PROJECT_DESCRIPTION.md](PROJECT_DESCRIPTION.md)** - Comprehensive project documentation
- **Jupyter Notebook** - Step-by-step analysis with code and visualizations
- **HTML Report** - Static version of the analysis for easy viewing

## 🤝 Contributing

This is an educational project completed as part of the MIT Applied Data Science Program. For questions or suggestions:

1. Open an issue for discussion
2. Fork the repository for experimentation
3. Follow standard data science best practices

## 📄 License

This project is for educational purposes as part of the Great Learning curriculum.

## 👨‍💻 Author

**Sandesh Badwaik**  
Project for MIT Applied Data Science Program  

---

## 🔗 Quick Links

- [📊 View Analysis Notebook](extraalearn_lead_conversion_prediction.ipynb)
- [📋 Detailed Documentation](PROJECT_DESCRIPTION.md)
- [📈 Dataset](ExtraaLearn.csv)

**⭐ Star this repository if you found it helpful!**