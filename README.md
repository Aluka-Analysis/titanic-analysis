# 🚢 Titanic Survival Prediction — Machine Learning Web App

![Python](https://img.shields.io/badge/Python-3.13-blue?style=flat-square&logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-Deployed-red?style=flat-square&logo=streamlit)
![ML](https://img.shields.io/badge/Machine%20Learning-Logistic%20Regression%20%7C%20Random%20Forest-green?style=flat-square)
![Status](https://img.shields.io/badge/Status-Live-brightgreen?style=flat-square)

> **A complete end-to-end Machine Learning web application** that predicts whether a passenger would have survived the Titanic disaster, built and deployed by **Aluka Precious Oluchukwu**.

🔗 **[View Live App](https://precious-titanic-analysis-gfhsuxsvmzuhbkfwixddpe.streamlit.app)**


##  Project Overview

This project is a full end-to-end machine learning application built on the famous Titanic dataset. The app allows users to input passenger characteristics — such as class, gender, age, and fare — and receive a real-time prediction on whether that passenger would have survived the disaster, along with a survival probability score.

The project covers the entire machine learning pipeline from raw data to a live deployed web application, including data cleaning, exploratory data analysis, feature engineering, model training, model comparison, and cloud deployment.


##  Live Application

The app is deployed and fully accessible at:

**https://precious-titanic-analysis-gfhsuxsvmzuhbkfwixddpe.streamlit.app**

The application has five main sections accessible via the navigation sidebar:

- **Home** — An introduction to the project, its purpose, and how to use it.
- **Prediction** — Users fill in passenger details and the model returns a prediction and survival probability in real time.
- **Dashboard** — Interactive charts exploring survival distributions by passenger class, gender, age and other titanic features.
- **Dataset Explorer** — Allows users to explore the cleaned dataset directly within the app.
- **About** - Users get more insight about the author and what the app entails.

---

##  Machine Learning Models

Two classification models were trained and compared for this project.

- **Logistic Regression** — Used as the baseline model. Interpretable and well-suited to binary classification problems.
- **Random Forest Classifier** — A more powerful ensemble model that builds multiple decision trees and combines their outputs for higher accuracy.

Both models were evaluated using accuracy score, confusion matrix, and ROC curve analysis. The best-performing model Logistic Regression with an accuracy of **80%** was saved for use in the live prediction app.


##  Project Structure
```
titanic-analysis/
│
├── app.py                        # Main Streamlit application file
├── requirements.txt              # Python dependencies for deployment
├── runtime.txt                   # Python runtime version specification
├── README.md                     # Project documentation
│
├── data/
│   ├── raw/
│   │   ├── train.csv             # Original Titanic training dataset
│   │   └── test.csv              # Original Titanic test dataset
│   └── processed/
│       ├── Cleaned Titanic.csv
│       └── Final Cleaned Titanic ML.csv
│
├── notebooks/
│   └── 01 titanic ml.ipynb       # Full ML pipeline notebook
│
└── Outputs/
    └── Visualization/
        ├── models/
        │   ├── titanic_Logistic_regression_model.pkl
        │   ├── titanic_Random_Forest_model.pkl
        │   └── titanic_scaler.pkl
        └── figures/
            ├── Correlation_Heatmap.png
            ├── Logistic_Regression_Confusion_Matrix.png
            ├── Random_Forest_Confusion_Matrix.png
            ├── Model_Accuracy_Comparison.png
            └── Survival_By_Class.png
```

##  Technologies Used

| Category | Tools |
|---|---|
| Language | Python 3.13 |
| Data Manipulation | Pandas, NumPy |
| Machine Learning | Scikit-learn |
| Visualisation | Matplotlib, Seaborn, Plotly, Altair |
| Web App Framework | Streamlit |
| Model Serialisation | Joblib |
| Deployment | Streamlit Cloud |
| Version Control | Git & GitHub |

##  Running the App Locally

**Step 1 — Clone the repository:**
```bash
git clone https://github.com/Aluka-Analysis/titanic-analysis.git
cd titanic-analysis
```

**Step 2 — Install dependencies:**
```bash
pip install -r requirements.txt
```

**Step 3 — Run the app:**
```bash
streamlit run app.py
```

**Step 4** — Open your browser at `http://localhost:8501`


##  Key Insights from the Data

- **Gender** was the strongest predictor of survival — female passengers survived at a significantly higher rate, reflecting the "women and children first" protocol.
- **Passenger class** was the second most important factor — first-class passengers survived at a much higher rate than third-class passengers.
- **Age** played a nuanced role — younger children had higher survival rates while middle-aged men had the lowest.
- **Fare** was correlated with survival largely because it served as a proxy for passenger class.



##  Acknowledgements

Special gratitude to the **Incubator Hub** bootcamp on YouTube and the facilitators **Isreal** and **Ezekiel**, whose teaching gave me my first real understanding of data analysis and data science. You cannot build a house without a foundation, they were mine.

To my eldest brother **Victor Aluka**, who saw potential in an idle laptop and pointed me toward this field, that one conversation changed my trajectory.

The Titanic dataset is sourced from **[Kaggle](https://www.kaggle.com/competitions/titanic)**.


##  About the Author

**Aluka Precious Oluchukwu** is a Data Analyst and a aspiring Machine Learning Engineer with a background in Philosophy from the University of Port Harcourt, Nigeria. He is currently building in public, one project at a time.

## Connect With Me

🔗 [GitHub](https://github.com/Aluka-Analysis) | 💼 [LinkedIn](https://www.linkedin.com/in/aluka-precious-b222a2356) | 🌐 [Live App](https://precious-titanic-analysis-gfhsuxsvmzuhbkfwixddpe.streamlit.app)


*Built with curiosity, persistence, and a Philosophy degree. App predicts survival instantly!* 


## If you found this project useful, please give it a star!
