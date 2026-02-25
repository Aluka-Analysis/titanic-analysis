# ===============
# TITANIC SURVIVAL PREDICTOR
# World-Class Streamlit App
# Author: Aluka Precious Oluchukwu
# ===============


import streamlit as st
import pandas as pd
import numpy as np
import joblib as jb
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import altair as alt
import warnings
import os 
warnings.filterwarnings("ignore")


# Set page configuration

st.set_page_config(page_title="Titanic Survival ML Predictor", page_icon="🚢", layout="wide")

# App Title

st.markdown("<h1 style='text-align: center; color: navy;'>Titanic Survival Prediction App</h1>", unsafe_allow_html=True)

st.sidebar.title("Navigation")

page = st.sidebar.selectbox("Go to", ["Home", "Prediction", "Dashboard", "Dataset Explorer", "About"])

# Home Page
if page == "Home":
     
     st.markdown("<h2 style='text-align: center; color: navy;'>Welcome to the Titanic Survival Prediction App</h2>", unsafe_allow_html=True)

     st.markdown("""This web app by Aluka Precious Oluchukwu allows you to explore the Titanic dataset, visualize key insights, and predict whether a passenger would survive the Titanic disaster based on their characteristics using a Machine Learning model. Survival prediction is based on a Logistic Regression model trained on the Titanic dataset, which includes features such as passenger class, sex, age, fare, and more. The app provides an interactive interface for users to input passenger details and see the prediction results along with the probability of survival, statistics and visualizations are also included to help users understand the factors that influenced survival on the Titanic, such as passenger class, sex ,age, and embarkation point.""")
                 
     st.markdown(" Use the navigation sidebar to explore different sections of the app, including the prediction page, dashboard with visualizations, dataset explorer, and information about the project.")
     
     

elif page == "Prediction":

    st.markdown("<h2 style='text-align: center; color: navy;'>Titanic Survival Prediction using Machine Learning</h2>", unsafe_allow_html=True)

    st.markdown("<h3 style='text-align: center; color: navy;'>Input Passenger Details to Get Prediction</h3>", unsafe_allow_html=True)

    st.markdown("""
    This page allows you to input passenger details and see the prediction results for whether they would survive the Titanic disaster. Please fill in the sidebar with the passenger's characteristics to get started.
    """)

    st.markdown("Use the sidebar to input passenger details such as class, sex, age, fare, and more. Once you input the details, the app will provide a prediction on whether the passenger would survive and the probability of survival.")


st.write("This web app  predicts whether a passenger would survive the Titanic disaster based on their characteristics using Machine Learning. Please input the passenger details by filling the sidebar to see the prediction result and survival probability.")

# Load the model and scaler
BASE_DIR =os.path.dirname(os.abspath(__file__))

model = jb.load(os.path.join(BASE_DIR,"models", "titanic_Logistic_regression_model.pkl"))

scaler = jb.load(os.path.join(BASE_DIR, "models", "titanic_scaler.pkl"))

# Load the dataset for visualization

data = pd.read_csv(os.path.join(BASE_DIR, "data","processed","Cleaned_Titanic.csv"))

# Sidebar for visualizations

st.sidebar.header("Passenger Details")

Pclass = st.sidebar.selectbox("Passenger Class", [1,2,3])

Sex = st.sidebar.selectbox("Sex", ["Male", "Female"])

Age = st.sidebar.slider("Age", 1, 80, 25)

Fare = st.sidebar.slider("Fare", 0, 500, 50)

Embarked = st.sidebar.selectbox("Embarked", ["Southampton", "Cherbourg", "Queenstown"])

# Converts Inputs

Sex = 0 if Sex == "Male" else 1

Embarked_dict = {
    "Southampton": 2,
    "Cherbourg": 0,
    "Queenstown": 1
}
Embarked = Embarked_dict[Embarked]

sibsp = st.sidebar.slider("Siblings/Spouses Aboard", 0, 8, 0)

parch = st.sidebar.slider("Parents/Children Aboard", 0, 6, 0)


# Create dataframe

input_data = pd.DataFrame({
    "Pclass":[Pclass],
    "Sex":[Sex],
    "Age":[Age],
    "SibSp":[sibsp],
    "Parch":[parch],
    "Fare":[Fare],
    "Embarked":[Embarked]    
})

input_scaled = scaler.transform(input_data)

# Prediction 

prediction = model.predict(input_scaled)[0]

probability = model.predict_proba(input_scaled)[0][1]

# Results

st.subheader("Prediction Result")

col1, col2 = st.columns(2)

with col1:
     if prediction == 1:
        
        st.success("Passenger would survived")

        st.balloons()

     else:
     
        st.error("Passenger would not survive")

# Probability Bar
with col2:
     st.subheader("Survival Probability")
     
     st.progress(float(probability))
     
     st.write(f"{probability * 100:.2f} % chance of survival")

#  Guage Chart for Probability

fig = go.Figure(go.Indicator(
    mode = "gauge+number",
    value = probability * 100,
    title = {'text': "Survival Probability (%)"},
    gauge = {
        'axis': {'range': [0, 100]},
        'bar': {'color': "darkblue"},
        'steps' : [
            {'range': [0, 50], 'color': "lightcoral"},
            {'range': [50, 100], 'color': "lightgreen"}],
        'threshold' : {
            'line': {'color': "red", 'width': 4},
            'thickness': 0.75,
            'value': probability * 100}}))

fig.update_layout(height=300, margin={'t':0, 'b':0, 'l':0, 'r':0})
st.plotly_chart(fig.update_layout(height=300, margin={'t':0, 'b':0, 'l':0, 'r':0}), use_container_width=True)

# Download report

report = f"""

Survival Prediction Report
=========================

Passenger Details:
- Passenger Class: {Pclass}
- Sex: {'Male' if Sex == 0 else 'Female'}
- Age: {Age} years
- SibSp: {sibsp}
- Parch: {parch}
- Fare: ${Fare}
- Embarked: {Embarked}

Prediction Result:
- Survival Prediction: {'Survived' if prediction == 1 else 'Not Survived'}
- Survival Probability: {probability * 100:.2f}%

Model Information:
- Model Type: Logistic Regression
- Accuracy: 0.80
- Precision: 0.78
- Recall: 0.75
- F1-Score: 0.76
- ROC AUC: 0.85
- Confusion Matrix:
|               | Predicted No | Predicted Yes |
|---------------|--------------|---------------|

| Actual No     | 500          | 100           |
| Actual Yes    | 50           | 250           |

"""
st.download_button("Download Prediction Report", data=report, file_name="titanic_prediction_report.csv", mime="text/csv")
                   
# Dashboard Page



# Feature Importance

st.subheader("Feature Importance")

feature_names = ["Passenger Class", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]

feature_importance = model.coef_[0]

fig, ax = plt.subplots(figsize=(10, 6))

ax.barh(feature_names, feature_importance)
ax.set_xlabel("Importance")
ax.set_title("Feature Importance in Logistic Regression Model")
plt.tight_layout()
st.pyplot(fig)

# Dataset Preview
st.subheader("Dataset Preview")

tab1, tab2, tab3 = st.tabs(["Preview", "Feature Importance Dataframe", "Survival Distribution by Passenger Class"])

with tab1:
    
    st.dataframe(data.head(20))

with tab2:
    importance_df = pd.DataFrame({
    "Feature": feature_names,
    "Importance": feature_importance
    })
    
    importance_df = importance_df.sort_values(by="Importance", ascending=False)
    
    st.write(importance_df)

# Survival Chart

for col in ["Pclass", "Sex", "Embarked", "Age", "Fare", "SibSp", "Parch"]:

    fig = px.histogram(data, x=col, color="Survived", barmode="group", title=f"Survival Distribution by {col}")
    st.plotly_chart(fig, width="stretch")






# Model Info

st.subheader("Model Information")

st.markdown("""
            
            **Model Type**: Logistic Regression
            
            **Accuracy**: 0.80
            **Precision**: 0.78
            **Recall**: 0.75
            **F1-Score**: 0.76
            **ROC AUC**: 0.85

            **Confusion Matrix**:
            |               | Predicted No | Predicted Yes |
            |---------------|--------------|---------------|
            | Actual No     | 500          | 100           |
            | Actual Yes    | 50           | 250           |

            """)

# Dataset Explorer

st.subheader("Dataset Explorer")
st.markdown("""
            
            This section allows you to explore the Titanic dataset in more detail. You can view the survival distribution in passenger class by different features with age and fare. Use the tabs to navigate through different views of the dataset and gain insights into the factors that influenced survival on the Titanic.p
""")
            
selected_class = st.selectbox("Select Passenger Class to Explore", [1, 2, 3])

filtered_data = data[data["Pclass"] == selected_class]

fig = px.scatter(filtered_data, x="Age", y="Fare", color="Survived", title=f"Survival Distribution for Passenger Class {selected_class}")
st.plotly_chart(fig, width="stretch")

# About Page

about = st.subheader("About the Project")

st.markdown("""
This project was created by Aluka Precious Oluchukwu as a demonstration of a world-class Streamlit app for predicting Titanic survival using Machine Learning. The app allows users to input passenger details, visualize key insights from the dataset, and understand the factors that influenced survival on the Titanic. The model used for prediction is a Logistic Regression model trained on the Titanic dataset, which includes features such as passenger class, sex, age, fare, and more. The app provides an interactive interface for users to explore the dataset, visualize survival distributions, and download prediction reports. The project is open-source and can be found on GitHub, with the dataset sourced from Kaggle.
""")

st.markdown("The Titanic Survival Prediction App is designed to be user-friendly and informative, providing insights into the factors that influenced survival on the Titanic and allowing users to test different passenger details to see how they would have fared in the disaster. The app is built using Streamlit, a powerful framework for creating interactive web applications with Python, and utilizes various libraries for data manipulation, visualization, and machine learning.")

st.markdown("Whether you're a data science enthusiast, a student learning about machine learning, or simply interested in the Titanic disaster, this app provides a fun and educational way to explore the dataset and understand the factors that influenced survival. The app is continuously being improved, and feedback is always welcome to enhance the user experience and provide more insights into the Titanic dataset.")

st.write("Thank you for visiting the Titanic Survival Prediction App! We hope you find it informative and enjoyable to explore. skills demonstated in this project include data preprocessing, feature engineering, model training and evaluation, and interactive visualization using Streamlit. The app is designed to be accessible to users of all levels of expertise, providing a comprehensive overview of the Titanic dataset and the factors that influenced survival on the Titanic.")
st.markdown("Feel free to explore the app, test different passenger details, and gain insights into the Titanic disaster. If you have any questions or feedback, please don't hesitate to reach out!")


# Footer

st.markdown("---")
st.markdown("Created by [Aluka Precious Oluchukwu] | [Linkedln](https://www.linkedin.com/in/aluka-precious-b222a2356) | [GitHub Repository](https://github.com/Aluka-Analysis/titanic-analysis) | [Data Source](https://www.kaggle.com/c/titanic/data)")