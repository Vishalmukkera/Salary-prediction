# 💼 Salary Prediction Web App

This project predicts a user's salary based on their **country**, **race**, **education level**, and **years of experience** using a trained machine learning regression model. The application is built with **Streamlit** for an interactive web interface and **Scikit-learn** for machine learning.

---

## 📌 Table of Contents

- [🔍 Overview](#-overview)
- [📊 Dataset](#-dataset)
- [🧠 Model and Approach](#-model-and-approach)
- [🌐 Streamlit Application](#-streamlit-application)


---

## 🔍 Overview

This salary prediction system uses a **regression model** to estimate salary based on demographic factors. It helps in understanding how different factors like education, geography, and ethnicity might influence salary trends.

Key components:
- Data preprocessing and encoding
- Model training with Scikit-learn
- Salary prediction using a `.pkl` model file
- Streamlit-based frontend for easy user interaction

---

## 📊 Dataset

**Filename**: `Salary_Data_Based_country_and_race.csv`

### Features Used:
- `Country`
- `Race`
- `Education Level`
- `Years of Experience`

### Target:
- `Salary` (in USD)

The dataset is cleaned and encoded before being used for model training.

---

## 🧠 Model and Approach

- **Model Used**: `Linear Regression`  
- **Libraries**: Scikit-learn, Pandas, NumPy  
- **Preprocessing**: 
  - Label Encoding for categorical variables
  - Train-test split (80-20)
  - Model evaluation using RMSE and R² score

The trained model is saved as `SalaryModel.pkl` using Pickle.

---

## 🌐 Streamlit Application

The application provides a simple UI for users to:
- Select their **Country**, **Race**, and **Education Level**
- Input their **Years of Experience**
- Get an instant **Salary Prediction**

### Run the App

```bash
streamlit run app.py
