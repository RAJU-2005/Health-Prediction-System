# 🩺 Health Prediction System Using Machine Learning

## 🧩 Problem Statement

Access to timely and quality healthcare is still a major challenge, especially in remote or under-resourced areas. Many people ignore early symptoms due to a lack of awareness or delay in medical consultation, which can lead to serious complications. There’s a growing need for technology-driven solutions that assist in early disease detection and help bridge the healthcare accessibility gap.

This project aims to solve that by building an intelligent **Health Prediction System** using **machine learning**, where users can input their symptoms and receive a predicted disease. While it doesn’t replace a doctor, it can serve as an initial health advisor, saving time and enabling quicker decisions — especially in areas where medical support is scarce.

---

## 🔍 Overview

This project is a **Health Prediction System** developed using **Python** and **Machine Learning** techniques. The system predicts possible diseases based on symptoms provided by the user. It was designed as a part of an internship project to explore how AI can be applied to real-world healthcare challenges, especially for early detection and accessibility in remote or under-resourced regions.

The system uses a **supervised learning** approach on a dataset of over 130 symptoms and 40+ diseases to predict the most likely diagnosis. The core machine learning algorithms used are:
- 🧠 Decision Tree Classifier
- 🌲 Random Forest Classifier

> ⚠️ Disclaimer: This is an educational project and **not a replacement for medical advice**. Always consult a qualified healthcare provider for diagnosis and treatment.

---

## 📁 Project Structure

- `Health_predication.ipynb` - Main Jupyter Notebook containing code for data preprocessing, model training, and evaluation.
- `README.md` – This file.
- `Training.csv` – Dataset (linked below)

---

## 📊 Dataset Overview

The dataset used for training and testing the model is publicly shared on Google Drive. It contains symptom data for over 130 medical conditions.

🔗 **Download the dataset here**:  
[👉 Click to open dataset folder on Google Drive](https://drive.google.com/drive/folders/1LEHKCgs56Pi2eVEfeLlwqr2CU26B0wb0?usp=drive_link)

### Dataset Details:
- **Features:** 132 binary columns (symptoms like fever, fatigue, headache, etc.)
- **Target:** `prognosis` column (disease name like Dengue, Typhoid, Diabetes, etc.)
- **Format:** CSV (Comma-Separated Values)

### Preprocessing Steps:
- Label Encoding for the `prognosis` column
- Train-Test Split using scikit-learn (70% training, 30% testing)
- Ensured dataset balance and integrity

---

## 🧪 Model Training & Evaluation

Two models were trained and tested:

| Model               | Accuracy | Precision | Recall | F1-Score |
|--------------------|----------|-----------|--------|----------|
| Decision Tree       | ~94%     | Good      | Good   | Good     |
| Random Forest       | ~97–98%  | Excellent | Excellent | Excellent |

Evaluation Metrics used:
- ✅ Accuracy Score
- 📊 Confusion Matrix
- 📋 Classification Report (Precision, Recall, F1)

---

## 🔧 Technologies & Libraries

- Python 3.x
- NumPy, Pandas
- Scikit-learn (`LabelEncoder`, `DecisionTreeClassifier`, `RandomForestClassifier`, `metrics`)

---

## 🚀 Future Enhancements

- 🌐 Web/Mobile App Deployment using **Flask**, **Streamlit**, or **Flutter**
- 🔄 Real-time symptom updates via **medical APIs**
- 🎚️ Include **symptom severity levels** (mild/moderate/severe)
- 🌍 Multilingual Support (Kannada, Hindi, etc.)
- 🔎 Explainable AI using **LIME** or **SHAP**
- 🏥 “Connect to Doctor” or “Nearest Hospital” feature using **GPS/Maps**
- 📈 Real-time health monitoring and alert system

---

## 📚 References

1. **Ahsan et al. (2021)** – ML-Based Disease Diagnosis  
   [Link to paper](https://doi.org/10.3390/healthcare9050528)

2. **Sood et al. (2022)** – Symptom-Based Disease Prediction

3. **Agarwal & Yadav (2023)** – Optimized Classifiers for Symptom-Based Detection

---

## 👨‍💻 Author

**Raju**  
B.Tech (Hons) in Data Science, Vidyashilp University  
Project on Predictive Healthcare  
📍 Bangalore, India  

---

⭐️ If you like this project, give it a **star** and feel free to contribute ideas or enhancements!

