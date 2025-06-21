# 🩺 Health Prediction System Using Machine Learning

## 🔍 Overview

This project is a **Health Prediction System** developed using **Python** and **Machine Learning** techniques. The system predicts possible diseases based on symptoms provided by the user. It was designed as a part of an internship project to explore how AI can be applied to real-world healthcare challenges, especially for early detection and accessibility in remote or under-resourced regions.

The system uses a **supervised learning** approach on a dataset of over 130 symptoms and 40+ diseases to predict the most likely diagnosis. The core machine learning algorithms used are:
- 🧠 Decision Tree Classifier
- 🌲 Random Forest Classifier

> ⚠️ Disclaimer: This is an educational project and **not a replacement for medical advice**. Always consult a qualified healthcare provider for diagnosis and treatment.

---

## 📁 Project Structure

- `Health_predication.ipynb` - Main Jupyter Notebook containing code for data preprocessing, model training, and evaluation.
- `Training.csv` *(dataset not included here)* – Symptom-disease dataset used for model training and testing.
- `README.md` – This file.

---

## 📊 Dataset Overview

- **Features:** 132 binary columns representing symptoms (0 - Absent, 1 - Present)
- **Target:** One multiclass categorical column (`prognosis`) representing disease names
- **Total Diseases Predicted:** ~40

### Preprocessing Steps:
- Label Encoding for target column
- Train-Test Split (70%-30%)
- Ensured class balance for unbiased evaluation

---

## 🧪 Model Training & Evaluation

Two models were trained and tested:

| Model               | Accuracy | Precision | Recall | F1-Score |
|--------------------|----------|-----------|--------|----------|
| Decision Tree       | ~94%     | Good      | Good   | Good     |
| Random Forest       | ~97-98%  | Excellent | Excellent | Excellent |

Evaluation Metrics used:
- ✅ Accuracy Score
- 📊 Confusion Matrix
- 📋 Classification Report (Precision, Recall, F1)

---

## 🔧 Technologies & Libraries

- Python 3.x
- NumPy, Pandas
- Scikit-learn (LabelEncoder, DecisionTree, RandomForest, metrics)

---

## 🚀 Future Enhancements

- Deploying the model using **Flask/Streamlit** as a web/mobile app
- Real-time symptom updates via **medical APIs**
- Add **severity-based scoring** (mild/moderate/severe symptoms)
- Support for **regional languages** (Kannada, Hindi, etc.)
- Integration with **Doctor Booking** and **Nearby Hospital Suggestions**
- Explainable AI (SHAP, LIME) for transparency in predictions

---

## 📚 References

1. Ahsan et al. (2021) - ML-Based Disease Diagnosis: [Healthcare Journal](https://doi.org/10.3390/healthcare9050528)
2. Sood et al. (2022) - Symptom-Based Disease Prediction
3. Agarwal & Yadav (2023) - Optimized Classifiers for Disease Detection

---

## 👨‍💻 Author

**Raju**  
B.Tech (Hons) in Data Science, Vidyashilp University  
Internship Project on Predictive Healthcare  
📍 Bangalore, India  

---

⭐️ If you like this project, give it a **star** and feel free to contribute ideas or enhancements!


