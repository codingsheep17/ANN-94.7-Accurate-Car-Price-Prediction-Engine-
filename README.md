# 🚗 ANN: Deep Learning Car Price Predictor

[![Python](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15-orange.svg)](https://www.tensorflow.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-red.svg)](https://streamlit.io/)
[![Accuracy](https://img.shields.io/badge/R2%20Score-0.947-green.svg)](#)

A high-performance Artificial Neural Network (ANN) designed to estimate the market value of used cars. By leveraging deep learning and rigorous data preprocessing, this model achieves near-human expert pricing accuracy.
---

## 🌟 Key Features
- **Brainpower:** Multi-layer ANN architecture (Dense 64 -> 32 -> 1).
- **Elite Preprocessing:** Handles skewed price distributions using **Log Transformation**.
- **Feature Engineering:** Custom logic to extract `Torque` and `RPM` from raw strings.
- **Interactive UI:** Deployed as a web app using **Streamlit** for real-time predictions.
- **Robust Scaling:** Uses `StandardScaler` to ensure balanced feature weight.
---

## 📈 Performance Metrics
- **R2 Score (Accuracy):** `0.9475` (Explains 94.7% of price variation)
- **Mean Absolute Error (MAE):** `₹101,998` (Solid for multi-lakh car valuations)
- **Loss Function:** Mean Squared Error (MSE) optimized with **Adam**.
---

## 🛠️ Tech Stack
- **Languages:** Python
- **Deep Learning:** TensorFlow, Keras
- **Data Science:** Pandas, NumPy, Scikit-Learn
- **Deployment:** Streamlit, Joblib
