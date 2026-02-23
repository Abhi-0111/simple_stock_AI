# simple_stock_AI
Predicting stock price of google using Tensorflow

# 🏦 NeuralFinance: Deep Learning for Market Forecasting

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow 2.15](https://img.shields.io/badge/TensorFlow-2.15-orange.svg)](https://tensorflow.org)
[![Kaggle Data](https://img.shields.io/badge/Dataset-S%26P500-green.svg)](https://www.kaggle.com/datasets/camnugent/sandp500)

## 📖 Project Architecture
`NeuralFinance` is a supervised learning pipeline that utilizes a Multi-Layer Perceptron (MLP) architecture built on **TensorFlow** to predict equity prices. While traditional finance uses linear moving averages, this model utilizes non-linear activation functions to capture complex market "regimes."

---

## 🧠 Model Mechanics: How it Thinks
Most beginners treat AI as a "black box." Here is the step-by-step logic of how this model processes a stock price:

### 1. The Sliding Window (Temporal Transformation)
Financial data is a continuous stream. To make it "learnable," we transform it into a **Sliding Window**.
* **Input Window ($X$):** The model looks at the last 5 days of closing prices ($t-5, t-4, t-3, t-2, t-1$).
* **Target Label ($y$):** The model tries to guess the price at time ($t$).



### 2. Feature Scaling (Min-Max Normalization)
Stock prices vary wildly (e.g., Google is ~$140, while Berkshire Hathaway is ~$600,000). Neural networks struggle with large, varied numbers because they cause "Gradient Explosion." 
We scale all data to a range of $[0, 1]$ using:
$$x_{scaled} = \frac{x - x_{min}}{x_{max} - x_{min}}$$

### 3. The Hidden Layers (Pattern Recognition)
Our model consists of two "Dense" layers:
* **Layer 1 (10 Neurons):** Acts as a feature extractor, looking for basic trends (e.g., "is the price consistently rising?").
* **Layer 2 (5 Neurons):** Acts as a refiner, identifying subtler relationships like "mean reversion" (e.g., "the price rose too fast, it might drop now").
* **Activation (ReLU):** We use the Rectified Linear Unit. If a signal is negative or "noise," ReLU turns it to zero. If it's a strong trend, it passes it through.



---

## 📊 Evaluation & Metrics
How do we know if the model is actually "smart" or just lucky?

### Loss Function: Mean Squared Error (MSE)
We use MSE to train the model. It calculates the difference between the **Predicted Price** and the **Actual Price**, squares it (to make all errors positive), and tries to minimize that number.
$$MSE = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2$$

### Accuracy vs. Reality
In Finance, **Accuracy** is a trap. A model can be 99% "accurate" but still lose money if it misses the big market crashes. We evaluate our model based on:
1. **Directional Symmetry:** Did the model correctly predict if the price would go UP or DOWN?
2. **MAE (Mean Absolute Error):** On average, how many dollars off was our prediction?



---

## 🛠️ Tech Stack & Workflow
1. **Data Source:** Automated retrieval via `kagglehub` from the `mlg-ulb/creditcardfraud` and `sandp500` datasets.
2. **Preprocessing:** `Scikit-Learn` for data splitting and scaling.
3. **Engine:** `TensorFlow/Keras` for the neural network construction.
4. **Visualization:** `Matplotlib` for backtesting results.

---

## 🚦 Strategic Challenges (The "Fintech" Reality)
Building a model is easy; making it work in the real world is hard. This project addresses (or acknowledges) three major hurdles:

* **Non-Stationarity:** Stock market rules change. A model trained in 2023 might not work in 2026 because the "math" of the market has shifted.
* **Overfitting:** If the model memorizes the past perfectly, it will fail in the future. We use **Dropout** or small layer sizes to prevent this.
* **The Random Walk Hypothesis:** Many economists believe stock prices are random. Our model assumes there is a "hidden signal" in the noise.

---

## 👨‍💻 How to Use
1. **Install:** `pip install tensorflow pandas scikit-learn kagglehub matplotlib`
2. **Execute:** Run `python model.py`
3. **Customization:** Change the `ticker` variable in the script to predict any S&P 500 company (e.g., `NVDA`, `TSLA`, `MSFT`).

---

## 📜 License & Disclaimer
*This project is for educational purposes. Trading stocks involves significant risk of loss. The author is not responsible for any financial losses incurred from the use of this code.*


