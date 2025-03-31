# 🎯 RNN Sentiment Analysis Project

## 🚀 **Project Title:** Reviews Sentiment Analysis using Recurrent Neural Network  
**Contributors:**  
- Ashish Michael Chauhan (055007)  

---

## 📚 **Project Overview**
This project uses a **Recurrent Neural Network (RNN)** architecture to perform **sentiment analysis** on movie reviews. The goal is to classify reviews as **positive** or **negative** by analyzing the sequential patterns in text data.

---

## 🔥 **Objective**
- To design, implement, and evaluate an RNN model for sentiment classification.  
- To enhance accuracy by applying text preprocessing and sequence modeling.  
- To generalize the model's performance across different review datasets.  

---

## ⚙️ **Tech Stack**
- **Language:** Python  
- **Libraries:**  
  - `pandas` – Data manipulation and preprocessing  
  - `numpy` – Numerical operations  
  - `scikit-learn` – Data splitting and evaluation  
  - `tensorflow` & `keras` – RNN model building and training  
  - `re` – Text cleaning and preprocessing  

---

## 📂 **Project Structure**

---

## 📊 **Dataset Description**
### **Training Data**
- **Source:** IMDB dataset of **50,000** movie reviews.  
- **Columns:**
  - `reviews`: Textual review of the movie.  
  - `sentiment`: Binary labels (1 for positive, 0 for negative).  
- **Train-Test Split:**  
  - **80%** for training  
  - **20%** for testing  

### **Testing Data**
- **Source:** Manually scraped **Metacritic reviews**.  
- **Columns:**  
  - `movie name`: The title of the movie.  
  - `rating`: Rating assigned to the movie.  
  - `reviews`: Review text.  
  - `sentiment`: Binary label (positive or negative).  

---

## 🔥 **Model Architecture**
The model uses the following layers:

### **1. Embedding Layer**
- **Input dimension:** 20,000 (vocabulary size)  
- **Output dimension:** 128 (word embedding size)  
- **Input length:** 400 (maximum sequence length)  

### **2. Recurrent Layer**
- **Type:** SimpleRNN  
- **Units:** 64  
- **Activation:** Tanh  
- **Regularization:** Dropout (0.2)  

### **3. Fully Connected Layer**
- **Type:** Dense layer  
- **Neurons:** 1  
- **Activation:** Sigmoid (for binary classification)  

---

## ⚙️ **Model Compilation and Training**
- **Loss Function:** Binary Crossentropy  
- **Optimizer:** Adam (`learning rate = 0.001`)  
- **Batch Size:** 32  
- **Epochs:** 15 (with early stopping)  
- **Validation Split:** 20%  

---

## 📈 **Model Evaluation**
- **Training Accuracy:** ~87% after 10 epochs  
- **Validation Accuracy:** ~79%  
- **Test Accuracy (Metacritic data):** ~78%  

---

## 💡 **Managerial Insights**
### ✅ **Model Effectiveness**
- The RNN model achieves **good accuracy** on the IMDB dataset.  
- However, it generalizes **less effectively** on Metacritic data due to different writing styles.  

### 🔥 **Improvement Areas**
- Switch to **LSTM/GRU** for better sequential modeling.  
- Use **TF-IDF** and improved text preprocessing techniques.  
- Fine-tune the model with more diverse datasets.  

---

## 🚀 **Business Applications**
- **Customer Sentiment Monitoring:**  
    - Analyze customer reviews to gauge public opinion.  
- **Brand Reputation Analysis:**  
    - Identify sentiment trends to manage PR crises.  
- **Automated Review Filtering:**  
    - Filter spam and fake reviews using automated sentiment analysis.  

---

## ✅ **Conclusion & Recommendations**
### **Immediate Steps:**
- Improve text preprocessing (e.g., stemming, lemmatization).  
- Fine-tune the model with **additional datasets**.  
- Experiment with **LSTM or GRU** for better results.  

### **Long-Term Strategy:**
- Deploy the model for **real-time sentiment tracking**.  
- Expand training data with reviews from **multiple platforms**.  
- Implement **A/B testing** to validate model improvements.  

---

## 📌 **References**
- IMDB Dataset: [IMDB Reviews](https://drive.google.com/file/d/1oEAYu8u_gmBYBjhZpZbc7T_VxPoHPU4L/view?usp=drive_link)  
- Metacritic Dataset: [Metacritic Reviews](https://drive.google.com/file/d/1d5ee3bWi2__Cn3856bBqvEOBoF3KIsDy/view?usp=sharing)  
- Model Architecture: [Model Link](https://drive.google.com/file/d/1K_uKqEs6jGqtuP8HaDA3qk3hvMlujFqE/view?usp=sharing)  

---
## Author:
**Ashish Michael Chauhan**  
