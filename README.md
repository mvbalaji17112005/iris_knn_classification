# 🌸 Iris Flower Classification using KNN

## 📌 Project Overview

This project demonstrates the use of **K-Nearest Neighbors (KNN)** for classifying iris flowers into three species:

* **Setosa**
* **Versicolor**
* **Virginica**

The model achieves ~95% accuracy and is deployed for real-time predictions.

---

## 📂 Project Structure

```
iris-knn-classification/
│── iris_knn.py          # Train & save the KNN model
│── predict_iris.py      # Load model & make real-time predictions
│── iris_knn_model.pkl   # Saved trained KNN model
│── scaler.pkl           # Saved StandardScaler
│── requirements.txt     # Dependencies
│── README.md            # Project documentation
```

---

## ⚙️ Workflow

1. **Load Dataset**

   * Uses the famous Iris dataset (`sklearn.datasets.load_iris`).
   * Features: sepal length, sepal width, petal length, petal width.
   * Target: flower species.

2. **Preprocessing**

   * Dataset split into train (70%) and test (30%).
   * Features standardized using `StandardScaler`.

3. **Model Training**

   * KNN (`k=5`) trained on scaled data.

4. **Evaluation**

   * Accuracy and classification report printed.
   * Achieved ~95% accuracy.

5. **Deployment**

   * Model and scaler saved using `joblib`.
   * `predict_iris.py` script allows real-time predictions.

---

## 🚀 How to Run

### 1. Clone the repository

```bash
git clone https://github.com/your-username/iris-knn-classification.git
cd iris-knn-classification
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Train the model (optional, already provided)

```bash
python iris_knn.py
```

### 4. Run prediction

```bash
python predict_iris.py
```

### Example Output

```
Predicted Class: setosa
```

---

## 📊 Results

* **Accuracy:** ~95%
* **Classification Report:** Shows precision, recall, F1-score for each species.

---

## 💡 Applications

* Botanical research (species classification).
* Education (intro ML project).
* Can be adapted to other classification problems like disease detection, sentiment analysis, etc.

---

## 🛠️ Tech Stack

* Python
* scikit-learn
* NumPy
* Joblib