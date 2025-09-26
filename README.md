## Iris Classification 

**1. Introduction**
“My project is an **Iris flower classification system** using the **K-Nearest Neighbors (KNN) algorithm**. The goal was to classify iris flowers into three species — *setosa, versicolor, and virginica* — based on their sepal and petal measurements.”

---

**2. Workflow**

**Step 1: Data Loading**

* I used the **Iris dataset** (a standard dataset with 150 samples and 4 features: sepal length, sepal width, petal length, petal width).
* Each sample is labeled as one of the 3 flower species.

**Step 2: Train-Test Split**

* I split the dataset into **training (70%) and testing (30%)** to evaluate performance properly.

**Step 3: Preprocessing (Scaling)**

* Since KNN is a **distance-based algorithm**, I applied **StandardScaler** to normalize the feature values.
* This ensures all features (like sepal length vs petal width) contribute equally.

**Step 4: Model Training (KNN)**

* I used **KNN with k = 5**.
* The algorithm classifies a flower based on the **majority class among its 5 nearest neighbors** in the dataset.

**Step 5: Model Evaluation**

* I tested the model on the test set and achieved **~95% accuracy**.
* I also generated a **classification report** (precision, recall, F1-score) for each flower class.

**Step 6: Deployment**

* I saved the trained model and scaler using **joblib**.
* Then I created a separate script (`predict_iris.py`) where we can input real-time flower measurements and get the predicted class instantly.
* Example: Input `[5.1, 3.5, 1.4, 0.2]` → Output: **Setosa**.

---

**3. Real-world Impact**
“This workflow can be adapted to many domains. For example, instead of flowers, the same process can classify **customers, products, or tickets** in real-time. It shows how ML models can automate decision-making and reduce manual effort.”