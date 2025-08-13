# 🍷 Wine Quality Prediction

Welcome to the Wine Quality Prediction project!  
This repository provides a complete workflow for predicting wine quality using machine learning, including data exploration, visualization, model training, and an interactive Streamlit web app for making predictions.

---

## 🚀 Features

- **Data Exploration:** Browse, filter, and understand the wine dataset.
- **Interactive Visualizations:** Explore feature distributions, correlations, and relationships with dynamic charts.
- **Model Training:** Trains multiple classifiers, evaluates them, and selects the best-performing model.
- **Prediction App:** Enter wine features and instantly predict wine quality.
- **Model Performance:** Review test metrics (accuracy, precision, recall, F1 score) and confusion matrix.

---

## 📁 Project Structure

```
Wine_Quality_prediction/
│
├── app.py                  # Streamlit web application
├── data/
│   └── WineQT.csv          # Wine quality dataset
├── models/
│   └── best_model.pkl      # Trained ML model
├── notebooks/
│   └── model_training.ipynb# Model training and analysis
├── outputs/
│   ├── test_metrics.csv    # Model performance metrics
│   ├── X_test.csv          # Test features (optional)
│   └── y_test.csv          # Test labels (optional)
└── README.md               # Project documentation
```

---

## 🛠️ Getting Started

1. **Clone the repository:**
   ```sh
   git clone https://github.com/yourusername/Wine_Quality_prediction.git
   cd Wine_Quality_prediction
   ```

2. **Install dependencies:**
   ```sh
   pip install -r requirements.txt
   ```

3. **(Optional) Train the model:**
   - Open `notebooks/model_training.ipynb` and run all cells to train and save the model.
   - This will generate `models/best_model.pkl` and `outputs/test_metrics.csv`.

4. **Run the Streamlit app:**
   ```sh
   streamlit run app.py
   ```

---

## 📝 Usage

- **Data Exploration:** Inspect the dataset, check for missing values, and filter by wine quality.
- **Visualisations:** Analyze feature distributions, correlations, and relationships interactively.
- **Prediction:** Input wine features to predict if the wine is of good quality (quality ≥ 7).
- **Model Performance:** Review accuracy, precision, recall, F1 score, and confusion matrix on the test set.

---

## ⚙️ Requirements

- Python 3.8 or higher
- See `requirements.txt` for all Python dependencies

---

## 📊 Dataset

- [UCI Machine Learning Repository - Wine Quality Data Set](https://archive.ics.uci.edu/ml/datasets/wine+quality)

---

## 🤝 Contributing

Contributions are welcome!  
Feel free to open issues or submit pull requests for improvements, bug fixes, or new features.

---

## 📄 License

This project is licensed under the MIT License.

---

*Made with ❤️ using Streamlit, scikit-learn, and pandas*

