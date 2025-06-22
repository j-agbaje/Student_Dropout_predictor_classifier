# Student Dropout Prediction Using Neural Networks

## Problem Statement

University dropout represents a critical challenge in higher education systems globally, with significant implications for both students and institutions. Current predictive approaches often rely on static demographic data or simplistic models that fail to capture complex, nonlinear relationships between academic performance, socioeconomic factors, and behavioral patterns. Many existing systems lack early detection capabilities and struggle with temporal dynamics of student data, resulting in delayed interventions. This project develops a comprehensive, machine learning-based early warning system that accurately predicts student dropout risk through advanced neural network architectures.

## Dataset Description

The project utilizes the "Predict Students' Dropout and Academic Success" dataset containing 4,424 student records across 36 features from a higher education institution. The dataset encompasses students enrolled in diverse undergraduate programs including agronomy, design, education, nursing, journalism, management, social service, and technologies. Each instance represents a student with comprehensive enrollment information, demographic characteristics, socioeconomic factors, and academic performance data from first and second semesters. The target variable has been reduced from three categories (dropout, enrolled, graduate) to binary classification focusing on "Graduated" vs "Dropout" predictions.

**Dataset Source:** [Predict Students' Dropout and Academic Success](https://archive.ics.uci.edu/dataset/697/predict+students+dropout+and+academic+success)

## Model Training Results

| Training Instance | Optimizer | Regularizer | Epochs | Early Stopping | Layers | Learning Rate | Accuracy | F1 Score | Recall | Precision |
|-------------------|-----------|-------------|---------|----------------|---------|---------------|----------|----------|---------|-----------|
| Instance 1 | Default | None | 200 | No | 3 (64-32-16) | Default | 0.8566 | 0.8274 | 0.8274 | 0.8274 |
| Instance 2 | Adam | None | 350 | Yes | 3 (128-64-32) | 0.001 | 0.8989 | 0.8736 | 0.8407 | 0.9091 |
| Instance 3 | Adam | L2 (0.00001) | 400 | Yes | 3 (128-64-32) | 0.01 | 0.8768 | 0.8534 | 0.8628 | 0.8442 |
| Instance 4 | RMSprop | L2 (0.00001) | 400 | Yes | 3 (128-64-32) | 0.001 | 0.8438 | 0.7962 | 0.7345 | 0.8691 |
| Instance 5 | RMSprop | L1 (0.00001) | 400 | Yes | 3 (256-128-64) | 0.001 | 0.8842 | 0.8518 | 0.8009 | 0.9095 |

*Note: All metrics reported are validation metrics*

## Random Forest Baseline Comparison

| Model | Accuracy | F1 Score | Recall | Precision | ROC-AUC |
|-------|----------|----------|---------|-----------|---------|
| Random Forest | 0.9138 | 0.8683 | 0.8245 | 0.9172 | 0.9512 |

## Video Presentation Link



**Random Forest Hyperparameters:**
- n_estimators: 300
- max_depth: 15
- min_samples_split: 5
- min_samples_leaf: 1
- max_features: 'log2'

## Key Findings and Analysis

### Neural Network Performance Summary

**Best Performing Neural Network:** While Instance 2 (Adam optimizer, no regularization) achieved the highest validation accuracy of 89.89% and F1-score of 87.36% is traditionally the best model. For the specific use-case of student dropout prediction, recall is the most important metric because false negatives - that is missing more students who are at dropout risk, is the worst case scenaron. Therefore, model_3 which has the highest recall of 86.28% seems to be the more production-ready  model(although both models could ue further improvement) albeit a lower accuracy of 87.68% and f1-score of 85.34%

**Overall Best Model:** Random Forest outperformed all neural network configurations with 91.38% accuracy and 86.83% F1-score.

### Critical Insights

1. **No Regularization Advantage**: The first two models using no dropout or regularization performed exceptionally well, suggesting the dataset doesn't require heavy regularization techniques for this particular problem.

2. **Architecture Optimization**: Doubling the neuron count in deeper layer structures (128-64-32) significantly improved performance compared to the baseline (64-32-16) architecture.

3. **Training Epochs Sweet Spot**: Increasing training epochs to 400 provided optimal results up to 400 neurons, but exceeding this threshold led to overfitting issues.

4. **Early Stopping Strategy**: Implementing early stopping with patience of 40 epochs allowed sufficient training time while preventing overfitting.

5. **Regularization Counterproductive**: Models 3, 4, and 5 showed that adding L1/L2 regularization actually degraded performance, indicating the dataset benefits from model complexity rather than regularization constraints.

6. **Optimizer Comparison**: Adam optimizer consistently outperformed RMSprop across different configurations, showing better convergence and stability.

7. **Generalization Success**: Model 2's validation accuracy exceeded training accuracy, demonstrating excellent generalization capabilities.

### Model Comparison: Neural Networks vs Random Forest

The Random Forest model significantly outperformed all neural network configurations, achieving superior accuracy (91.38% vs 89.89%) and comparable F1-scores. This suggests that for this particular dataset, the ensemble method's ability to handle feature interactions and reduce overfitting provides advantages over deep learning approaches. The Random Forest's interpretability and robust performance make it the recommended approach for this student dropout prediction task.

## Instructions for Running the Project

### Prerequisites
```bash
pip install tensorflow scikit-learn pandas numpy matplotlib seaborn
```

### Loading and Running the Notebook
1. Clone the repository or download the notebook file
2. Open the Jupyter notebook in your preferred environment
3. Ensure the dataset is in the same directory or update the file path
4. Run all cells sequentially

### Loading the Best Saved Model
```python
# For Neural Network (Instance 2)
from tensorflow.keras.models import load_model
best_nn_model = load_model('best_model.keras')

# For Random Forest (Recommended)
import joblib
rf_model = joblib.load('rf_model.pkl')
```

### Making Predictions
```python
# Using the loaded model
predictions = rf_model.predict(X_test)
prediction_probabilities = rf_model.predict_proba(X_test)
```

## Repository Structure
```
project/
├── student_dropout_prediction.ipynb
├── best_neural_network_model.h5
├── best_random_forest_model.pkl
├── data/
│   └── dataset.csv
└── README.md
```

## Conclusion

This project successfully demonstrates the application of both neural networks and ensemble methods for student dropout prediction. While neural networks showed promising results, the Random Forest model emerged as the superior approach for this specific dataset, offering better accuracy and interpretability for practical deployment in educational institutions.
