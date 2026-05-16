# BreastGuard AI

BreastGuard AI is an AI-powered breast cancer risk prediction and clinical report analysis system developed using Machine Learning and Flask.

The platform assists in early-stage breast cancer screening by analyzing 30 clinically significant diagnostic features extracted either manually or automatically from uploaded PDF reports.

---

# Features

- AI-powered breast cancer risk prediction
- Supports 30 clinical diagnostic features
- PDF medical report upload and automated feature extraction
- Real-time prediction and confidence scoring
- Low / Moderate / High risk classification
- Modern futuristic healthcare dashboard UI
- Interactive analytics visualization
- Responsive frontend design
- Machine Learning pipeline with feature scaling and calibrated probabilities

---

# Tech Stack

## Frontend
- HTML5
- TailwindCSS
- JavaScript
- Chart.js
- jQuery

## Backend
- Flask (Python)

## Machine Learning
- Scikit-Learn
- Random Forest Classifier
- Logistic Regression
- Decision Tree Classifier
- StandardScaler
- CalibratedClassifierCV

## Data Processing
- NumPy
- Pandas

## PDF Processing
- PyMuPDF (fitz)
- Regex-based feature extraction

## Model Serialization
- Joblib

---

# Machine Learning Workflow

1. Data preprocessing and cleaning
2. Label encoding of diagnosis labels
3. Feature scaling using StandardScaler
4. Train-test split (80-20)
5. Model training using multiple classifiers
6. Hyperparameter optimization of Random Forest
7. Probability calibration for realistic confidence scores
8. Model serialization using Joblib
9. Flask API integration
10. Frontend deployment

---

# Dataset

The project uses the Breast Cancer Wisconsin Diagnostic Dataset.

The dataset contains 30 clinically significant diagnostic features including:

- radius_mean
- texture_mean
- perimeter_mean
- area_mean
- smoothness_mean
- concavity_mean
- symmetry_worst
- fractal_dimension_worst

and more.


