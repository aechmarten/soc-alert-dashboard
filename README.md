# SOC Alert Risk-Scoring Dashboard

An interactive platform for cybersecurity analysts to prioritize alerts using a LightGBM multiclass classifier and SHAP-based explainability.

## 🚀 Features

- **Data preprocessing**: Aggregates Microsoft GUIDE alerts into incident-level records.
- **Model training**: Trains a LightGBM classifier to distinguish True Positive, False Positive, and Benign incidents.
- **Explainability**: Global SHAP beeswarm and per-incident SHAP waterfall visualizations.
- **Interactive dashboard**: Streamlit app for EDA charts, risk-score distribution, feature importance, and detailed incident explanations.

## 📂 Repository Structure

```
├── data/
│   ├── GUIDE_Test.csv       # Testing dataset without IncidentGrade (download via Kaggle)
│   └── GUIDE_Train.csv      # Training dataset with IncidentGrade (download via Kaggle)
├── download_data.sh         # Script to fetch full dataset from Kaggle
├── train.py                   # Training pipeline: data prep, model train, metrics export
├── webapp.py                # Streamlit dashboard for EDA and explainability
├── requirements.txt         # Pinned Python dependencies
├── README.md                # This guide
└── .gitignore               # Excludes large files and env folders
```

## 🛠️ Prerequisites

- **Python** 3.9–3.11
- **Git**
- **Kaggle API** credentials (for full dataset)
- **8 GB+ RAM** recommended

## ⚙️ Installation

1. **Clone the repo**:
   ```bash
   git clone https://github.com/aechmarten/soc-alert-dashboard.git
   cd soc-alert-dashboard
   ```
2. **Create a virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate    # Linux/macOS
   venv\\Scripts\\activate   # Windows
   ```
3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## 📥 Data Setup

1. Download the “Microsoft Security Incident Prediction” dataset ZIP from Kaggle: [https://www.kaggle.com/datasets/Microsoft/microsoft-security-incident-prediction](https://www.kaggle.com/datasets/Microsoft/microsoft-security-incident-prediction)

2. Extract the ZIP so that `GUIDE_Train.csv` and `GUIDE_Test.csv` is placed in the `data/` folder at the repository root:

   ```bash
   unzip security-incident-prediction.zip -d data/
   ```

## ▶️ Usage

### 1. Train the Model

```bash
python train.py 
```

### 2. Launch the Dashboard

```bash
streamlit run webapp.py \
  --server.maxUploadSize=2048 \
  --server.maxMessageSize=2048
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

#### Dashboard Workflow

1. **Upload** your prepared `GUIDE_Test.csv` file.
2. Explore **EDA charts**:
   - Daily alert vs. incident volumes
   - Incident class ratios
   - Feature distributions
3. View **Global Model Insights** (SHAP beeswarm, feature importance, risk-score distribution).
4. Generate **Per-Incident Explanations** by selecting an incident index.

## 🐞 Troubleshooting

- **Slow initial load**: The first SHAP computation takes \~5 minutess; subsequent interactions are cached.


