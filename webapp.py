import streamlit as st
import joblib
import shap
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
import seaborn as sns
from PIL import Image

categories = ["OrgId", "DetectorId", "Category", "SuspicionLevel", "ActionGrouped"]

@st.cache_resource(show_spinner=False)
def load_model(path="model.pkl"):
    model = joblib.load(path)
    expl = shap.TreeExplainer(model)
    return model, expl

st.set_page_config(page_title="SOC Alert Dashboard", layout="wide")
st.title("SOC Alert Dashboard")
st.markdown("Upload a CSV of alert data")

with st.spinner("Loading model"):
    clf, explainer = load_model("model.pkl")

uploaded_file = st.file_uploader(
        label="Upload a CSV file",
        type="csv",
        help="This CSV must have the same columns as training data"
)

if uploaded_file is None:
    st.info("Please upload a CSV")
    st.stop()

@st.cache_data(show_spinner=False)
def load_and_prep(df: pd.DataFrame):
    if "Timestamp" in df.columns and not pd.api.types.is_datetime64_any_dtype(df["Timestamp"]):
        df["Timestamp"] = pd.to_datetime(df["Timestamp"], utc=True, errors="coerce")

    def majority_label(labels):
        count = Counter(labels)
        common_two = count.most_common(2)
        if len(common_two) == 1 or common_two[0][1] > common_two[1][1]:
            return common_two[0][0]
        return "TruePositive"

    for col in ("OrgId", "DetectorId"):
        counts = df[col].value_counts()
        rare = counts[counts < 10].index
        df.loc[df[col].isin(rare), col] = -1

    num_cols = [
            c for c in df.select_dtypes(include=["int64", "float64"]).columns
        if c not in categories + ["IncidentId"]]

    agg_dict = {c: "sum" for c in num_cols}
    agg_dict.update({cat: "first" for cat in categories})
    
    incident_df = df.groupby("IncidentId", sort=False).agg(agg_dict).reset_index()

    for cat in categories:
        if cat in incident_df.columns:
            incident_df[cat] = incident_df[cat].astype("category")

    return incident_df

@st.cache_data(show_spinner=False)
def load_csv(uploaded_file):
    return pd.read_csv(uploaded_file)

df_raw = load_csv(uploaded_file)

feature_cols = clf.booster_.feature_name() if hasattr(clf, "booster_") else clf.feature_name_

df_features = load_and_prep(df_raw)
display_df = df_features.copy()

@st.cache_data(show_spinner=False)
def compute_risk(df_features, tp_index):
    probability = clf.predict_proba(df_features)[:, tp_index]
    return (probability * 100).round(1)

with st.spinner("Scoring Incidents"):
    tp_index = list(clf.classes_).index("TruePositive")
    display_df["RiskScore"] = compute_risk(df_features, tp_index)

@st.cache_data(show_spinner=False)
def compute_global_shap(df_features):
    values = explainer(df_features)
    tp_index = list(clf.classes_).index("TruePositive")

    shap_tp = shap.Explanation(
        values = values.values[..., tp_index],
        base_values = values.base_values[tp_index],
        data = values.data,
        feature_names = values.feature_names
    )

    return shap_tp

with st.expander("Global Feature Impact"):
    shap_tp = compute_global_shap(df_features)

    fig, ax = plt.subplots(figsize=(8,6))
    shap.summary_plot(shap_tp, df_features, plot_type="dot", show=False, max_display=20)
    st.pyplot(fig)

incident_hist = Image.open("histplot.png")
with st.expander("IncidentGrade Histogram"):
    st.image(incident_hist, caption="Histogram of IncidentGrade distribution from training data")

cm_image = Image.open("cm_test.png")

with st.expander("Confusion Matrix"):
    st.image(cm_image, caption="Confusion Matrix on hold-out test set")


st.subheader("Ranked Incidents by Risk Score")
display_df = display_df.sort_values("RiskScore", ascending=False).reset_index(drop=True)

for col in display_df.columns:
    if isinstance(display_df[col].dtype, pd.CategoricalDtype) or display_df[col].dtype == object:
        display_df[col] = display_df[col].astype(str)

st.dataframe(display_df, use_container_width=True)

st.subheader("SHAP Explanation")

row_to_explain = st.number_input(
        "select row index to explain",
        min_value=0, max_value=len(display_df)-1, value = 0
)

def compute_row_shap(row_to_explain):
    X_local = df_features.iloc[[row_to_explain]]
    local_cols = X_local.columns.tolist()

    if "RiskScore" in X_local.columns:
        X_local = X_local.drop(columns=["RiskScore"])

    shap_explainer = explainer(X_local)#[:, :, tp_index]
    return shap.Explanation(
        values=shap_explainer.values[0, :, tp_index].reshape(1, -1),
        base_values=shap_explainer.base_values[0, tp_index].reshape(1),
        data=shap_explainer.data[0].reshape(1, -1),
        feature_names=shap_explainer.feature_names
    )

if 'local_shap_cache' not in st.session_state:
    st.session_state['local_shap_cache'] = {}

if row_to_explain not in st.session_state['local_shap_cache']:
    st.session_state['local_shap_cache'][row_to_explain] = \
            compute_row_shap(row_to_explain)

explanation = st.session_state['local_shap_cache'][row_to_explain]

fig_loc, ax_loc = plt.subplots(figsize=(6,4))
shap.plots.waterfall(explanation[0], max_display=10, show=False)
st.pyplot(fig_loc)



