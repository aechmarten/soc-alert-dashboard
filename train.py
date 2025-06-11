import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from collections import Counter
import matplotlib.pyplot as plt
import shap
import joblib
import seaborn as sns

#Load Data
df = pd.read_csv("data/GUIDE_Train.csv")
categories = ["OrgId", "DetectorId", "Category", "SuspicionLevel", "ActionGrouped"]

              #Used during aggregtion to get the most common grade in alerts for incident
def majority_label(labels):
    cnt = Counter(labels)
    common_two = cnt.most_common(2)
    if len(common_two) == 1 or common_two[0][1] > common_two[1][1]:
        return common_two[0][0]
    return "TruePositive"

sns.histplot(df,x='IncidentGrade', hue='IncidentGrade', kde=True, bins=30)
hist = plt.gcf()
hist.savefig('histplot.png')


#Handle rare IDs
for col in ("OrgId", "DetectorId"):
    counts = df[col].value_counts()
    rare = counts[counts < 10].index
    df.loc[df[col].isin(rare), col] = -1

#List all numeric columns that we want to sum per incident
numeric_cols = [
    c for c in df.select_dtypes(include=["int64", "float64"]).columns
        if c not in categories + ["IncidentGrade", "IncidentId"]]

#Aggregegate alerts into incidents
agg_dict = {c: "sum" for c in numeric_cols}
agg_dict.update({cat: "first" for cat in categories})
agg_dict["IncidentGrade"] = majority_label

incident_df = (
    df.groupby("IncidentId", sort=False)
      .agg(agg_dict)
      .reset_index()
      .dropna(subset=["IncidentGrade"])
)

#print statistics
print(f" Alert-level rows: {len(df):,}")
print(f" Incident-level rows: {len(incident_df):,}")
print(" Incident columns:", incident_df.columns.tolist()[:10], "…")

#Change objects to categories for our category cols
for cat in categories:
    incident_df[cat] = incident_df[cat].astype("category")

#Split data for training
X = incident_df.drop(columns="IncidentGrade")
y = incident_df["IncidentGrade"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, random_state=42, stratify=y
)

#Get class weight with inverse frequency
class_counts = y_train.value_counts()
class_weight = {cls: len(y_train) / cnt for cls, cnt in class_counts.items()}

clf = lgb.LGBMClassifier(
    objective="multiclass",
    num_class=3,
    class_weight=class_weight,
#    categorical_feature=["OrgId", "DetectorId", "Category", "SuspicionLevel", "ActionGrouped"],
    n_estimators=600,
    learning_rate=0.02,
    n_jobs=-1
)

print(f" Training on {X_train.shape[0]:,} incidents, {X_train.shape[1]} features (5 categorical + {len(numeric_cols)} numeric)")
clf.fit(X_train, y_train, categorical_feature=categories)
print(clf.classes_)

#print model stats
y_pred = clf.predict(X_test)
print("\nClassification Report")
print(classification_report(y_test, y_pred, digits=4))

#save confusion matrix
cm = confusion_matrix(y_test, y_pred, labels=clf.classes_)
fig, ax = plt.subplots(figsize=(4,4))
sns.heatmap(cm, annot=True, fmt="d", xticklabels=clf.classes_, yticklabels=clf.classes_, ax=ax)
fig.savefig("cm_test.png", bbox_inches="tight")
"""
#SHAP explanations
print("\nComputing SHAP values")
explainer = shap.TreeExplainer(clf)
shap_values = explainer(X_test)

#summary plot
shap.summary_plot(shap_values, X_test)

#summary plot for TruePositive (0:BenignPositive,1:FalsePositive,2:TruePositive)
shap.summary_plot(shap_values[:, :, 2], X_test)
    """
#Save model
joblib.dump(clf, "model.pkl")



