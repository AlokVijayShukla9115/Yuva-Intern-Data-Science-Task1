import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, roc_curve, auc, classification_report

print("📥 Initializing Week 4 Predictive Modeling Pipeline...")

# 1. Dataset Generation (Binary Classification Simulation)
np.random.seed(42)
X = np.random.normal(loc=0, scale=1.5, size=(400, 2))
# Creating dependency structure for target classification
y = (X[:, 0] + X[:, 1] * 1.5 + np.random.normal(0, 1, 400) > 0.5).astype(int)

# 2. Stratified Data Splitting
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)

# 3. Model Training (Logistic Regression)
model = LogisticRegression(solver='lbfgs')
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

print("\n🎯 Model Execution Performance Profile:")
print(classification_report(y_test, y_pred))

# 4. Visualization 1: Confusion Matrix
sns.set_theme(style="white")
plt.figure(figsize=(6, 5))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
            xticklabels=['Negative (0)', 'Positive (1)'],
            yticklabels=['Negative (0)', 'Positive (1)'])
plt.title('Figure 1: Logistic Regression Confusion Matrix', fontsize=12, fontweight='bold')
plt.xlabel('Predicted Class Labels')
plt.ylabel('True Class Labels')
plt.savefig('confusion_matrix_profile.png', bbox_inches='tight')
plt.close()

# 5. Visualization 2: ROC Curve Plot
fpr, tpr, thresholds = roc_curve(y_test, y_proba)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(6, 5))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC Curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=1.5, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.title('Figure 2: Receiver Operating Characteristic (ROC)', fontsize=12, fontweight='bold')
plt.xlabel('False Positive Rate (1 - Specificity)')
plt.ylabel('True Positive Rate (Sensitivity)')
plt.legend(loc="lower right")
plt.grid(True, linestyle=':', alpha=0.6)
plt.savefig('roc_curve_trajectory.png', bbox_inches='tight')
plt.close()

print("\n🎉 Model evaluation complete. 2 charts exported successfully.")
