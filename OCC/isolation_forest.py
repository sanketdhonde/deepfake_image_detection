import numpy as np
import torch
import torch.nn as nn
from sklearn.ensemble import IsolationForest
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# Load extracted features
print("Loading test features...")
data = np.load("/home/user/Deepfake/OCC/test_features.npz")
test_features = data['features']
test_labels = data['labels']  # 1 = real, 0 = fake

print("Loading train features...")
train_data = np.load("/home/user/Deepfake/extra/train_features.npz")
train_features = train_data['features']

# Train Isolation Forest on real features
print("Training Isolation Forest...")
clf = IsolationForest(n_estimators=100, contamination='auto', random_state=42)
clf.fit(train_features)

# Predict anomaly scores and labels
print("Evaluating on test set...")
scores = clf.decision_function(test_features)  # The lower, the more abnormal
preds = clf.predict(test_features)  # -1 for anomaly, 1 for normal

# Map to 0 (fake) and 1 (real)
preds = (preds == 1).astype(int)

# Classification report
print("\nClassification Report:")
print(classification_report(test_labels, preds, target_names=["Fake", "Real"]))
print(f"ROC AUC: {roc_auc_score(test_labels, -scores):.4f}")  # use -scores because lower = more anomalous

# Optional: visualize scores
plt.figure(figsize=(8,5))
plt.hist(-scores[test_labels==1], bins=50, alpha=0.6, label='Real', color='blue')
plt.hist(-scores[test_labels==0], bins=50, alpha=0.6, label='Fake', color='red')
plt.title("Isolation Forest Anomaly Score Histogram")
plt.xlabel("-Anomaly Score (higher = more anomalous)")
plt.ylabel("Frequency")
plt.legend()
plt.tight_layout()
plt.show()

# t-SNE Visualization
print("Generating t-SNE visualization...")
tsne = TSNE(n_components=2, perplexity=30, random_state=42)
test_features_2d = tsne.fit_transform(test_features)

plt.figure(figsize=(8,6))
plt.scatter(test_features_2d[test_labels==1, 0], test_features_2d[test_labels==1, 1], 
            c='blue', label='Real', alpha=0.6)
plt.scatter(test_features_2d[test_labels==0, 0], test_features_2d[test_labels==0, 1], 
            c='red', label='Fake', alpha=0.6)
plt.title('t-SNE Visualization of Test Features')
plt.legend()
plt.tight_layout()
plt.show()
