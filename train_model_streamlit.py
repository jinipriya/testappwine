import joblib

from sklearn.datasets import load_wine
from sklearn.ensemble import RandomForestClassifier

# Load dataset
data = load_wine()

X = data.data
y = data.target

# Create and train model
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X, y)

# Save trained model
joblib.dump(model, "wine_model.joblib")

print("Model trained and saved successfully!")