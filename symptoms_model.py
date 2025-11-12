import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report
import pickle

# Load tabular symptom data
data = pd.read_csv("symptoms_sample_200.csv")  # Columns: fever, cough, etc. + label
if "filename" in data.columns:
    data = data.drop("filename",axis=1)
X = data.drop("label", axis=1)
y = data["label"]

# Encoding labels
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Feature scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
print("Class counts:",pd.Series(y_encoded).value_counts())

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_encoded, test_size=0.2, random_state=42)

# Model
model = RandomForestClassifier(n_estimators=100,class_weight='balanced',random_state=42)
model.fit(X_train, y_train)

# Save the model
pickle.dump(model, open("symptom_model.pkl", "wb"))
pickle.dump(scaler, open("scaler.pkl", "wb"))
pickle.dump(le, open("label_encoder.pkl", "wb"))
pickle.dump(X.columns.tolist(),open("symptom_feature_columns.pkl","wb"))

# Evaluation
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred, zero_division=0))
