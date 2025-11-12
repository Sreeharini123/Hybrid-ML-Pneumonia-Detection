import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from sklearn.preprocessing import StandardScaler
import pickle

# Load models
cnn_model = load_model("cnn_xray_model.h5")
symptom_model = pickle.load(open("symptom_model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))
le = pickle.load(open("label_encoder.pkl", "rb"))

# Load test data
feature_columns=pickle.load(open("symptom_feature_columns.pkl","rb"))
symptom_values=[45,1,1,1,1]
symptoms = pd.DataFrame([symptom_values],columns=feature_columns)
# Image processing
img_path = "person18_virus_49.jpeg"
img = image.load_img(img_path, target_size=(224, 224))
img_array = image.img_to_array(img) / 255.0
img_array = np.expand_dims(img_array, axis=0)
# CNN prediction
cnn_pred = cnn_model.predict(img_array)[0][0]  # Probability of pneumonia

# Symptoms processing
symptoms_scaled = scaler.transform(symptoms)
symptom_pred = symptom_model.predict_proba(symptoms_scaled)[0][1]  # Probability of pneumonia

# Fusion (simple average)
final_score = (cnn_pred + symptom_pred) / 2
final_label = "PNEUMONIA" if final_score > 0.5 else "NORMAL"

print(f"CNN: {cnn_pred:.2f}, Symptoms: {symptom_pred:.2f}, Final: {final_score:.2f} → {final_label}")
