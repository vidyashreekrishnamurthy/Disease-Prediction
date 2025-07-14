from sklearn.ensemble import RandomForestClassifier
import joblib
import numpy as np

# Dummy data for testing
X = np.random.randint(1, 10, size=(10, 5))
y = ['DiseaseA', 'DiseaseB', 'DiseaseC', 'DiseaseA', 'DiseaseB', 'DiseaseC', 'DiseaseA', 'DiseaseB', 'DiseaseC', 'DiseaseA']

model = RandomForestClassifier()
model.fit(X, y)

# Save the model
joblib.dump(model, 'model.pkl')

print("Model saved as model.pkl")
