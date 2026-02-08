import pickle
import sys
import os

print(f"Python version: {sys.version}")
try:
    path = 'public/char_svm_model.pkl'
    print(f"Loading {path} with pickle...")
    with open(path, 'rb') as f:
        model = pickle.load(f)
    print("SUCCESS: Model loaded with pickle")
except Exception as e:
    print(f"ERROR with pickle: {e}")
    try:
        import joblib
        print(f"Loading {path} with joblib...")
        model = joblib.load(path)
        print("SUCCESS: Model loaded with joblib")
    except Exception as e2:
        print(f"ERROR with joblib: {e2}")
