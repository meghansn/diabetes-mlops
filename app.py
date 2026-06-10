import joblib
from fastapi import FastAPI
from pydantic import BaseModel

# =============================================================================
# 1. MODEL SERIALIZATION & INITIALIZATION
# =============================================================================
# When a web server spins up, we want to load our machine learning model into
# memory *once* at startup. This ensures that every time a user requests a 
# prediction, the model is already waiting in RAM and can answer instantly.

try:
    # joblib.load opens the binary file we generated in train.py and reconstructs
    # the trained mathematical formula (the weights and biases).
    model = joblib.load("model.joblib")
    print("✅ MLOps Log: model.joblib successfully loaded into RAM.")
except FileNotFoundError:
    # Defensive programming: If the model file is missing, we gracefully crash the 
    # server immediately with a clear error message rather than failing silently later.
    print("❌ Critical Error: model.joblib not found in root directory!")
    print("👉 Please execute 'python train.py' to generate the model artifact first.")
    raise

# We initialize the core FastAPI application object.
# This object handles routing incoming web requests to the correct Python functions.
app = FastAPI(
    title="Diabetes Progression Prediction API",
    description="Production-ready MLOps microservice serving real-time predictions using Ridge Regression.",
    version="1.0.0"
)


# =============================================================================
# 2. DATA VALIDATION SCHEMA (The Pydantic Gatekeeper)
# =============================================================================
# In production MLOps, you cannot trust user input. If a user sends a text string
# like "high" instead of a number for blood pressure, Scikit-learn will crash.
#
# Pydantic solves this by acting as a strict gatekeeper. By defining this class,
# FastAPI will automatically inspect incoming JSON payloads. If the payload data
# types don't match our schema perfectly, the API automatically rejects the 
# request with a helpful error message before it ever touches our ML model.

class PatientFeatures(BaseModel):
    # We require exactly 10 floating-point numbers matching the Scikit-learn schema
    age: float
    sex: float
    bmi: float
    bp: float
    s1: float  # tc (Total Serum Cholesterol)
    s2: float  # ldl (Low-Density Lipoproteins)
    s3: float  # hdl (High-Density Lipoproteins)
    s4: float  # tch (Total Cholesterol / HDL Ratio)
    s5: float  # ltg (Log of Serum Triglycerides Level)
    s6: float  # glu (Blood Sugar Level)


# =============================================================================
# 3. API ROUTING & ENDPOINTS
# =============================================================================

@app.get("/")
def read_root():
    """
    HEALTH CHECK ENDPOINT
    In cloud environments (like AWS or Google Cloud), orchestrators constantly 
    ping the root '/' path. If it returns a 200 OK status code, the platform 
    knows our container is healthy and running smoothly.
    """
    return {
        "status": "Healthy",
        "message": "Diabetes Prediction Container is Live. Navigate to /docs for the interactive UI."
    }


@app.post("/predict")
def predict_progression(data: PatientFeatures):
    """
    REAL-TIME INFERENCE ENDPOINT
    This is where the magic happens. It listens for HTTP POST requests containing
    a JSON payload matching our PatientFeatures schema.
    """
    
    # STEP 1: RESHAPE THE DATA
    # Incoming JSON arrives as a clean object: {"age": 0.03, "sex": -0.04, ...}
    # However, Scikit-learn algorithms strictly expect data shaped as a 2D array
    # or matrix (rows x columns), even if we are only predicting for a single patient.
    # We unpack our Pydantic variables into a nested Python list: [[value1, value2, ...]]
    features = [[
        data.age, data.sex, data.bmi, data.bp, 
        data.s1, data.s2, data.s3, data.s4, data.s5, data.s6
    ]]
    
    # STEP 2: CALCULATE THE PREDICTION
    # We feed the formatted 2D list into our pre-loaded model's .predict() method.
    # Because we passed a list containing one row of data, .predict() returns a 
    # list containing one answer: [predicted_value]. We grab the first element using [0].
    prediction = model.predict(features)[0]
    
    # STEP 3: RETURN THE RESPONSE
    # FastAPI automatically serializes this native Python dictionary back into a 
    # standard JSON response payload to send back across the web to the user.
    return {
        "status": "Success",
        "input_processed": data.model_dump(),  # Echoes back the validated inputs for tracking
        "predicted_disease_progression": round(float(prediction), 2) # Rounds the regression target index
    }