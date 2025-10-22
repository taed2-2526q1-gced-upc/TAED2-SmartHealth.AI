import os
import re
from pathlib import Path
from fastapi import FastAPI, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
import joblib, yaml, json
import pandas as pd
from pathlib import Path
import numpy as np
from dotenv import load_dotenv
import google.generativeai as genai

# Get the project root directory 
project_root = Path(__file__).parent.parent.parent

# Load environment variables from .env file FIRST
load_dotenv(project_root / ".env")

def _to_scalar(v, default=0):
    if isinstance(v, np.generic):
        return v.item()
    if isinstance(v, (list, tuple, np.ndarray, pd.Series)):
        arr = np.asarray(v)
        if arr.size == 0:
            return default
        x = arr.ravel()[0]
        return x.item() if isinstance(x, np.generic) else x
    return v

# Store API key globally
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    print("WARNING: GOOGLE_API_KEY not found! Advice generation will not work.")
else:
    # Configure Google Gemini AFTER loading env vars
    try:
        genai.configure(api_key=GOOGLE_API_KEY)
        print("Google Gemini configured successfully!")
    except Exception as e:
        print(f"Error configuring Gemini: {e}")



params = yaml.safe_load(open(project_root / "params.yaml", "r", encoding="utf-8"))
model_dir = project_root / params["train"]["model_dir"]
model_path = model_dir / "model.joblib"
feat_path  = model_dir / "features.json"
cls_path   = model_dir / "classes.json"

app = FastAPI(
    title="SmartHealth-AI API",
    version="1.0.0",
    description="Simple FastAPI server for obesity model (RandomForest) with LLM-powered advice",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def root():
    return {"message": "SmartHealth-AI API is running. See /docs, /healthz, /predict, /generate-advice"}

model = joblib.load(model_path)
feature_order = json.load(open(feat_path, "r", encoding="utf-8"))["feature_order"]
try:
    class_labels = json.load(open(cls_path, "r", encoding="utf-8"))["classes"]
except FileNotFoundError:
    class_labels = getattr(model, "classes_", [])

@app.get("/healthz")
def healthz():
    return {
        "status": "ok", 
        "model_loaded": True, 
        "n_features": len(feature_order),
        "gemini_api_configured": GOOGLE_API_KEY is not None
    }

@app.post("/predict")
def predict(
    record: dict = Body(
        ...,
        example={
            "Gender": 1,
            "Age": 21.0,
            "Height": 1.62,
            "Weight": 64.0,
            "family_history_with_overweight": 1,
            "FAVC": 0,
            "FCVC": 2.0,
            "NCP": 3.0,
            "CAEC": 1,
            "SMOKE": 0,
            "CH2O": 2.0,
            "SCC": 0,
            "FAF": 0.0,
            "TUE": 1.0,
            "CALC": 0,
            "MTRANS_automobile": 0,
            "MTRANS_bike": 0,
            "MTRANS_motorbike": 0,
            "MTRANS_walking": 0
        },
    )
):
    """
    Predict obesity category based on input features.
    This endpoint only returns the model prediction without advice.
    """
    missing = [c for c in feature_order if c not in record]
    if missing:
        raise HTTPException(status_code=400, detail=f"Missing features: {missing}")

    clean = {k: _to_scalar(record[k]) for k in feature_order}
    df = pd.DataFrame([clean])
    try:
        df = df.apply(pd.to_numeric)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Type error in input: {e}")

    # Get prediction probabilities
    proba = model.predict_proba(df)[0]
    top_i = int(np.argmax(proba))

    # Build response
    resp = {
        "label": str(class_labels[top_i]),
        "confidence": float(proba[top_i]),
        "probabilities": {str(class_labels[i]): float(proba[i]) for i in range(len(class_labels))},
    }

    return resp


@app.post("/generate-advice")
def generate_advice_llm(
    data: dict = Body(
        ...,
        example={
            "user_inputs": {
                "Gender": 1,
                "Age": 21.0,
                "Height": 1.62,
                "Weight": 64.0,
                "family_history_with_overweight": 1,
                "FAVC": 0,
                "FCVC": 2.0,
                "NCP": 3.0,
                "CAEC": 1,
                "SMOKE": 0,
                "CH2O": 2.0,
                "SCC": 0,
                "FAF": 0.0,
                "TUE": 1.0,
                "CALC": 0,
            },
            "prediction": {
                "label": "Normal_Weight",
                "confidence": 0.85,
            }
        }
    )
):
    """
    Generate personalized health advice using Google Gemini based on user inputs and model prediction.
    """
    
    if not GOOGLE_API_KEY:
        raise HTTPException(
            status_code=503, 
            detail="Advice generation unavailable. GOOGLE_API_KEY not configured."
        )
    
    try:
        user_inputs = data.get("user_inputs", {})
        prediction = data.get("prediction", {})
        
        # Add these mappings at the top of the generate_advice_llm function
        caec_calc_map = {0: "No", 1: "Sometimes", 2: "Frequently", 3: "Always"}
        fcvc_map = {1: "Never", 2: "Sometimes", 3: "Always"}
        ncp_map = {1: "1 meal", 2: "2 meals", 3: "3 meals", 4: "4+ meals"}
        ch2o_map = {1: "Less than 1L", 2: "1-2L", 3: "More than 2L"}
        faf_map = {0: "None", 1: "1-2 days", 2: "2-4 days", 3: "4-5 days"}
        tue_map = {0: "0-2 hours", 1: "3-5 hours", 2: "More than 5 hours"}

        # Then use them in the prompt:
        prompt = f"""You are a health and wellness advisor. Based on the following information about a person, provide personalized, actionable health advice that does not require taking medication.

        **Person's Information:**
        - Sex: {"Female" if user_inputs.get("Gender") == 1 else "Male"}
        - Age: {user_inputs.get("Age")} years
        - Height: {user_inputs.get("Height")} meters
        - Weight: {user_inputs.get("Weight")} kg
        - Family history of overweight: {"Yes" if user_inputs.get("family_history_with_overweight") == 1 else "No"}
        - Frequent consumption of high caloric food: {"Yes" if user_inputs.get("FAVC") == 1 else "No"}
        - Vegetable consumption: {fcvc_map.get(int(user_inputs.get("FCVC", 1)), "Unknown")}
        - Number of main meals per day: {ncp_map.get(int(user_inputs.get("NCP", 3)), "Unknown")}
        - Food consumption between meals: {caec_calc_map.get(int(user_inputs.get("CAEC", 0)), "Unknown")}
        - Smoking: {"Yes" if user_inputs.get("SMOKE") == 1 else "No"}
        - Daily water intake: {ch2o_map.get(int(user_inputs.get("CH2O", 2)), "Unknown")}
        - Monitors calorie intake: {"Yes" if user_inputs.get("SCC") == 1 else "No"}
        - Physical activity frequency: {faf_map.get(int(user_inputs.get("FAF", 0)), "Unknown")}
        - Daily technology usage: {tue_map.get(int(user_inputs.get("TUE", 1)), "Unknown")}
        - Alcohol consumption: {caec_calc_map.get(int(user_inputs.get("CALC", 0)), "Unknown")}

        **ML Model (Random Forest) Prediction:**
        - Obesity Category: {prediction.get("label")}
        - Confidence: {prediction.get("confidence", 0) * 100:.1f}%

        Please provide personalized and easy to do health recommendations. 
        Provide as many recommendations as you think are relevant and helpful for this person's situation (typically 2 to 7 recommendations). Each recommendation should be:
        - Specific and practical
        - Based on the person's current habits and prediction
        - Focused on diet, exercise, lifestyle habits
        - Encouraging and supportive in tone

        Return ONLY a valid JSON object with this exact structure (no markdown, no extra text):
        {{
            "advice": ["recommendation 1", "recommendation 2", ...],
            "note": "These are AI-generated suggestions based on your answers to the form. Please consult with healthcare professionals for personalized medical advice."
        }}"""
        
        # Call Google Gemini with explicit API key
        model_gemini = genai.GenerativeModel('gemini-2.5-flash')        
        response = model_gemini.generate_content(prompt)
                        
        # Extract JSON from response
        response_text = response.text.strip()
        
        # Remove markdown code blocks if present
        response_text = re.sub(r'^```json\s*', '', response_text)
        response_text = re.sub(r'^```\s*', '', response_text)
        response_text = re.sub(r'\s*```$', '', response_text)
        
        # Try to find JSON object
        json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
        if json_match:
            advice_data = json.loads(json_match.group())
        else:
            advice_data = json.loads(response_text)
        
        # Validate the response structure
        if "advice" not in advice_data or not isinstance(advice_data["advice"], list):
            raise ValueError("Invalid response structure from LLM")
        
        return advice_data
        
    except json.JSONDecodeError as e:
        print(f"❌ JSON parsing error: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to parse LLM response as JSON: {str(e)}")
    except Exception as e:
        print(f"❌ Error in generate_advice_llm: {e}")
        raise HTTPException(status_code=500, detail=f"Error generating advice: {str(e)}")


@app.post("/predict-with-advice")
def predict_with_advice(
    record: dict = Body(
        ...,
        example={
            "Gender": 1,
            "Age": 21.0,
            "Height": 1.62,
            "Weight": 64.0,
            "family_history_with_overweight": 1,
            "FAVC": 0,
            "FCVC": 2.0,
            "NCP": 3.0,
            "CAEC": 1,
            "SMOKE": 0,
            "CH2O": 2.0,
            "SCC": 0,
            "FAF": 0.0,
            "TUE": 1.0,
            "CALC": 0,
            "MTRANS_automobile": 0,
            "MTRANS_bike": 0,
            "MTRANS_motorbike": 0,
            "MTRANS_walking": 0,
        },
    )
):
    """
    Combined endpoint: Get prediction AND personalized advice in one call.
    """
    # Get prediction
    prediction_result = predict(record)
    
    # Generate advice
    advice_result = generate_advice_llm({
        "user_inputs": record,
        "prediction": {
            "label": prediction_result["label"],
            "confidence": prediction_result["confidence"]
        }
    })
    
    # Combine results
    prediction_result["personalized_advice"] = advice_result
    
    return prediction_result