import json
import os
from pathlib import Path
import re

from dotenv import load_dotenv
from fastapi import Body, FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
import google.generativeai as genai
import joblib
import numpy as np
import pandas as pd
import yaml

# Get the project root directory
project_root = Path(__file__).parent.parent.parent

# Load environment variables from .env file
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
feat_path = model_dir / "features.json"
cls_path = model_dir / "classes.json"

app = FastAPI(
    title="SmartHealth-AI API",
    version="1.0.0",
    description="Simple FastAPI server for obesity model (RandomForest) with LLM-powered advice",
)

app.mount(
    "/static",
    StaticFiles(directory=project_root / "taed2_smarthealth_ai" / "api" / "static"),
    name="static",
)


@app.get("/", include_in_schema=False)
def root():
    index_path = project_root / "taed2_smarthealth_ai" / "api" / "static" / "index.html"
    if index_path.exists():
        return FileResponse(index_path)
    return {"message": "SmartHealth-AI API is running. See /docs for API documentation."}


# Initialize as None - will be loaded on startup
model = None
feature_order = []
class_labels = []


@app.on_event("startup")
def load_model():
    """Load model and features on startup"""
    global model, feature_order, class_labels
    try:
        model = joblib.load(model_path)
        feature_order = json.load(open(feat_path, "r", encoding="utf-8"))["feature_order"]
        try:
            class_labels = json.load(open(cls_path, "r", encoding="utf-8"))["classes"]
        except FileNotFoundError:
            class_labels = getattr(model, "classes_", [])
        print("Model loaded successfully")
    except Exception as e:
        print(f"Warning: Could not load model: {e}")


@app.get("/healthz")
def healthz():
    return {
        "status": "ok",
        "model_loaded": True,
        "n_features": len(feature_order),
        "gemini_api_configured": GOOGLE_API_KEY is not None,
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
            "MTRANS_walking": 0,
        },
    )
):
    """
    Predict obesity category based on input features.

    This endpoint uses a Random Forest classifier to predict obesity levels based on eating habits
    and physical condition. It returns only the model prediction without personalized advice.

    **Input Features:**

    - **Gender**: Sex (0 = Male, 1 = Female)
    - **Age**: Age in years (e.g., 21.0)
    - **Height**: Height in meters (e.g., 1.62)
    - **Weight**: Weight in kilograms (e.g., 64.0)
    - **family_history_with_overweight**: Family history of overweight (0 = No, 1 = Yes)
    - **FAVC**: Frequent consumption of high caloric food (0 = No, 1 = Yes)
    - **FCVC**: Frequency of vegetable consumption (1 = Never, 2 = Sometimes, 3 = Always)
    - **NCP**: Number of main meals per day (1 = 1 meal, 2 = 2 meals, 3 = 3 meals, 4 = 4+ meals)
    - **CAEC**: Consumption of food between meals (0 = No, 1 = Sometimes, 2 = Frequently, 3 = Always)
    - **SMOKE**: Do you smoke? (0 = No, 1 = Yes)
    - **CH2O**: Daily water consumption (1 = Less than 1L, 2 = 1-2L, 3 = More than 2L)
    - **SCC**: Do you monitor calorie intake? (0 = No, 1 = Yes)
    - **FAF**: Physical activity frequency per week (0 = None, 1 = 1-2 days, 2 = 2-4 days, 3 = 4-5 days)
    - **TUE**: Time using technology devices per day (0 = 0-2 hours, 1 = 3-5 hours, 2 = More than 5 hours)
    - **CALC**: Alcohol consumption frequency (0 = Never, 1 = Sometimes, 2 = Frequently, 3 = Always)
    - **MTRANS_automobile**: Main transportation is automobile (0 = No, 1 = Yes)
    - **MTRANS_bike**: Main transportation is bicycle (0 = No, 1 = Yes)
    - **MTRANS_motorbike**: Main transportation is motorbike (0 = No, 1 = Yes)
    - **MTRANS_walking**: Main transportation is walking (0 = No, 1 = Yes)

    **Note**: For transportation, only ONE of the MTRANS fields should be set to 1, others should be 0.

    **Response:**

    Returns a JSON object with:
    - **label**: Predicted obesity category (e.g., "Insufficient Weight", "Normal Weight", "Overweight Level I",
    "Overweight Level II", "Obesity Type I", "Obesity Type II", "Obesity Type III")
    - **confidence**: Model confidence score (0.0 to 1.0)
    - **probabilities**: Probability distribution across all obesity categories

    **Example Usage:**

    Send a POST request with all required features. The model will return the predicted obesity category
    along with confidence scores for each possible category.
    """

    if model is None or not feature_order:
        raise HTTPException(status_code=503, detail="Model not loaded")

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

    # Build response with NUMERIC labels (as before)
    resp = {
        "label": str(class_labels[top_i]),  # Returns "0", "1", "2", etc.
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
                "confidence": 0.363,
                "probabilities": {
                    "Insufficient_Weight": 0.063,
                    "Normal_Weight": 0.363,
                    "Overweight_Level_I": 0.300,
                    "Overweight_Level_II": 0.183,
                    "Obesity_Type_I": 0.057,
                    "Obesity_Type_II": 0.033,
                    "Obesity_Type_III": 0.000,
                },
            },
        },
    )
):
    """
    Generate personalized health advice using Google Gemini AI.

    This endpoint takes user health inputs and the model's prediction, then uses Google's Gemini LLM
    to generate personalized, actionable health recommendations tailored to the individual's habits
    and predicted obesity category.

    **Input Structure:**

    The endpoint expects a JSON object with two main sections:

    1. **user_inputs**: Dictionary containing all health and lifestyle features (same as /predict endpoint)
    - **Gender**: Sex (0 = Male, 1 = Female)
    - **Age**: Age in years
    - **Height**: Height in meters
    - **Weight**: Weight in kilograms
    - **family_history_with_overweight**: Family history (0 = No, 1 = Yes)
    - **FAVC**: Frequent high caloric food consumption (0 = No, 1 = Yes)
    - **FCVC**: Vegetable consumption (1 = Never, 2 = Sometimes, 3 = Always)
    - **NCP**: Main meals per day (1-4)
    - **CAEC**: Food between meals (0 = No, 1 = Sometimes, 2 = Frequently, 3 = Always)
    - **SMOKE**: Smoking status (0 = No, 1 = Yes)
    - **CH2O**: Water consumption (1 = <1L, 2 = 1-2L, 3 = >2L)
    - **SCC**: Calorie monitoring (0 = No, 1 = Yes)
    - **FAF**: Physical activity (0 = None, 1 = 1-2 days, 2 = 2-4 days, 3 = 4-5 days)
    - **TUE**: Technology usage (0 = 0-2h, 1 = 3-5h, 2 = >5h)
    - **CALC**: Alcohol consumption (0 = Never, 1 = Sometimes, 2 = Frequently, 3 = Always)

    2. **prediction**: Dictionary with the model's prediction results
    - **label**: Predicted obesity category (e.g., "Normal_Weight", "Obesity_Type_I")
    - **confidence**: Model confidence score (0.0 to 1.0)
    - **probabilities**: Full probability distribution across all categories (used for borderline cases)

    **Response:**

    Returns a JSON object with:
    - **advice**: Array of 2-7 personalized health recommendations based on the person's habits and prediction
    - **note**: Disclaimer! Remind the users to consult healthcare professionals

    **Important Notes:**

    - Requires GOOGLE_API_KEY to be configured in environment variables
    - Generates non-medical advice focused on diet, exercise, and lifestyle changes
    - Recommendations are AI-generated and should not replace professional medical advice

    **Typical Workflow:**

    1. First, call /predict to get obesity category prediction
    2. Then, call /generate-advice with both user inputs and prediction results
    3. Alternatively, use /predict-with-advice to get both in one call
    """

    if not GOOGLE_API_KEY:
        raise HTTPException(
            status_code=503, detail="Advice generation unavailable. GOOGLE_API_KEY not configured."
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

        label_mapping = {
            "0": "Insufficient_Weight",
            "1": "Normal_Weight",
            "2": "Overweight_Level_I",
            "3": "Overweight_Level_II",
            "4": "Obesity_Type_I",
            "5": "Obesity_Type_II",
            "6": "Obesity_Type_III",
        }

        # Convert label and probabilities to readable names
        readable_label = label_mapping.get(
            str(prediction.get("label")), str(prediction.get("label"))
        )

        # Convert probabilities dict keys from numbers to readable names
        probabilities = prediction.get("probabilities", {})
        readable_probabilities = {
            label_mapping.get(str(k), str(k)): v for k, v in probabilities.items()
        }

        # Prompt given to Gemini
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
        - Predicted Category: {readable_label}
        - Confidence: {prediction.get("confidence", 0) * 100:.1f}%
        - Full Probability Distribution: {readable_probabilities}

        **Important**: Consider the full probability distribution when providing advice. If the person is close to another category (within 5-10% probability), mention preventive measures or warnings about potential risks.
        For example, if someone is 50% Normal Weight but 42% Overweight Level I, acknowledge they're on the borderline and focus on prevention.

        Please provide personalized and easy to do health recommendations. 
        Provide as many recommendations as you think are relevant and helpful for this person's situation (typically 2 to 7 recommendations). Each recommendation should be:
        - Specific and practical
        - Based on the person's current habits and prediction
        - Consider the probability distribution (mention if they're borderline between categories)
        - Focused on diet, exercise, lifestyle habits
        - Encouraging and supportive in tone

        Return ONLY a valid JSON object with this exact structure (no markdown, no extra text):
        {{
            "advice": ["recommendation 1", "recommendation 2", ...],
            "note": "These are AI-generated suggestions based on your answers to the form. Please consult with healthcare professionals for personalized medical advice."
        }}"""

        # Call Google Gemini with explicit API key
        model_gemini = genai.GenerativeModel("gemini-2.5-flash")
        response = model_gemini.generate_content(prompt)

        # Extract JSON from response
        response_text = response.text.strip()

        # Remove markdown code blocks if present
        response_text = re.sub(r"^```json\s*", "", response_text)
        response_text = re.sub(r"^```\s*", "", response_text)
        response_text = re.sub(r"\s*```$", "", response_text)

        # Try to find JSON object
        json_match = re.search(r"\{.*\}", response_text, re.DOTALL)
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
        raise HTTPException(
            status_code=500, detail=f"Failed to parse LLM response as JSON: {str(e)}"
        )
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
    advice_result = generate_advice_llm(
        {
            "user_inputs": record,
            "prediction": {
                "label": prediction_result["label"],
                "confidence": prediction_result["confidence"],
                "probabilities": prediction_result["probabilities"],  # ADD THIS LINE!
            },
        }
    )

    # Combine results
    prediction_result["personalized_advice"] = advice_result

    return prediction_result
