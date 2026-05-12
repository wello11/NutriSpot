import json
import os
import traceback
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing_extensions import Literal

from meal_model import UserProfile, MealRecommendationSystem
from chatbot_engine import FoodChatbot


app = FastAPI(
    title="Nutrition & Meal Recommendation API",
    description="API for predicting nutrition targets, recommending meals, and chatting with an AI assistant.",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ===== Paths (Docker safe) =====
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR, "nutrition_model.joblib")
RECIPES_PATH = os.path.join(BASE_DIR, "recips.json")
DATASET_PATH = os.path.join(BASE_DIR, "nutrition_dataset.csv")

system: Optional[MealRecommendationSystem] = None
chatbot: Optional[FoodChatbot] = None
startup_errors: Dict[str, str] = {}


# =========================
# Pydantic Models (fixes Swagger docs)
# =========================

class UserProfileInput(BaseModel):
    gender: Literal["male", "female"] = Field(..., example="male", description="Gender")
    age: int = Field(..., example=25, description="Age in years")
    height: float = Field(..., example=175.0, description="Height in cm")
    weight: float = Field(..., example=75.0, description="Weight in kg")
    mealsPerDay: Literal[2, 3] = Field(3, example=3, description="Meals per day")
    allergies: List[Literal["Lactose", "Gluten", "Nuts", "None"]] = Field(
        default=["None"], example=["None"], description="List of allergies"
    )
    goal: Literal[
        "Lose weight", "Gain weight", "Improve health", "Maintain weight", "Build muscle"
    ] = Field(..., example="Lose weight", description="Fitness goal")
    healthCondition: Literal[
        "High Blood Pressure", "Heart Disease", "Diabetes", "None"
    ] = Field("None", example="None", description="Health condition")
    healthNotes: str = Field("", example="", description="Additional health notes")
    activityLevel: Literal["Sedentary", "Light", "Moderate", "High"] = Field(
        ..., example="Sedentary", description="Activity level"
    )
    dietType: Literal["High protein", "Vegan", "Low carb", "Keto"] = Field(
        ..., example="High protein", description="Diet type"
    )
    calories: Optional[float] = Field(None, example=1500.0, description="Daily calories cap")
    proteins: Optional[float] = Field(None, example=150.0, description="Daily protein cap (g)")
    carbs: Optional[float] = Field(None, example=150.0, description="Daily carbs cap (g)")
    fats: Optional[float] = Field(None, example=70.0, description="Daily fats cap (g)")


class ChatRequest(BaseModel):
    message: str = Field(..., example="hi", description="User message to the chatbot")
    user_profile: Optional[UserProfileInput] = Field(None, description="Optional user profile context")
    recommendations: Optional[List[Dict[str, Any]]] = Field(
        None, description="Previous meal recommendations context"
    )


# =========================
# Startup
# =========================
@app.on_event("startup")
def load_models():
    global system, chatbot, startup_errors

    print("===== STARTUP DEBUG =====")
    print("CWD:", os.getcwd())
    print("BASE_DIR:", BASE_DIR)
    if os.path.exists(BASE_DIR):
        print("FILES:", os.listdir(BASE_DIR))
    print("=========================")

    if not os.path.exists(RECIPES_PATH):
        msg = "[ERROR] recips.json missing"
        print(msg)
        startup_errors["recipes"] = msg
        return

    # ===== Load model =====
    try:
        if os.path.exists(MODEL_PATH):
            system = MealRecommendationSystem(
                model_path=MODEL_PATH,
                recipes_path=RECIPES_PATH
            )
            print("[OK] Model loaded")
        else:
            print("[WARN] Model file not found")
            system = None
    except Exception as e:
        msg = f"Model load failed: {str(e)}\n{traceback.format_exc()}"
        print(msg)
        startup_errors["model_load"] = msg
        system = None

    # ===== Fallback: retrain if load failed =====
    if system is None and os.path.exists(DATASET_PATH):
        try:
            from meal_model import train_best_model
            print("[INFO] Training fallback model...")
            train_best_model(csv_path=DATASET_PATH, model_output_path=MODEL_PATH, cv=3)
            system = MealRecommendationSystem(model_path=MODEL_PATH, recipes_path=RECIPES_PATH)
            print("[OK] Model retrained successfully")
        except Exception as e:
            msg = f"Retrain failed: {str(e)}\n{traceback.format_exc()}"
            print(msg)
            startup_errors["model_retrain"] = msg
            system = None
    elif system is None:
        startup_errors["model_retrain"] = "nutrition_dataset.csv not found, cannot retrain."

    # ===== Chatbot =====
    try:
        with open(RECIPES_PATH, "r", encoding="utf-8") as f:
            recipes = json.load(f)
        chatbot = FoodChatbot(foods_data=recipes, recommendation_engine=system)
        print("[OK] Chatbot initialized")
    except Exception as e:
        msg = f"Chatbot failed: {str(e)}\n{traceback.format_exc()}"
        print(msg)
        startup_errors["chatbot"] = msg
        chatbot = None


# =========================
# Health Check
# =========================
@app.get("/", summary="Health Check")
def health():
    return {
        "status": "ok",
        "model_loaded": system is not None,
        "chatbot_loaded": chatbot is not None,
        "errors": startup_errors if startup_errors else None
    }


# =========================
# Recommend Meals
# =========================
@app.put(
    "/api/recommend-meals",
    summary="Get Meal Recommendations",
    description="Generates predicted nutritional targets and recommends meals based on user profile."
)
def recommend(profile: UserProfileInput):
    if system is None:
        raise HTTPException(
            status_code=503,
            detail=f"Model not loaded. Startup errors: {startup_errors}"
        )
    try:
        user = UserProfile(
            age=profile.age,
            gender=profile.gender.capitalize(),
            height_cm=profile.height,
            weight_kg=profile.weight,
            activity_level=profile.activityLevel,
            fitness_goal=profile.goal,
            dietary_preference=profile.dietType,
            allergies=profile.allergies,
            health_conditions=[profile.healthCondition],
            meals_per_day=profile.mealsPerDay,
            notes=profile.healthNotes,
            max_calories=profile.calories,
            max_protein=profile.proteins,
            max_carbs=profile.carbs,
            max_fats=profile.fats,
        )
        return system.recommend(user)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# =========================
# Chat
# =========================
@app.put(
    "/api/chat",
    summary="Chat with AI Nutrition Bot",
    description="Interact with the AI chatbot. Optionally provide user profile and previous recommendations as context."
)
def chat(req: ChatRequest):
    if chatbot is None:
        raise HTTPException(
            status_code=503,
            detail=f"Chatbot not initialized. Startup errors: {startup_errors}"
        )
    try:
        profile_dict = None
        if req.user_profile:
            p = req.user_profile
            profile_dict = {
                "gender": p.gender,
                "age": p.age,
                "height": p.height,
                "weight": p.weight,
                "allergies": p.allergies,
                "diet_type": p.dietType,
                "health_conditions": [p.healthCondition],
                "activity_level": p.activityLevel,
                "fitness_goal": p.goal,
                "meals_per_day": p.mealsPerDay,
                "notes": p.healthNotes,
                "max_calories": p.calories,
                "max_protein": p.proteins,
                "max_carbs": p.carbs,
                "max_fats": p.fats,
            }

        response = chatbot.respond(
            user_msg=req.message,
            user_profile=profile_dict,
            recommendations=req.recommendations
        )
        return {"response": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# =========================
# Run locally
# =========================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000)