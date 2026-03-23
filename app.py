import streamlit as st
import onnxruntime as ort
import numpy as np
import json
import requests
import os
import gdown
from PIL import Image

# ── Download model from Google Drive ─────────────────────
@st.cache_resource
def load_model():
    model_path = "nutrivision_model.onnx"
    if not os.path.exists(model_path):
        st.info("Downloading model for first time... please wait")
        gdown.download(
            "https://drive.google.com/uc?id=1PDSmmZfU96B5ntNQ8X775a4IwLZkRWx8",
            model_path,
            quiet=False
        )
    session = ort.InferenceSession(model_path)
    return session

# ── Load data files ───────────────────────────────────────
@st.cache_resource
def load_data():
    class_names  = open("nutrivision_save/classes.txt").read().splitlines()
    nutrition_db = json.load(open("nutrivision_save/nutrition_db.json"))
    guidelines   = json.load(open("nutrivision_save/medical_guidelines.json"))
    return class_names, nutrition_db, guidelines

# ── Page config ───────────────────────────────────────────
st.set_page_config(
    page_title="NutriVision AI",
    page_icon="🥗",
    layout="centered"
)

# ── Header ────────────────────────────────────────────────
st.markdown("""
<div style="background:#1d9e75;padding:20px;border-radius:12px;margin-bottom:24px">
  <h1 style="color:white;margin:0;font-size:28px">NutriVision AI</h1>
  <p style="color:#9fe1cb;margin:4px 0 0">Personalized food safety and nutrition assistant</p>
</div>
""", unsafe_allow_html=True)

# ── Load everything ───────────────────────────────────────
session = load_model()
CLASS_NAMES, NUTRITION_DB, GUIDELINES = load_data()

OPENROUTER_API_KEY = st.secrets["OPENROUTER_API_KEY"]

FREE_MODELS = [
    "google/gemma-3-12b-it:free",
    "google/gemma-3-27b-it:free",
    "deepseek/deepseek-r1-0528-qwen3-8b:free",
]

# ── Functions ─────────────────────────────────────────────
def preprocess_image(image):
    img = image.resize((224, 224))
    img = np.array(img).astype(np.float32)
    # MobileNetV2 preprocessing
    img = img / 127.5 - 1.0
    img = np.expand_dims(img, axis=0)
    return img

def identify_food(image):
    img        = preprocess_image(image)
    input_name = session.get_inputs()[0].name
    outputs    = session.run(None, {input_name: img})
    predictions = outputs[0][0]
    top_idx     = np.argmax(predictions)
    confidence  = float(predictions[top_idx] * 100)
    return CLASS_NAMES[top_idx], confidence

def get_advice(food_name, nutrition, condition, guidelines):
    prompt = f"""You are NutriVision AI, a friendly nutrition assistant.
Food: {food_name.replace("_"," ").title()}
Calories: {nutrition.get("calories")} kcal | Protein: {nutrition.get("protein_g")}g
Sugar: {nutrition.get("sugar_g")}g | Sodium: {nutrition.get("sodium_mg")}mg | Fat: {nutrition.get("fat_g")}g
User condition: {condition}
Guidelines: {json.dumps(guidelines)}
Reply with:
1. SAFE / CAUTION / AVOID
2. Why (2-3 sentences)
3. One thing to watch out for
4. One healthier alternative
Keep it friendly and under 150 words."""

    for m in FREE_MODELS:
        try:
            r = requests.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": m,
                    "messages": [{"role": "user", "content": prompt}]
                },
                timeout=30
            )
            data = r.json()
            if "choices" in data:
                return data["choices"][0]["message"]["content"]
        except:
            continue

    # Rule-based fallback
    sodium = nutrition.get("sodium_mg", 0)
    sugar  = nutrition.get("sugar_g", 0)
    cals   = nutrition.get("calories", 0)
    limits = guidelines.get("nutrient_limits", {})

    if condition == "high_blood_pressure":
        limit  = limits.get("sodium_mg_per_day", 1500)
        pct    = round(sodium / limit * 100)
        status = "AVOID" if pct > 50 else "CAUTION" if pct > 30 else "SAFE"
        return f"""1. {status}
2. This food contains {sodium}mg sodium ({pct}% of your {limit}mg daily limit).
3. Watch out for portion size — extra servings multiply sodium intake fast.
4. Try grilled vegetables or a salad with low-sodium dressing instead."""

    elif condition == "diabetes":
        limit  = limits.get("sugar_g_per_meal", 15)
        status = "AVOID" if sugar > limit*2 else "CAUTION" if sugar > limit else "SAFE"
        return f"""1. {status}
2. This food contains {sugar}g of sugar vs your {limit}g meal limit.
3. Watch out for hidden sugars in sauces and dressings.
4. Try a protein-rich meal with vegetables and whole grains instead."""

    return f"""1. CAUTION
2. This food has {cals} calories. Eat in moderation.
3. Watch portion sizes carefully.
4. Balance with vegetables and lean proteins."""

# ── UI ────────────────────────────────────────────────────
col1, col2 = st.columns(2)

with col1:
    st.subheader("Upload food photo")
    uploaded = st.file_uploader("", type=["jpg","jpeg","png"])

    condition = st.selectbox(
        "Your health condition",
        ["diabetes", "high_blood_pressure", "vitamin_d_deficiency", "none"]
    )

    analyze_btn = st.button("Analyze food", use_container_width=True)

with col2:
    if uploaded and analyze_btn:
        image = Image.open(uploaded).convert("RGB")
        st.image(image, use_container_width=True)

        with st.spinner("Analyzing your food..."):
            food_name, confidence = identify_food(image)
            nutrition             = NUTRITION_DB.get(food_name, {})
            condition_key         = condition.lower().replace(" ", "_")
            guidelines            = GUIDELINES.get(condition_key, {})
            advice                = get_advice(food_name, nutrition, condition, guidelines)

        st.markdown(f"### {food_name.replace('_',' ').title()}")
        st.caption(f"Confidence: {confidence:.1f}%")

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Calories", nutrition.get("calories", "—"))
        m2.metric("Sodium",   f"{nutrition.get('sodium_mg','—')}mg")
        m3.metric("Sugar",    f"{nutrition.get('sugar_g','—')}g")
        m4.metric("Protein",  f"{nutrition.get('protein_g','—')}g")

        if "AVOID" in advice:
            st.error(advice)
        elif "CAUTION" in advice:
            st.warning(advice)
        else:
            st.success(advice)

    elif not uploaded:
        st.info("Upload a food photo on the left to get started")
