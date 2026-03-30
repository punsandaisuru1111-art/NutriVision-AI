import streamlit as st
import onnxruntime as ort
import numpy as np
import json
import requests
import os
import gdown
import cv2

# ── Page config ───────────────────────────────────────────
st.set_page_config(
    page_title="NutriVision AI",
    page_icon="🥗",
    layout="centered"
)

# ── Custom CSS ────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #f0f7f0; }

    /* ── Animated hero header ── */
    .hero-box {
        position: relative;
        overflow: hidden;
        background: linear-gradient(135deg, #1d9e75 0%, #0f6e56 60%, #085041 100%);
        padding: 36px 24px 28px;
        border-radius: 20px;
        margin-bottom: 28px;
        text-align: center;
    }
    .hero-box::before {
        content: "🍛🥥🦐🍚🫔🐟🥗🍗🫘🌿";
        position: absolute;
        top: -10px; left: 0;
        font-size: 36px;
        letter-spacing: 8px;
        opacity: 0.12;
        white-space: nowrap;
        animation: slideFood 18s linear infinite;
        width: 200%;
    }
    .hero-box::after {
        content: "🫓🥜🍋🌶️🧄🥬🫙🧅🌾🍳";
        position: absolute;
        bottom: -10px; left: -50%;
        font-size: 30px;
        letter-spacing: 10px;
        opacity: 0.10;
        white-space: nowrap;
        animation: slideFood 22s linear infinite reverse;
        width: 200%;
    }
    @keyframes slideFood {
        0%   { transform: translateX(0); }
        100% { transform: translateX(-50%); }
    }
    .hero-title {
        position: relative;
        color: white;
        font-size: 38px;
        font-weight: 700;
        margin: 0 0 6px;
        text-shadow: 0 2px 8px rgba(0,0,0,0.18);
    }
    .hero-sub {
        position: relative;
        color: #9fe1cb;
        font-size: 16px;
        margin: 0;
    }

    /* ── Health score ring ── */
    .score-wrap {
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        padding: 16px 0 8px;
    }
    .score-ring {
        position: relative;
        width: 120px;
        height: 120px;
    }
    .score-ring svg {
        transform: rotate(-90deg);
    }
    .score-ring .bg-ring {
        fill: none;
        stroke: #e0f0e8;
        stroke-width: 10;
    }
    .score-ring .fg-ring {
        fill: none;
        stroke-width: 10;
        stroke-linecap: round;
        transition: stroke-dashoffset 1.2s ease;
    }
    .score-num {
        position: absolute;
        top: 50%; left: 50%;
        transform: translate(-50%, -50%);
        font-size: 28px;
        font-weight: 700;
        color: #0f6e56;
    }
    .score-label {
        font-size: 13px;
        color: #666;
        margin-top: 6px;
        font-weight: 500;
    }

    /* ── Food card ── */
    .food-card {
        background: white;
        border-radius: 16px;
        padding: 20px;
        margin: 12px 0;
        border: 1.5px solid #e0f0e8;
        box-shadow: 0 2px 12px rgba(29,158,117,0.08);
    }

    /* ── Nutrition bars ── */
    .nutrition-label {
        font-size: 13px;
        color: #666;
        margin-bottom: 2px;
    }

    /* ── Status badges ── */
    .badge-safe {
        background: #e6f4ea; color: #1e7e34;
        border: 1.5px solid #1e7e34;
        padding: 6px 18px; border-radius: 20px;
        font-weight: bold; font-size: 15px;
        display: inline-block; margin-bottom: 10px;
    }
    .badge-caution {
        background: #fff8e1; color: #e65100;
        border: 1.5px solid #e65100;
        padding: 6px 18px; border-radius: 20px;
        font-weight: bold; font-size: 15px;
        display: inline-block; margin-bottom: 10px;
    }
    .badge-avoid {
        background: #fdecea; color: #c62828;
        border: 1.5px solid #c62828;
        padding: 6px 18px; border-radius: 20px;
        font-weight: bold; font-size: 15px;
        display: inline-block; margin-bottom: 10px;
    }

    /* ── Advice box ── */
    .advice-box {
        background: #f8fffe;
        border-left: 4px solid #1d9e75;
        padding: 14px 16px;
        border-radius: 0 12px 12px 0;
        margin-top: 10px;
        font-size: 15px;
        line-height: 1.6;
    }

    /* ── Pulse animation for spinner ── */
    .analyzing-box {
        text-align: center;
        padding: 20px;
    }
    .pulse-plate {
        font-size: 48px;
        display: inline-block;
        animation: pulse 1s ease-in-out infinite;
    }
    @keyframes pulse {
        0%, 100% { transform: scale(1);   opacity: 1; }
        50%       { transform: scale(1.2); opacity: 0.7; }
    }

    /* ── Tabs ── */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px; background: white;
        border-radius: 12px; padding: 6px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 10px; font-weight: 500; font-size: 15px;
    }

    /* ── Button ── */
    .stButton > button {
        background: linear-gradient(135deg, #1d9e75, #0f6e56);
        color: white; border: none; border-radius: 10px;
        padding: 10px 20px; font-size: 16px;
        font-weight: 600; width: 100%;
        transition: opacity 0.2s;
    }
    .stButton > button:hover { opacity: 0.88; }

    /* ── Fact chips ── */
    .fact-chip {
        display: inline-block;
        background: #f0f7f4;
        border: 1px solid #c8e6d9;
        border-radius: 20px;
        padding: 4px 14px;
        font-size: 13px;
        color: #0f6e56;
        margin: 3px 3px 3px 0;
    }
</style>
""", unsafe_allow_html=True)

# ── Sri Lankan foods ──────────────────────────────────────
SL_FOODS = {
    "🫓 Kottu Roti":             {"calories": 450, "protein_g": 18, "sugar_g": 4,  "sodium_mg": 780, "fat_g": 16},
    "🍛 Rice and Curry":         {"calories": 520, "protein_g": 22, "sugar_g": 3,  "sodium_mg": 620, "fat_g": 14},
    "🫔 Egg Hoppers":            {"calories": 180, "protein_g": 6,  "sugar_g": 2,  "sodium_mg": 210, "fat_g": 7},
    "🫔 Plain Hoppers":          {"calories": 120, "protein_g": 3,  "sugar_g": 1,  "sodium_mg": 150, "fat_g": 3},
    "🍝 String Hoppers":         {"calories": 150, "protein_g": 3,  "sugar_g": 1,  "sodium_mg": 180, "fat_g": 2},
    "🥥 Pol Sambol":             {"calories": 120, "protein_g": 2,  "sugar_g": 3,  "sodium_mg": 290, "fat_g": 10},
    "🫘 Dhal Curry":             {"calories": 180, "protein_g": 9,  "sugar_g": 4,  "sodium_mg": 480, "fat_g": 6},
    "🐟 Fish Curry":             {"calories": 220, "protein_g": 24, "sugar_g": 3,  "sodium_mg": 560, "fat_g": 10},
    "🍗 Chicken Curry":          {"calories": 310, "protein_g": 28, "sugar_g": 4,  "sodium_mg": 640, "fat_g": 16},
    "🫓 Egg Roti":               {"calories": 280, "protein_g": 10, "sugar_g": 2,  "sodium_mg": 340, "fat_g": 12},
    "🫓 Coconut Roti":           {"calories": 240, "protein_g": 5,  "sugar_g": 2,  "sodium_mg": 220, "fat_g": 11},
    "🌾 Pittu":                  {"calories": 200, "protein_g": 5,  "sugar_g": 1,  "sodium_mg": 160, "fat_g": 3},
    "🫘 Wade":                   {"calories": 160, "protein_g": 6,  "sugar_g": 1,  "sodium_mg": 280, "fat_g": 8},
    "🦐 Isso Wade":              {"calories": 190, "protein_g": 10, "sugar_g": 1,  "sodium_mg": 380, "fat_g": 10},
    "🍚 Kiri Bath":              {"calories": 320, "protein_g": 6,  "sugar_g": 8,  "sodium_mg": 120, "fat_g": 10},
    "🍱 Lamprais":               {"calories": 680, "protein_g": 30, "sugar_g": 5,  "sodium_mg": 890, "fat_g": 28},
    "🫘 Parippu Curry":          {"calories": 160, "protein_g": 8,  "sugar_g": 3,  "sodium_mg": 420, "fat_g": 5},
    "🌿 Gotukola Sambol":        {"calories": 60,  "protein_g": 3,  "sugar_g": 2,  "sodium_mg": 180, "fat_g": 3},
    "🧅 Seeni Sambol":           {"calories": 180, "protein_g": 3,  "sugar_g": 12, "sodium_mg": 340, "fat_g": 8},
    "🌿 Polos Curry":            {"calories": 140, "protein_g": 4,  "sugar_g": 3,  "sodium_mg": 380, "fat_g": 6},
    "🫙 Jackfruit Curry":        {"calories": 130, "protein_g": 3,  "sugar_g": 5,  "sodium_mg": 360, "fat_g": 5},
    "🦐 Prawn Curry":            {"calories": 200, "protein_g": 22, "sugar_g": 3,  "sodium_mg": 620, "fat_g": 9},
    "🦀 Crab Curry":             {"calories": 180, "protein_g": 20, "sugar_g": 2,  "sodium_mg": 580, "fat_g": 8},
    "🍗 Devilled Chicken":       {"calories": 340, "protein_g": 30, "sugar_g": 6,  "sodium_mg": 720, "fat_g": 18},
    "🍳 Fried Rice Sri Lankan":  {"calories": 380, "protein_g": 12, "sugar_g": 3,  "sodium_mg": 680, "fat_g": 12},
}

# ── Load model and data ───────────────────────────────────
@st.cache_resource
def load_model():
    model_path = "nutrivision_model.onnx"
    if not os.path.exists(model_path):
        st.info("Downloading model... please wait 1-2 minutes")
        file_id = "1PDSmmZfU96B5ntNQ8X775a4IwLZkRWx8"
        try:
            gdown.download(id=file_id, output=model_path, quiet=False, fuzzy=True)
        except Exception as e:
            import requests as req
            url = f"https://drive.google.com/uc?export=download&id={file_id}&confirm=t"
            r   = req.get(url, stream=True, timeout=120)
            with open(model_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=32768):
                    if chunk:
                        f.write(chunk)
    if not os.path.exists(model_path):
        st.error("Model download failed. Check Google Drive permissions.")
        st.stop()
    return ort.InferenceSession(model_path)

@st.cache_resource
def load_data():
    class_names  = open("nutrivision_save/classes.txt").read().splitlines()
    nutrition_db = json.load(open("nutrivision_save/nutrition_db.json"))
    guidelines   = json.load(open("nutrivision_save/medical_guidelines.json"))
    return class_names, nutrition_db, guidelines

session                               = load_model()
CLASS_NAMES, NUTRITION_DB, GUIDELINES = load_data()
OPENROUTER_API_KEY                    = os.environ.get("OPENROUTER_API_KEY", "")
FREE_MODELS = ["google/gemma-3-12b-it:free", "google/gemma-3-27b-it:free"]

# ── Health score calculator ───────────────────────────────
def calculate_health_score(nutrition, condition):
    score  = 100
    cal    = nutrition.get("calories",   0)
    sodium = nutrition.get("sodium_mg",  0)
    sugar  = nutrition.get("sugar_g",    0)
    fat    = nutrition.get("fat_g",      0)
    prot   = nutrition.get("protein_g",  0)

    # Base penalties
    if cal   > 600: score -= 20
    elif cal > 400: score -= 10
    if sodium > 800: score -= 25
    elif sodium > 500: score -= 12
    if sugar  > 15:  score -= 20
    elif sugar > 8:  score -= 10
    if fat    > 20:  score -= 15
    elif fat  > 12:  score -= 7

    # Protein bonus
    if prot > 20: score += 10
    elif prot > 10: score += 5

    # Condition-specific adjustments
    if condition == "diabetes":
        if sugar > 10: score -= 20
        if sugar > 20: score -= 15
    elif condition == "high_blood_pressure":
        if sodium > 600: score -= 20
        if sodium > 900: score -= 15
    elif condition == "vitamin_d_deficiency":
        if prot > 15: score += 8

    return max(0, min(100, score))

def score_color(score):
    if score >= 70: return "#1d9e75"
    if score >= 45: return "#e65100"
    return "#c62828"

def score_emoji(score):
    if score >= 70: return "✅"
    if score >= 45: return "⚠️"
    return "🚫"

# ── Health score ring HTML ────────────────────────────────
def health_score_html(score):
    r      = 50
    circ   = 2 * 3.14159 * r
    offset = circ * (1 - score / 100)
    color  = score_color(score)
    label  = "Excellent" if score >= 70 else "Moderate" if score >= 45 else "Avoid"
    return f"""
<div class="score-wrap">
  <div class="score-ring">
    <svg viewBox="0 0 120 120" width="120" height="120">
      <circle class="bg-ring" cx="60" cy="60" r="{r}"/>
      <circle class="fg-ring"
        cx="60" cy="60" r="{r}"
        stroke="{color}"
        stroke-dasharray="{circ:.1f}"
        stroke-dashoffset="{offset:.1f}"/>
    </svg>
    <div class="score-num" style="color:{color}">{score}</div>
  </div>
  <div class="score-label">Health score — {label}</div>
</div>
"""

# ── Fact chips ────────────────────────────────────────────
def fact_chips(nutrition):
    chips = []
    cal  = nutrition.get("calories",  0)
    prot = nutrition.get("protein_g", 0)
    fat  = nutrition.get("fat_g",     0)
    sug  = nutrition.get("sugar_g",   0)
    sod  = nutrition.get("sodium_mg", 0)

    if cal  < 200: chips.append("🟢 Low calorie")
    if prot > 20:  chips.append("💪 High protein")
    if fat  < 5:   chips.append("🟢 Low fat")
    if sug  < 5:   chips.append("🟢 Low sugar")
    if sod  < 300: chips.append("🟢 Low sodium")
    if sod  > 700: chips.append("🔴 High sodium")
    if fat  > 20:  chips.append("🔴 High fat")
    if sug  > 12:  chips.append("🔴 High sugar")

    html = "".join(f'<span class="fact-chip">{c}</span>' for c in chips)
    return f'<div style="margin: 8px 0 4px">{html}</div>'

# ── LLM advice ────────────────────────────────────────────
def get_advice(food_name, nutrition, condition, guidelines):
    prompt = f"""You are NutriVision AI, a friendly nutrition assistant.
Food: {food_name}
Calories : {nutrition.get("calories")} kcal
Protein  : {nutrition.get("protein_g")}g
Sugar    : {nutrition.get("sugar_g")}g
Sodium   : {nutrition.get("sodium_mg")}mg
Fat      : {nutrition.get("fat_g")}g
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
                headers={"Authorization": f"Bearer {OPENROUTER_API_KEY}",
                         "Content-Type": "application/json"},
                json={"model": m, "messages": [{"role": "user", "content": prompt}]},
                timeout=30
            )
            data = r.json()
            if "choices" in data:
                return data["choices"][0]["message"]["content"]
        except:
            continue

    sodium = nutrition.get("sodium_mg", 0)
    sugar  = nutrition.get("sugar_g",   0)
    cals   = nutrition.get("calories",  0)
    limits = guidelines.get("nutrient_limits", {})

    if condition == "high_blood_pressure":
        limit  = limits.get("sodium_mg_per_day", 1500)
        pct    = round(sodium / limit * 100)
        status = "AVOID" if pct > 50 else "CAUTION" if pct > 30 else "SAFE"
        return f"1. {status}\n2. Contains {sodium}mg sodium ({pct}% of your {limit}mg daily limit).\n3. Watch portion sizes carefully.\n4. Try gotukola sambol or a light vegetable curry instead."
    elif condition == "diabetes":
        limit  = limits.get("sugar_g_per_meal", 15)
        status = "AVOID" if sugar > limit*2 else "CAUTION" if sugar > limit else "SAFE"
        return f"1. {status}\n2. Contains {sugar}g sugar vs your {limit}g meal limit.\n3. Watch for hidden sugars in curries.\n4. Try string hoppers with dhal curry instead."
    return f"1. CAUTION\n2. This food has {cals} calories. Eat in moderation.\n3. Watch portion sizes carefully.\n4. Balance with vegetables and lean proteins."

# ── Nutrition bars ────────────────────────────────────────
def show_nutrition_bars(nutrition):
    cal = nutrition.get("calories",   0)
    pro = nutrition.get("protein_g",  0)
    fat = nutrition.get("fat_g",      0)
    sug = nutrition.get("sugar_g",    0)
    sod = nutrition.get("sodium_mg",  0)

    st.markdown("#### Nutrition breakdown")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown(f"<p class='nutrition-label'>🔥 Calories — {cal} kcal</p>", unsafe_allow_html=True)
        st.progress(min(cal / 800, 1.0))
        st.markdown(f"<p class='nutrition-label'>💪 Protein — {pro}g</p>", unsafe_allow_html=True)
        st.progress(min(pro / 50, 1.0))
        st.markdown(f"<p class='nutrition-label'>🧈 Fat — {fat}g</p>", unsafe_allow_html=True)
        st.progress(min(fat / 60, 1.0))
    with c2:
        st.markdown(f"<p class='nutrition-label'>🍬 Sugar — {sug}g</p>", unsafe_allow_html=True)
        st.progress(min(sug / 50, 1.0))
        st.markdown(f"<p class='nutrition-label'>🧂 Sodium — {sod}mg</p>", unsafe_allow_html=True)
        st.progress(min(sod / 2000, 1.0))

# ── Advice card ───────────────────────────────────────────
def show_advice_card(advice):
    if "AVOID" in advice:
        st.markdown('<span class="badge-avoid">🚫 AVOID</span>', unsafe_allow_html=True)
    elif "CAUTION" in advice:
        st.markdown('<span class="badge-caution">⚠️ CAUTION</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span class="badge-safe">✅ SAFE</span>', unsafe_allow_html=True)
    clean = advice.replace("1. AVOID","").replace("1. CAUTION","").replace("1. SAFE","").strip()
    st.markdown(f'<div class="advice-box">{clean}</div>', unsafe_allow_html=True)

# ── Vision model ──────────────────────────────────────────
def identify_food(image_array):
    img         = cv2.resize(image_array, (224, 224))
    img         = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img         = img.astype(np.float32) / 127.5 - 1.0
    img         = np.expand_dims(img, axis=0)
    input_name  = session.get_inputs()[0].name
    outputs     = session.run(None, {input_name: img})
    predictions = outputs[0][0]
    top_idx     = np.argmax(predictions)
    return CLASS_NAMES[top_idx], float(predictions[top_idx] * 100)

# ══════════════════════════════════════════════════════════
#  LAYOUT
# ══════════════════════════════════════════════════════════

# ── Animated hero header ──────────────────────────────────
st.markdown("""
<div class="hero-box">
  <div class="hero-title">🥗 NutriVision AI</div>
  <p class="hero-sub">Your personalized Sri Lankan food safety & nutrition assistant</p>
</div>
""", unsafe_allow_html=True)

# ── Tabs ──────────────────────────────────────────────────
tab1, tab2 = st.tabs(["🇱🇰  Sri Lankan Food", "🌍  International Food"])

# ══════════════════════════════════════════════════════════
#  TAB 1 — Sri Lankan
# ══════════════════════════════════════════════════════════
with tab1:
    st.markdown('<div class="food-card">', unsafe_allow_html=True)
    st.markdown("#### Select your dish")

    sl_food = st.selectbox(
        "What did you eat?",
        sorted(SL_FOODS.keys()),
        key="sl_food"
    )

    # Live nutrition preview
    preview = SL_FOODS[sl_food]
    p1, p2, p3, p4 = st.columns(4)
    p1.metric("🔥 Cal",    preview["calories"])
    p2.metric("💪 Protein", f'{preview["protein_g"]}g')
    p3.metric("🧈 Fat",    f'{preview["fat_g"]}g')
    p4.metric("🧂 Sodium", f'{preview["sodium_mg"]}mg')

    condition_sl = st.selectbox(
        "Your health condition",
        ["none", "diabetes", "high_blood_pressure", "vitamin_d_deficiency"],
        key="sl_cond"
    )

    if st.button("Get nutrition advice 🔍", key="sl_btn"):
        nutrition  = SL_FOODS[sl_food]
        guidelines = GUIDELINES.get(condition_sl, {})
        score      = calculate_health_score(nutrition, condition_sl)

        with st.spinner("Analyzing your food..."):
            advice = get_advice(sl_food, nutrition, condition_sl, guidelines)

        st.divider()

        # Score ring + food name side by side
        col_score, col_info = st.columns([1, 2])
        with col_score:
            st.markdown(health_score_html(score), unsafe_allow_html=True)
        with col_info:
            st.markdown(f"### {sl_food}")
            st.markdown(fact_chips(nutrition), unsafe_allow_html=True)
            em = score_emoji(score)
            st.markdown(
                f'<div style="margin-top:10px;font-size:14px;color:#555">'
                f'{em} Overall health score for your condition: '
                f'<b style="color:{score_color(score)}">{score}/100</b></div>',
                unsafe_allow_html=True
            )

        st.divider()
        show_nutrition_bars(nutrition)
        st.divider()
        st.markdown("#### Health advice")
        show_advice_card(advice)

    st.markdown('</div>', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════
#  TAB 2 — International
# ══════════════════════════════════════════════════════════
with tab2:
    st.markdown('<div class="food-card">', unsafe_allow_html=True)
    st.markdown("#### Upload a food photo")

    uploaded = st.file_uploader(
        "Take or upload a photo of your food",
        type=["jpg", "jpeg", "png"],
        key="intl_upload"
    )

    condition_intl = st.selectbox(
        "Your health condition",
        ["none", "diabetes", "high_blood_pressure", "vitamin_d_deficiency"],
        key="intl_cond"
    )

    if uploaded and st.button("Analyze photo 📸", key="intl_btn"):
        file_bytes  = np.asarray(bytearray(uploaded.read()), dtype=np.uint8)
        image_array = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        display_img = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)

        st.image(display_img, use_container_width=True, caption="Your food photo")

        with st.spinner("AI is analyzing your food..."):
            food_name, confidence = identify_food(image_array)
            nutrition     = NUTRITION_DB.get(food_name, {})
            condition_key = condition_intl.lower().replace(" ", "_")
            guidelines    = GUIDELINES.get(condition_key, {})
            advice        = get_advice(food_name, nutrition, condition_intl, guidelines)
            score         = calculate_health_score(nutrition, condition_intl)

        st.divider()

        col_score, col_info = st.columns([1, 2])
        with col_score:
            st.markdown(health_score_html(score), unsafe_allow_html=True)
        with col_info:
            st.markdown(f"### 🍽️ {food_name.replace('_',' ').title()}")
            st.caption(f"Detected with {confidence:.1f}% confidence")
            st.markdown(fact_chips(nutrition), unsafe_allow_html=True)

        st.divider()
        show_nutrition_bars(nutrition)
        st.divider()
        st.markdown("#### Health advice")
        show_advice_card(advice)

    elif not uploaded:
        st.info("📸 Upload a food photo to get instant nutrition analysis")

    st.markdown('</div>', unsafe_allow_html=True)

# ── Footer ────────────────────────────────────────────────
st.markdown("""
<div style="text-align:center;margin-top:32px;color:#aaa;font-size:13px">
    NutriVision AI — Built with TensorFlow, ONNX and Streamlit<br>
    <i>Not a substitute for professional medical advice</i>
</div>
""", unsafe_allow_html=True)
