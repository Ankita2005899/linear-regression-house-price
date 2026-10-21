"""House Price Predictor - multi-tab Streamlit app (run: streamlit run app.py)."""
import base64, io, json
import pandas as pd
import streamlit as st
from src import config, predict as predictor

st.set_page_config(page_title="House Price Predictor", page_icon="🏠", layout="wide")
OUT, CITIES = config.OUT_DIR, ["Mumbai", "Pune", "Nagpur"]
DEFAULTS = dict(f_area=1800.0, f_bed=3, f_bath=2, f_age=5, f_dist=8.0, f_city="Pune")
for k, v in DEFAULTS.items():
    st.session_state.setdefault(k, v)
st.session_state.setdefault("uses", 0)

@st.cache_data
def load_data():
    return pd.read_csv(config.DATA_PATH)

def load_metrics():
    p = OUT / "best_metrics.json"
    return json.loads(p.read_text()) if p.exists() else {}

if not config.MODEL_PATH.exists():
    from src import train
    train.main()

PROMPT = """You are helping estimate inputs for a house-price model from photos.
Return ONLY a JSON object, no other text, with these keys:
"photos": list with one item per photo in order: {"room_type": "bedroom|bathroom|kitchen|living|exterior|floor_plan|other", "note": "short"},
"bedrooms": number of DISTINCT bedrooms clearly visible across photos, or null,
"bathrooms": number of DISTINCT bathrooms clearly visible across photos, or null,
"area_sqft": total built-up area in square feet, or null,
"area_source": "printed_on_image" (a floor plan or text shows the area; convert sq m to sq ft) or "rough_estimate" or "unknown",
"age_years": rough age of the building from its condition, or null,
"confidence": "low|medium|high",
"notes": "one or two sentences on what you could not tell".
Never invent values. Use null when the photos do not show it. Rooms not photographed are NOT counted."""

def prep(file):
    from PIL import Image
    im = Image.open(file).convert("RGB")
    im.thumbnail((1280, 1280))
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=85)
    return base64.standard_b64encode(buf.getvalue()).decode()

def get_key():
    try:
        k = st.secrets.get("GEMINI_API_KEY")
    except Exception:
        k = None
    return k or st.session_state.get("user_key")

def parse_json(text):
    return json.loads(text[text.index("{"): text.rindex("}") + 1])

def clamp(v, lo, hi, cast):
    return cast(max(lo, min(hi, float(v))))

def analyse():
    files = (st.session_state.get("photos") or [])[:6]
    key = get_key()
    if not files:
        st.session_state.ai = {"error": "Upload at least one photo first."}
        return
    if not key:
        st.session_state.ai = {"error": "No API key. Add one in the sidebar, or type the details manually."}
        return
    if st.session_state.uses >= 10:
        st.session_state.ai = {"error": "Analysis limit reached for this session."}
        return
    try:
        from google import genai
        from google.genai import types
        parts = [types.Part.from_bytes(data=base64.b64decode(prep(f)), mime_type="image/jpeg") for f in files]
        parts.append(PROMPT)
        try:
            first = st.secrets.get("GEMINI_MODEL")
        except Exception:
            first = None
        client, last_err, text = genai.Client(api_key=key), None, None
        for model in [m for m in (first, "gemini-2.5-flash", "gemini-flash-latest") if m]:
            try:
                r = client.models.generate_content(
                    model=model, contents=parts,
                    config=types.GenerateContentConfig(response_mime_type="application/json"))
                text = r.text
                break
            except Exception as e:
                last_err = e
        if text is None:
            raise last_err
        res = parse_json(text)
    except Exception as e:
        st.session_state.ai = {"error": f"Photo analysis failed: {e}"}
        return
    st.session_state.uses += 1
    st.session_state.ai = res
    s = st.session_state
    if res.get("area_sqft"): s.f_area = clamp(res["area_sqft"], 100, 10000, float)
    if res.get("bedrooms"): s.f_bed = clamp(res["bedrooms"], 1, 10, int)
    if res.get("bathrooms"): s.f_bath = clamp(res["bathrooms"], 1, 10, int)
    if res.get("age_years") is not None: s.f_age = clamp(res["age_years"], 0, 100, int)

with st.sidebar:
    st.header("Photo analysis key (free)")
    st.caption("Only needed for the photo option. The key is kept for this session only.")
    st.text_input("Google Gemini API key", type="password", key="user_key")

st.title("🏠 House Price Predictor")
st.caption("Linear regression (Ridge) trained on 1,000 sample houses.")
t1, t2, t3, t4 = st.tabs(["🔮 Predict", "📊 Data Explorer", "📈 Model Performance", "ℹ️ About"])

with t1:
    mode = st.radio("How do you want to give the house details?",
                    ["⌨️ I'll type the details", "📷 I have photos"], horizontal=True)
    if mode.startswith("📷"):
        st.file_uploader("Upload photos (house, bathroom, rooms, floor plan). Max 6.",
                         type=["jpg", "jpeg", "png"], accept_multiple_files=True, key="photos")
        st.button("🔍 Analyse photos", on_click=analyse)
        ai = st.session_state.get("ai")
        if ai and "error" in ai:
            st.error(ai["error"])
        elif ai:
            st.success(f"Photos analysed (confidence: {ai.get('confidence', 'unknown')}). "
                       "Check and correct the values below before predicting.")
            if ai.get("photos"):
                st.dataframe(pd.DataFrame(ai["photos"]), hide_index=True)
            if ai.get("area_source") != "printed_on_image":
                st.warning("Area was not printed on any photo, so it is only a rough guess. "
                           "If you know the real area, type it below.")
            if ai.get("notes"):
                st.info(ai["notes"])
    c1, c2 = st.columns(2)
    area = c1.number_input("Area (sq ft)", 100.0, 10000.0, step=50.0, key="f_area")
    bed = c1.number_input("Bedrooms", 1, 10, key="f_bed")
    bath = c1.number_input("Bathrooms", 1, 10, key="f_bath")
    age = c2.number_input("Age of house (years)", 0, 100, key="f_age")
    dist = c2.number_input("Distance from city centre (km)", 0.0, 100.0, step=0.5, key="f_dist")
    city = c2.selectbox("City", CITIES, key="f_city")
    if st.button("Predict price", type="primary"):
        row = [dict(area=area, bedrooms=bed, bathrooms=bath, age=age, distance_city=dist, city=city)]
        price, rmse = float(predictor.predict(row)[0]), load_metrics().get("RMSE", 0)
        st.metric("Estimated price", f"{price:,.0f}")
        st.caption(f"Likely range: {price - rmse:,.0f} to {price + rmse:,.0f} (typical test error ±{rmse:,.0f}).")
        d = load_data()
        for col, val in [("area", area), ("age", age), ("distance_city", dist)]:
            if not d[col].min() <= val <= d[col].max():
                st.warning(f"{col} = {val} is outside the training data "
                           f"({d[col].min():.0f} to {d[col].max():.0f}), so this estimate is less reliable.")

with t2:
    d = load_data()
    st.write(f"{d.shape[0]} rows, {d.shape[1]} columns")
    st.dataframe(d.head(50))
    st.subheader("Summary statistics")
    st.dataframe(d.describe().T)
    st.subheader("Missing values")
    st.bar_chart(d.isna().sum())
    col = st.selectbox("Plot a numeric column against price", [c for c in config.NUMERIC])
    st.scatter_chart(d, x=col, y="price", color="city")

with t3:
    m = load_metrics()
    if m:
        a, b, c, e = st.columns(4)
        a.metric("Best model", m["best_model"].title())
        b.metric("R²", f"{m['R2']:.3f}")
        c.metric("RMSE", f"{m['RMSE']:,.0f}")
        e.metric("MAE", f"{m['MAE']:,.0f}")
    p = OUT / "model_comparison.csv"
    if p.exists():
        cmp = pd.read_csv(p)
        st.subheader("Model comparison")
        st.dataframe(cmp, hide_index=True)
        st.bar_chart(cmp.set_index("model")["RMSE"])
    for f, cap in [("ridge_diagnostics.png", "Actual vs predicted, residuals and Q-Q plot"),
                   ("ridge_coefficients.png", "Which features push the price up or down"),
                   ("correlation_heatmap.png", "Correlation between features"),
                   ("distributions.png", "Feature distributions"),
                   ("gradient_descent_loss.png", "From-scratch gradient descent loss")]:
        if (OUT / f).exists():
            st.subheader(cap)
            st.image(str(OUT / f))

with t4:
    st.markdown("""
**Pipeline:** data → EDA → cleaning → scaling and one-hot encoding → five models compared
(Linear, Ridge, Lasso, ElasticNet, Polynomial+Ridge) with 5-fold cross-validation → best model saved → this app.

**About the photo option:** the model needs area, bedrooms, bathrooms, age, distance and city.
A photo can show bedrooms, bathrooms and condition fairly well, but it **cannot measure area or distance**.
Area is only reliable if a floor plan shows it printed. Always check the values before predicting.

**Limits:** the training data is synthetic, so prices are illustrative, not real market prices.
""")
