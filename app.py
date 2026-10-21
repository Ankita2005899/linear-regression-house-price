"""House Price Predictor - multi-tab Streamlit app (run: streamlit run app.py)."""
import base64, io, json, math, requests
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

PROMPT = """Each photo is preceded by a text label giving its room type. Estimate the floor area of each room in square metres.
Return ONLY a JSON object: {"rooms": [{"room_type": "label", "area_sqm": number or null, "note": "short"}], "notes": "one sentence on what you could not tell"}.
List the rooms in the same order as the photos. Use null if the size cannot be judged. Never invent values."""
SQM = 10.7639
ROOMS = {"Bedroom": (3, 12.0), "Bathroom": (2, 4.0), "Living room": (1, 20.0), "Kitchen": (1, 9.0)}
CENTRES = {"Mumbai": (18.9388, 72.8354), "Pune": (18.5196, 73.8553), "Nagpur": (21.1458, 79.0882)}
for r, (n, sz) in ROOMS.items():
    st.session_state.setdefault(f"rn_{r}", n)
    st.session_state.setdefault(f"rsz_{r}", sz)

def prep(file):
    from PIL import Image
    im = Image.open(file).convert("RGB")
    im.thumbnail((1280, 1280))
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=85)
    return buf.getvalue()

def get_key():
    try:
        k = st.secrets.get("GEMINI_API_KEY")
    except Exception:
        k = None
    return k or st.session_state.get("user_key")

def parse_json(text):
    return json.loads(text[text.index("{"): text.rindex("}") + 1])

def analyse_rooms():
    s = st.session_state
    items = [(r, f) for r in ROOMS if s.get(f"rmode_{r}") == "Upload photo" for f in (s.get(f"ph_{r}") or [])][:12]
    key = get_key()
    if not items:
        s.ai = {"error": "Choose 'Upload photo' for a room and upload a photo first."}
    elif not key:
        s.ai = {"error": "No API key. Paste your Gemini key in the sidebar, or type the sizes instead."}
    elif s.uses >= 10:
        s.ai = {"error": "Analysis limit reached for this session."}
    else:
        try:
            from google import genai
            from google.genai import types
            parts = []
            for r, f in items:
                parts += [f"Photo of a {r}:", types.Part.from_bytes(data=prep(f), mime_type="image/jpeg")]
            parts.append(PROMPT)
            try:
                first = st.secrets.get("GEMINI_MODEL")
            except Exception:
                first = None
            client, err, text = genai.Client(api_key=key), None, None
            for model in [m for m in (first, "gemini-2.5-flash", "gemini-flash-latest") if m]:
                try:
                    text = client.models.generate_content(
                        model=model, contents=parts,
                        config=types.GenerateContentConfig(response_mime_type="application/json")).text
                    break
                except Exception as e:
                    err = e
            if text is None:
                raise err
            res = parse_json(text)
            sizes = {r: [] for r in ROOMS}
            for (r, _), room in zip(items, res.get("rooms", [])):
                if room.get("area_sqm"):
                    sizes[r].append(max(1.0, min(200.0, float(room["area_sqm"]))))
            for r in ROOMS:
                files = s.get(f"ph_{r}") or []
                s[f"rpx_{r}"], s[f"rsig_{r}"] = sizes[r], [(f.name, f.size) for f in files]
            s.uses += 1
            s.ai = res
        except Exception as e:
            s.ai = {"error": f"Photo analysis failed: {e}"}

def hav(a, b):
    p1, p2, dl = math.radians(a[0]), math.radians(b[0]), math.radians(b[1] - a[1])
    x = math.sin((p2 - p1) / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 6371 * 2 * math.asin(math.sqrt(x))

def locate():
    s = st.session_state
    try:
        r = requests.get("https://nominatim.openstreetmap.org/search", timeout=10,
                         params={"q": s.get("addr", ""), "format": "json", "limit": 1, "countrycodes": "in"},
                         headers={"User-Agent": "house-price-student-project"}).json()
        if not r:
            s.loc = {"error": "Address not found. Try adding the area, city and pincode."}
            return
        pt = (float(r[0]["lat"]), float(r[0]["lon"]))
        city = min(CENTRES, key=lambda c: hav(pt, CENTRES[c]))
        d = hav(pt, CENTRES[city])
        s.f_city, s.f_dist = city, round(min(d, 100.0), 1)
        s.loc = {"city": city, "km": d, "name": r[0].get("display_name", "")}
    except Exception as e:
        s.loc = {"error": f"Location lookup failed: {e}"}

def photo_sizes(r):
    s, files = st.session_state, st.session_state.get(f"ph_{r}") or []
    ok = files and s.get(f"rsig_{r}") == [(f.name, f.size) for f in files]
    return s.get(f"rpx_{r}", []) if ok else []

def room_total(r):
    s, n = st.session_state, st.session_state[f"rn_{r}"]
    if s[f"rmode_{r}"] == "Upload photo" and photo_sizes(r):
        v = photo_sizes(r)[:n]
        return sum(v) + max(0, n - len(v)) * sum(v) / len(v)
    return n * s[f"rsz_{r}"]

with st.sidebar:
    st.header("Photo analysis key (free)")
    st.caption("Only needed to read room sizes from photos. Kept for this session only.")
    st.text_input("Google Gemini API key", type="password", key="user_key")

st.title("🏠 House Price Predictor")
st.caption("Linear regression (Ridge) trained on 1,000 sample houses.")
t1, t2, t3, t4 = st.tabs(["🔮 Predict", "📊 Data Explorer", "📈 Model Performance", "ℹ️ About"])

with t1:
    st.subheader("1. House size")
    mode = st.radio("How do you want to give the size?", ["Type the total area", "Room by room (photo or size)"],
                    horizontal=True, key="size_mode")
    if mode.startswith("Type"):
        u1, u2 = st.columns(2)
        unit = u1.selectbox("Unit", ["square metres (sq m)", "square feet (sq ft)"], key="unit")
        total = u2.number_input("Total area", 10.0, 10000.0, 150.0 if "metres" in unit else 1600.0, step=5.0, key="total")
        area_sqft = total * SQM if "metres" in unit else total
        bed = st.number_input("Bedrooms", 1, 10, 3, key="bed_t")
        bath = st.number_input("Bathrooms", 1, 10, 2, key="bath_t")
    else:
        st.caption("For each room type give how many there are, then type the size of one room or upload one photo of it.")
        for r in ROOMS:
            a, b, c = st.columns([1, 1.3, 2])
            a.number_input(f"{r}s", 1 if r in ("Bedroom", "Bathroom") else 0, 10, key=f"rn_{r}")
            b.radio("Size from", ["Type size", "Upload photo"], key=f"rmode_{r}", horizontal=True)
            if st.session_state[f"rmode_{r}"] == "Upload photo":
                files = c.file_uploader(f"{r} photos (one per {r.lower()})", type=["jpg", "jpeg", "png"],
                                        accept_multiple_files=True, key=f"ph_{r}")
                v = photo_sizes(r)
                if files and v:
                    c.caption(f"Sizes read (sq m): {', '.join(f'{x:.1f}' for x in v)}. Rooms without a photo use the average.")
                elif files:
                    c.caption("Click 'Read sizes from photos' below.")
                if len(files) > st.session_state[f"rn_{r}"]:
                    c.warning(f"You uploaded {len(files)} photos but the count is {st.session_state[f'rn_{r}']}. Increase the count.")
            else:
                c.number_input(f"Size of one {r.lower()} (sq m)", 1.0, 200.0, step=0.5, key=f"rsz_{r}")
        st.button("🔍 Read sizes from photos", on_click=analyse_rooms)
        ai = st.session_state.get("ai")
        if ai and "error" in ai:
            st.error(ai["error"])
        elif ai:
            st.success("Photos analysed. The sizes read from the photos are used in the total. They are rough guesses, so type the sizes instead if you know them.")
            if ai.get("rooms"):
                st.dataframe(pd.DataFrame(ai["rooms"]), hide_index=True)
            if ai.get("notes"):
                st.info(ai["notes"])
        sqm = sum(room_total(r) for r in ROOMS)
        area_sqft, bed, bath = sqm * SQM, st.session_state["rn_Bedroom"], st.session_state["rn_Bathroom"]
    st.info(f"Area used by the model: **{area_sqft:,.0f} sq ft** ({area_sqft / SQM:,.0f} sq m), {bed} bedrooms, {bath} bathrooms")

    st.subheader("2. Location")
    lm = st.radio("How do you want to give the location?", ["Choose city and distance", "Type the address"], horizontal=True)
    if lm.startswith("Type"):
        st.text_input("Address (area, city, pincode)", key="addr")
        st.button("📍 Find location", on_click=locate)
        loc = st.session_state.get("loc")
        if loc and "error" in loc:
            st.error(loc["error"])
        elif loc:
            st.success(f"Nearest model city: {loc['city']}, about {loc['km']:.1f} km (straight line) from the centre. {loc['name']}")
            if loc["km"] > 60:
                st.warning("This address is far from Mumbai, Pune and Nagpur, so the estimate is not reliable.")
    l1, l2 = st.columns(2)
    city = l1.selectbox("City", CITIES, key="f_city")
    dist = l2.number_input("Distance from city centre (km)", 0.0, 100.0, step=0.5, key="f_dist")

    st.subheader("3. Age and currency")
    g1, g2, g3 = st.columns(3)
    age = g1.number_input("Age of house (years)", 0, 100, key="f_age")
    cur = g2.radio("Show price in", ["US dollars ($)", "Indian rupees (₹)"])
    rate = g3.number_input("Today's rate: ₹ for $1", 1.0, 1000.0, 85.0, step=0.1,
                           help="Type the current dollar rate, for example from Google.")
    if st.button("Predict price", type="primary"):
        row = [dict(area=area_sqft, bedrooms=bed, bathrooms=bath, age=age, distance_city=dist, city=city)]
        usd, rmse = float(predictor.predict(row)[0]), load_metrics().get("RMSE", 0)
        inr = "₹" in cur
        f = (lambda v: f"₹{v * rate:,.0f}") if inr else (lambda v: f"${v:,.0f}")
        st.metric("Estimated price", f(usd))
        st.caption((f"Likely range: {f(usd - rmse)} to {f(usd + rmse)}. Also: ${usd:,.0f} = ₹{usd * rate:,.0f} at ₹{rate:g} per $1. "
                    "The model's prices are treated as US dollars.").replace("$", "\\$"))
        d = load_data()
        for col, val in [("area", area_sqft), ("age", age), ("distance_city", dist)]:
            if not d[col].min() <= val <= d[col].max():
                st.warning(f"{col} = {val:,.0f} is outside the training data "
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

**About the photo option:** photos are used only to estimate the size of each room (sq m). Photo sizes are rough guesses, so correct them if you know the real size. Bedroom and bathroom counts, age and distance are typed by you.
A photo can show bedrooms, bathrooms and condition fairly well, but it **cannot measure area or distance**.
Area is only reliable if a floor plan shows it printed. Always check the values before predicting.

**Limits:** the training data is synthetic, so prices are illustrative, not real market prices.
""")