"""
SkinDiseaseAI - Streamlit app with login/register, profile menu,
prediction history, and medicine suggestions.

Run:
    streamlit run app.py
"""

import json
import os

import streamlit as st
from PIL import Image

import config
import database as db
from medications import get_advice
from predict import predict_image
from recommendation import get_recommendation

st.set_page_config(page_title="SkinDiseaseAI", page_icon="🩺", layout="centered")
db.init_db()

# ---------------------------------------------------------------- styling
st.markdown(
    """
    <style>
    .block-container {padding-top: 2rem; max-width: 900px;}
    .brand {font-size: 1.8rem; font-weight: 700; color: #0f766e;}
    .hero {
        background: linear-gradient(135deg, #0f766e 0%, #14b8a6 100%);
        color: white; padding: 1.4rem 1.6rem; border-radius: 16px; margin-bottom: 1.2rem;
    }
    .hero h2 {margin: 0; color: white;}
    .hero p {margin: .3rem 0 0 0; opacity: .9;}
    .card {
        border: 1px solid #e5e7eb; border-radius: 14px; padding: 1rem 1.2rem;
        background: #ffffff10; margin-bottom: .8rem;
    }
    .doctor {
        background: #fef2f2; border: 1px solid #fca5a5; color: #991b1b;
        padding: 1rem 1.2rem; border-radius: 14px;
    }
    div.stButton > button {border-radius: 10px;}
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------- session
st.session_state.setdefault("user", None)
st.session_state.setdefault("view", "home")
st.session_state.setdefault("last_saved", None)


def logout():
    st.session_state.user = None
    st.session_state.view = "home"
    st.session_state.last_saved = None


# ---------------------------------------------------------------- auth page
def auth_page():
    st.caption("AI-assisted skin analysis. Log in or create an account to continue.")

    login_tab, register_tab = st.tabs(["🔐 Login", "📝 Register"])

    with login_tab:
        with st.form("login_form"):
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            submitted = st.form_submit_button("Log in", width="stretch")
        if submitted:
            user = db.authenticate(username, password)
            if user:
                st.session_state.user = user
                st.session_state.view = "home"
                st.rerun()
            else:
                st.error("Invalid username or password.")

    with register_tab:
        with st.form("register_form"):
            new_user = st.text_input("Choose a username")
            email = st.text_input("Email")
            pw1 = st.text_input("Password (min 6 characters)", type="password")
            pw2 = st.text_input("Confirm password", type="password")
            created = st.form_submit_button("Create account", width="stretch")
        if created:
            if pw1 != pw2:
                st.error("Passwords do not match.")
            else:
                ok, msg = db.create_user(new_user, email, pw1)
                (st.success if ok else st.error)(msg)


# ---------------------------------------------------------------- top bar
def top_bar(user=None):
    """Fixed header shown on every page. Brand is always at the same spot."""
    left, right = st.columns([3, 1])
    with left:
        st.markdown('<div class="brand">🩺 SkinDiseaseAI</div>', unsafe_allow_html=True)
    if user is None:
        st.divider()
        return
    with right:
        with st.popover(f"👤 {user['username']}", width="stretch"):
            st.caption(user["email"])
            if st.button("🏠 Analyze", width="stretch", key="nav_home"):
                st.session_state.view = "home"
                st.rerun()
            if st.button("🕘 Past data", width="stretch", key="nav_hist"):
                st.session_state.view = "history"
                st.rerun()
            if st.button("🚪 Sign out", width="stretch", key="nav_out"):
                logout()
                st.rerun()
    st.divider()


# ---------------------------------------------------------------- home page
def render_result(predictions):
    top_class, top_prob = predictions[0]
    advice = get_advice(top_class, top_prob)

    st.subheader("Prediction")
    if top_prob < 0.50:
        st.metric("Most likely condition", "Unknown", delta=f"low confidence ({top_prob * 100:.1f}%)", delta_color="off")
    else:
        st.metric("Most likely condition", top_class.replace("_", " "), delta=f"{top_prob * 100:.1f}% confidence")

    st.write("**Other possibilities:**")
    for name, prob in predictions[1:]:
        st.write(f"- {name.replace('_', ' ')}: {prob * 100:.1f}%")

    return advice, top_class


def home_page(user):
    st.markdown(
        f'<div class="hero"><h2>Welcome back, {user["username"]} 👋</h2>'
        "<p>Upload a clear photo of the affected skin area to get an analysis.</p></div>",
        unsafe_allow_html=True,
    )

    if not os.path.exists(config.BEST_MODEL_PATH):
        st.error(f"No trained model found at `{config.BEST_MODEL_PATH}`. Run `python train.py` first.")
        st.stop()

    uploaded = st.file_uploader("Upload a skin image", type=["jpg", "jpeg", "png"])
    if uploaded is None:
        st.info("Upload an image above to get started.")
        return

    image = Image.open(uploaded)
    col1, col2 = st.columns(2)
    with col1:
        st.image(image, caption="Uploaded image", width="stretch")

    with st.spinner("Analyzing image..."):
        predictions = predict_image(image, top_k=3)

    # Save once per uploaded file (Streamlit reruns the script on every click)
    key = f"{uploaded.name}-{uploaded.size}"
    if st.session_state.last_saved != key:
        db.save_prediction(user["id"], image, predictions)
        st.session_state.last_saved = key

    with col2:
        advice, top_class = render_result(predictions)

    st.divider()
    st.subheader("Suggested medicines & care")

    if advice["see_doctor"]:
        st.markdown(f'<div class="doctor">🩺 <b>Please consult a doctor.</b><br>{advice["message"]}</div>', unsafe_allow_html=True)
        if advice.get("prescribed"):
            st.markdown("**Treatments a doctor may prescribe** (information only, do not self-medicate):")
            for med in advice["prescribed"]:
                st.markdown(f'<div class="card">💊 {med}</div>', unsafe_allow_html=True)
    else:
        for med in advice["medicines"]:
            st.markdown(f'<div class="card">💊 {med}</div>', unsafe_allow_html=True)
        st.info(advice["message"])
        rec = get_recommendation(top_class)
        if rec["note"]:
            st.caption(rec["note"])

    st.caption(
        "⚠️ Student/personal ML project, not a medical device. It is not a substitute "
        "for professional diagnosis."
    )


# ---------------------------------------------------------------- history page
def history_page(user):
    st.subheader("🕘 Your past results")
    rows = db.get_history(user["id"])
    if not rows:
        st.info("No past analyses yet. Upload an image on the Analyze page.")
        return

    for r in rows:
        with st.container(border=True):
            c1, c2 = st.columns([1, 2])
            with c1:
                if r["image_path"] and os.path.exists(r["image_path"]):
                    st.image(r["image_path"], width="stretch")
            with c2:
                label = r["predicted_class"].replace("_", " ")
                if r["confidence"] < 0.50:
                    label = "Unknown (low confidence)"
                st.markdown(f"**{label}** - {r['confidence'] * 100:.1f}%")
                st.caption(r["created_at"][:19].replace("T", " "))
                for name, prob in json.loads(r["top3"] or "[]")[1:]:
                    st.write(f"- {name.replace('_', ' ')}: {prob * 100:.1f}%")


# ---------------------------------------------------------------- router
user = st.session_state.user
top_bar(user)  # same header on every page (login, register, analyze, history)
if user is None:
    auth_page()
else:
    if st.session_state.view == "history":
        history_page(user)
    else:
        home_page(user)