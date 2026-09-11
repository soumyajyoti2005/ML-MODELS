"""ui.py — PulseBloom · CardiaCare landing page (claymorphism edition).

Soft squishy clay shapes in light pink / light violet / soft creamy white,
a matching clay sidebar, and a day ⇄ night theme toggle in the top-right
corner of every page. Run: streamlit run ui.py
"""

import streamlit as st
import utils

st.set_page_config(
    page_title="PulseBloom · CardiaCare",
    page_icon="💓",
    layout="wide",
    initial_sidebar_state="expanded",
)


def side_brand():
    """Clay sidebar (shared implementation lives in utils)."""
    utils.side_brand()


def clay_topbar():
    """Clay topbar with right-corner theme toggle (shared impl lives in utils)."""
    return utils.clay_topbar()

def hero(theme):
    st.markdown('<div class="clay-divider" style="margin-top:0.6rem"></div>', unsafe_allow_html=True)
    left, right = st.columns([1.05, 0.95], gap="large")

    with left:
        st.markdown(
            '<span class="clay-kicker">🫀&nbsp;CLAYMORPH&nbsp;HEALTH&nbsp;STUDIO</span>',
            unsafe_allow_html=True,
        )
        st.markdown(
            '<div class="clay-hero-h1">Your heart,<br/>painted in <span class="hl">soft clay</span>.</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            '<p class="clay-hero-sub">A squishy-soft cardiovascular risk companion. Feed it your '
            "vitals and lab findings — it replays the exact pipeline from <b>main.ipynb</b> "
            "and gives a gentle, honest read-out in seconds.</p>",
            unsafe_allow_html=True,
        )
        mood = "moonlit" if theme == "night" else "daylight"
        st.markdown(
            f'<div style="margin:1rem 0 0.4rem"><span class="clay-ecg">'
            f'<img src="{utils.svg_uri("ecg_heartbeat.svg")}" alt="ECG pulse"/>'
            f'<span class="clay-ecg-label"><b>918</b> patients analysed · <b>{mood}</b> mode active</span></span></div>',
            unsafe_allow_html=True,
        )
        c1, c2, _ = st.columns([1.0, 1.0, 1.4])
        with c1:
            utils.safe_page_link("pages/1_💓_Predict.py", label="💓 Start Predict", key="cta_predict")
        with c2:
            utils.safe_page_link("pages/2_📈_Dashboard.py", label="📈 Insights", key="cta_insights")
        st.markdown(
            '<div style="margin-top:0.7rem">'
            '<span class="clay-value-badge">🔒 Private by design</span>'
            '<span class="clay-value-badge">⚡ Instant read-out</span>'
            '<span class="clay-value-badge">🧪 Notebook-true</span>'
            "</div>",
            unsafe_allow_html=True,
        )

    with right:
        st.markdown(
            f'<div class="clay-frame"><img src="{utils.svg_uri("hero_cardio.svg")}" alt="Clay heart scene"/>'
            f'<div class="clay-float"><span class="dot"></span>'
            f"{'Night garden calm · model armed' if theme == 'night' else 'Morning calm · model armed'}"
            f"</div></div>",
            unsafe_allow_html=True,
        )

def category_cards():
    st.markdown('<hr class="clay-divider"/>', unsafe_allow_html=True)
    st.markdown('<div class="clay-section">Explore the clay studio</div>', unsafe_allow_html=True)
    st.markdown(
        '<p class="clay-sub">Everything is hand-shaped — soft shadows, pill buttons, pastel glows.</p>',
        unsafe_allow_html=True,
    )

    cards = [
        ("icon_cardio.svg", "Risk Predict", "Vitals & labs → instant heart-risk read-out.", "pages/1_💓_Predict.py", "clay-cat-pink"),
        ("icon_vitals.svg", "Insights", "Four themed charts over the full cohort.", "pages/2_📈_Dashboard.py", "clay-cat-violet"),
        ("icon_labs.svg", "Notebook True", "Identical preprocessing to main.ipynb.", None, "clay-cat-violet"),
        ("icon_ecg.svg", "Made for Humans", "Gentle wording, zero jargon walls.", None, "clay-cat-pink"),
    ]
    cols = st.columns(4, gap="medium")
    for col, (ico, name, sub, link, cls) in zip(cols, cards):
        with col:
            img = f'<img class="clay-cat-ico" src="{utils.svg_uri(ico)}" alt="{name}"/>'
            body = (
                f'<div class="clay-cat {cls}">{img}'
                f'<div class="clay-cat-name">{name}</div>'
                f'<div class="clay-cat-sub">{sub}</div></div>'
            )
            st.markdown(body, unsafe_allow_html=True)
            if link and st.button(f"Open {name} →", key=f"open_{name}", use_container_width=True):
                st.switch_page(link)


def footer():
    st.markdown('<hr class="clay-divider"/>', unsafe_allow_html=True)
    st.markdown(
        '<div class="footer-trust">🛡️ Educational demo — not a medical device · '
        "always consult a certified cardiologist</div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="clay-footer">PulseBloom · CardiaCare — squishy claymorphism '
        "crafted with Streamlit, scikit-learn &amp; Plotly</div>",
        unsafe_allow_html=True,
    )


def main():
    theme = utils.apply_theme()
    side_brand()
    clay_topbar()
    hero(theme)
    category_cards()
    footer()


if __name__ == "__main__":
    main()