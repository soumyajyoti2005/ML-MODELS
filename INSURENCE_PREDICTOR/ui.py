"""ui.py — Suncover landing page (Streamlit entry point).

Streamlit's native `pages/` folder provides the 3-page navigation:
  ui.py              -> this landing page
  pages/1_🔮_Predict.py
  pages/2_📊_Dashboard.py
"""

import base64
import os

import streamlit as st

import utils

ROOT = os.path.dirname(os.path.abspath(__file__))
ILI = os.path.join(ROOT, "assets", "illustrations")
CSS = os.path.join(ROOT, "assets", "style.css")

st.set_page_config(
    page_title="Suncover — Smart Insurance Charge Predictor",
    page_icon="☀️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

utils.load_css(CSS)


def svg_uri(name):
    """Base64 data-URI for an SVG in assets/illustrations (no server needed)."""
    with open(os.path.join(ILI, name), "rb") as f:
        return "data:image/svg+xml;base64," + base64.b64encode(f.read()).decode("ascii")


def hero():
    left, right = st.columns([7, 5], gap="large", vertical_alignment="center")

    with left:
        st.markdown(
            '<span class="nb-tag" style="transform:rotate(-3deg)">AI-POWERED</span> '
            '<span class="nb-eyebrow">SUNCOVER // SMART INSURANCE CHARGE PREDICTOR</span>',
            unsafe_allow_html=True,
        )
        st.markdown(
            '<h1 class="hero-title">Your medical bill,<br>predicted '
            '<span class="hl">before</span> it hits.</h1>',
            unsafe_allow_html=True,
        )
        st.markdown(
            '<p class="hero-sub">LINEAR REGRESSION / 1,337 RECORDS / '
            "100% PRIVATE &amp; LOCAL — NO DATA LEAVES THIS MACHINE</p>",
            unsafe_allow_html=True,
        )

        st.markdown('<div class="nb-cta-zone">', unsafe_allow_html=True)
        st.page_link("pages/1_🔮_Predict.py", label="Get My Estimate →")
        st.page_link("pages/2_📊_Dashboard.py", label="View Insights", icon="📊")
        st.markdown("</div>", unsafe_allow_html=True)

    with right:
        st.markdown(
            f'<div class="nb-frame">'
            f'<img src="{svg_uri("hero.svg")}" alt="Suncover hero artwork — sun, family and shield"/>'
            f"</div>",
            unsafe_allow_html=True,
        )


def feature_row():
    st.markdown('<div class="nb-divider"></div>', unsafe_allow_html=True)
    st.markdown('<div class="nb-section">WHY SUNCOVER</div>', unsafe_allow_html=True)
    st.markdown(
        '<p class="nb-section-sub">FOUR REASONS TO TRUST A MODEL THAT LIVES ON YOUR SIDE.</p>',
        unsafe_allow_html=True,
    )

    items = [
        ("icon_bolt.svg", "Instant Estimate", "ONE CLICK · ZERO WAITING", -2),
        ("icon_graph.svg", "Data-Driven Model", "HAND-BUILT IN main.ipynb", 0),
        ("icon_eye.svg", "Visual Insights", "PLOTLY CHARTS · NO FRILLS", 2),
        ("icon_lock.svg", "Private & Local", "COMPUTES ON YOUR MACHINE", 0),
    ]

    cols = st.columns(4, gap="medium")
    for col, (icon, label, line, rot) in zip(cols, items):
        with col:
            st.markdown(
                f'<div class="nb-feature" style="transform:rotate({rot}deg)">'
                f'<img src="{svg_uri(icon)}" alt="{label}"/>'
                f'<div class="f-label">{label}</div>'
                f'<div class="f-line">{line}</div>'
                f"</div>",
                unsafe_allow_html=True,
            )


def footer():
    st.markdown(
        '<div class="nb-divider"></div>'
        '<div class="nb-footer">SUNCOVER v1.0 · trained from '
        "<b>main.ipynb</b> · Linear Regression · "
        '<span class="nb-stamp">v1.0 · LINEAR REGRESSION</span></div>',
        unsafe_allow_html=True,
    )


hero()
feature_row()
footer()