"""Shared presentation for the document workspace and research lab."""
from html import escape
from pathlib import Path

import streamlit as st


def apply_theme():
    st.html(Path(__file__).with_name("theme.css"))


def brand():
    st.html('<div class="brand"><span class="brand-mark">R</span>'
            '<div><strong>RAGShield</strong><span>DOCUMENTS &amp; DISCOVERY</span></div></div>')


def hero(eyebrow: str, title: str, description: str):
    st.html(f'<section class="workspace-hero"><div class="eyebrow">{escape(eyebrow)}</div>'
            f'<h1>{escape(title)}</h1><p>{escape(description)}</p></section>')


def step_card(number: str, title: str, description: str):
    with st.container(border=True):
        st.html(f'<div class="step-number">{escape(number)}</div>')
        st.markdown(f"**{title}**")
        st.caption(description)
