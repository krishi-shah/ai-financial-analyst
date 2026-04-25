"""
Streamlit UI Module
Web interface for the AI Financial Analyst RAG system.
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from typing import Dict
from datetime import datetime
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from retrieval.rag_pipeline import FinancialRAGPipeline
from sentiment.sentiment_analyzer import FinancialSentimentAnalyzer
from data_ingestion.news_scraper import fetch_financial_news, get_company_news
from data_ingestion.earnings_call_parser import parse_earnings_transcript, chunk_transcript
from data_ingestion.reports_parser import parse_pdf_report, chunk_report

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Design tokens
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ACCENT = "#00D4AA"
ACCENT_DIM = "#00D4AA30"
BG_DARK = "#0A0E14"
BG_CARD = "#111827"
BG_ELEVATED = "#1A1F2E"
BORDER = "#1E2D3D"
TEXT_PRIMARY = "#E8EDF2"
TEXT_SECONDARY = "#8899AA"
TEXT_MUTED = "#556677"
POSITIVE = "#00D4AA"
NEGATIVE = "#FF5F6D"
NEUTRAL = "#6C7A8D"
WARNING = "#F0C040"


CUSTOM_CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

/* ── Base ── */
html, body, [class*="css"] {{
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
}}

#MainMenu {{ visibility: hidden; }}
footer {{ visibility: hidden; }}
header {{ visibility: hidden; }}

.stApp {{
    background: {BG_DARK};
}}

/* ── Keyframes ── */
@keyframes fadeInUp {{
    from {{ opacity: 0; transform: translateY(12px); }}
    to {{ opacity: 1; transform: translateY(0); }}
}}
@keyframes pulse-glow {{
    0%, 100% {{ opacity: 1; }}
    50% {{ opacity: 0.5; }}
}}
@keyframes shimmer {{
    0% {{ background-position: -200% 0; }}
    100% {{ background-position: 200% 0; }}
}}
@keyframes slideInLeft {{
    from {{ opacity: 0; transform: translateX(-16px); }}
    to {{ opacity: 1; transform: translateX(0); }}
}}

/* ── Sidebar ── */
section[data-testid="stSidebar"] {{
    background: linear-gradient(180deg, #0D1220 0%, #111827 100%);
    border-right: 1px solid {BORDER};
}}

/* ── Tab bar ── */
.stTabs [data-baseweb="tab-list"] {{
    background: transparent;
    border-bottom: 1px solid {BORDER};
    gap: 0;
    padding: 0 4px;
}}
.stTabs [data-baseweb="tab"] {{
    color: {TEXT_SECONDARY};
    border-radius: 6px 6px 0 0;
    padding: 10px 24px;
    font-size: 0.82rem;
    font-weight: 500;
    letter-spacing: 0.02em;
    transition: all 0.2s ease;
    border-bottom: 2px solid transparent;
}}
.stTabs [aria-selected="true"] {{
    color: {ACCENT} !important;
    border-bottom: 2px solid {ACCENT} !important;
    background: {ACCENT}08 !important;
}}
.stTabs [data-baseweb="tab"]:hover {{
    color: {TEXT_PRIMARY};
    background: {BG_ELEVATED}80;
}}

/* ── Metric cards ── */
[data-testid="stMetric"] {{
    background: {BG_CARD};
    border: 1px solid {BORDER};
    border-left: 3px solid {ACCENT};
    border-radius: 8px;
    padding: 16px 20px;
    animation: fadeInUp 0.4s ease-out;
}}
[data-testid="stMetricLabel"] {{
    color: {TEXT_SECONDARY} !important;
    font-size: 0.7rem !important;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    font-weight: 500;
}}
[data-testid="stMetricValue"] {{
    color: {TEXT_PRIMARY} !important;
    font-size: 1.5rem !important;
    font-weight: 700;
    font-family: 'JetBrains Mono', monospace !important;
}}

/* ── Glass panel ── */
.glass-panel {{
    background: {BG_CARD};
    border: 1px solid {BORDER};
    border-radius: 12px;
    padding: 24px;
    animation: fadeInUp 0.5s ease-out;
}}

/* ── Answer box ── */
.answer-box {{
    background: {BG_CARD};
    border: 1px solid {BORDER};
    border-left: 4px solid {ACCENT};
    border-radius: 0 10px 10px 0;
    padding: 20px 24px;
    margin: 12px 0 24px 0;
    color: {TEXT_PRIMARY};
    line-height: 1.8;
    font-size: 0.92rem;
    animation: slideInLeft 0.5s ease-out;
}}

/* ── Source chip ── */
.source-chip {{
    display: inline-block;
    background: {BG_DARK};
    border: 1px solid {BORDER};
    border-radius: 6px;
    padding: 3px 10px;
    font-size: 0.72rem;
    font-family: 'JetBrains Mono', monospace;
    color: {ACCENT};
    margin: 2px 4px 2px 0;
    transition: border-color 0.2s;
}}
.source-chip:hover {{
    border-color: {ACCENT};
}}

/* ── Verdict blocks ── */
.verdict-positive {{
    background: linear-gradient(90deg, {POSITIVE}15 0%, transparent 100%);
    border-left: 4px solid {POSITIVE};
    border-radius: 0 10px 10px 0;
    padding: 16px 24px;
    display: flex;
    align-items: center;
    gap: 12px;
    margin-bottom: 16px;
    animation: slideInLeft 0.4s ease-out;
}}
.verdict-positive .verdict-icon {{
    width: 36px; height: 36px; border-radius: 50%;
    background: {POSITIVE}20;
    display: flex; align-items: center; justify-content: center;
    font-size: 1.1rem;
}}
.verdict-positive .verdict-text {{
    color: {POSITIVE};
    font-size: 1.1rem;
    font-weight: 700;
    letter-spacing: 0.06em;
    text-transform: uppercase;
}}
.verdict-negative {{
    background: linear-gradient(90deg, {NEGATIVE}15 0%, transparent 100%);
    border-left: 4px solid {NEGATIVE};
    border-radius: 0 10px 10px 0;
    padding: 16px 24px;
    display: flex;
    align-items: center;
    gap: 12px;
    margin-bottom: 16px;
    animation: slideInLeft 0.4s ease-out;
}}
.verdict-negative .verdict-icon {{
    width: 36px; height: 36px; border-radius: 50%;
    background: {NEGATIVE}20;
    display: flex; align-items: center; justify-content: center;
    font-size: 1.1rem;
}}
.verdict-negative .verdict-text {{
    color: {NEGATIVE};
    font-size: 1.1rem;
    font-weight: 700;
    letter-spacing: 0.06em;
    text-transform: uppercase;
}}
.verdict-neutral {{
    background: linear-gradient(90deg, {NEUTRAL}15 0%, transparent 100%);
    border-left: 4px solid {NEUTRAL};
    border-radius: 0 10px 10px 0;
    padding: 16px 24px;
    display: flex;
    align-items: center;
    gap: 12px;
    margin-bottom: 16px;
    animation: slideInLeft 0.4s ease-out;
}}
.verdict-neutral .verdict-icon {{
    width: 36px; height: 36px; border-radius: 50%;
    background: {NEUTRAL}20;
    display: flex; align-items: center; justify-content: center;
    font-size: 1.1rem;
}}
.verdict-neutral .verdict-text {{
    color: {NEUTRAL};
    font-size: 1.1rem;
    font-weight: 700;
    letter-spacing: 0.06em;
    text-transform: uppercase;
}}

/* ── News card ── */
.news-card {{
    background: {BG_CARD};
    border: 1px solid {BORDER};
    border-radius: 10px;
    padding: 20px;
    margin-bottom: 12px;
    transition: border-color 0.2s ease, transform 0.15s ease;
    animation: fadeInUp 0.5s ease-out;
}}
.news-card:hover {{
    border-color: {ACCENT}60;
    transform: translateY(-1px);
}}
.news-card-title {{
    font-size: 0.95rem;
    font-weight: 600;
    color: {TEXT_PRIMARY};
    margin-bottom: 8px;
    line-height: 1.4;
}}
.news-card-meta {{
    font-size: 0.72rem;
    color: {TEXT_SECONDARY};
    margin-bottom: 10px;
    display: flex;
    align-items: center;
    gap: 8px;
}}
.news-card-desc {{
    font-size: 0.84rem;
    color: #BCC8D4;
    line-height: 1.65;
}}
.source-badge {{
    display: inline-block;
    background: {ACCENT}12;
    border: 1px solid {ACCENT}40;
    border-radius: 4px;
    padding: 2px 8px;
    font-size: 0.65rem;
    color: {ACCENT};
    text-transform: uppercase;
    letter-spacing: 0.06em;
    font-weight: 500;
}}

/* ── Status rows ── */
.status-row {{
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 10px 12px;
    border-bottom: 1px solid {BORDER}80;
    font-size: 0.82rem;
    color: {TEXT_PRIMARY};
    transition: background 0.15s;
}}
.status-row:hover {{
    background: {BG_ELEVATED}40;
}}
.status-row:last-child {{ border-bottom: none; }}
.dot-green  {{ width: 8px; height: 8px; border-radius: 50%; background: {POSITIVE}; flex-shrink: 0; animation: pulse-glow 2s ease-in-out infinite; }}
.dot-yellow {{ width: 8px; height: 8px; border-radius: 50%; background: {WARNING}; flex-shrink: 0; }}
.dot-red    {{ width: 8px; height: 8px; border-radius: 50%; background: {NEGATIVE}; flex-shrink: 0; }}
.status-label {{ color: {TEXT_SECONDARY}; font-size: 0.72rem; font-family: 'JetBrains Mono', monospace; }}

/* ── Section label ── */
.section-label {{
    font-size: 0.68rem;
    font-weight: 600;
    color: {TEXT_SECONDARY};
    text-transform: uppercase;
    letter-spacing: 0.14em;
    margin: 24px 0 10px 0;
    padding-bottom: 6px;
    border-bottom: 1px solid {BORDER};
}}

/* ── Hero stat ── */
.hero-stat {{
    background: {BG_CARD};
    border: 1px solid {BORDER};
    border-radius: 12px;
    padding: 20px;
    text-align: center;
    animation: fadeInUp 0.5s ease-out;
    transition: border-color 0.2s;
}}
.hero-stat:hover {{
    border-color: {ACCENT}50;
}}
.hero-stat-value {{
    font-size: 2rem;
    font-weight: 800;
    color: {ACCENT};
    font-family: 'JetBrains Mono', monospace;
    line-height: 1;
    margin-bottom: 4px;
}}
.hero-stat-label {{
    font-size: 0.68rem;
    color: {TEXT_SECONDARY};
    text-transform: uppercase;
    letter-spacing: 0.1em;
    font-weight: 500;
}}
.hero-stat-detail {{
    font-size: 0.72rem;
    color: {TEXT_MUTED};
    margin-top: 4px;
}}

/* ── Pipeline step ── */
.pipeline {{
    display: flex;
    align-items: stretch;
    gap: 0;
    margin: 16px 0;
}}
.pipeline-step {{
    flex: 1;
    text-align: center;
    padding: 16px 12px;
    background: {BG_CARD};
    border: 1px solid {BORDER};
    position: relative;
    transition: border-color 0.2s;
}}
.pipeline-step:first-child {{ border-radius: 10px 0 0 10px; }}
.pipeline-step:last-child {{ border-radius: 0 10px 10px 0; }}
.pipeline-step:not(:last-child)::after {{
    content: '';
    position: absolute;
    right: -1px;
    top: 50%;
    transform: translateY(-50%);
    width: 0;
    height: 0;
    border-top: 8px solid transparent;
    border-bottom: 8px solid transparent;
    border-left: 8px solid {ACCENT}40;
    z-index: 1;
}}
.pipeline-step:hover {{
    border-color: {ACCENT}50;
}}
.pipeline-step-num {{
    font-size: 0.6rem;
    color: {ACCENT};
    font-weight: 700;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    margin-bottom: 6px;
}}
.pipeline-step-title {{
    font-size: 0.85rem;
    font-weight: 600;
    color: {TEXT_PRIMARY};
    margin-bottom: 4px;
}}
.pipeline-step-desc {{
    font-size: 0.72rem;
    color: {TEXT_SECONDARY};
    line-height: 1.4;
}}

/* ── Expanders ── */
details {{
    border: 1px solid {BORDER} !important;
    border-left: 3px solid {BORDER} !important;
    border-radius: 0 8px 8px 0 !important;
    background: {BG_CARD} !important;
    margin-bottom: 8px !important;
    transition: border-color 0.2s !important;
}}
details[open] {{
    border-left-color: {ACCENT} !important;
}}
summary {{
    font-size: 0.84rem !important;
    color: #BCC8D4 !important;
    font-weight: 500 !important;
}}

/* ── Buttons ── */
.stButton > button[kind="primary"],
.stButton > button {{
    border-radius: 8px;
    font-weight: 600;
    letter-spacing: 0.03em;
    transition: all 0.2s ease;
    font-size: 0.84rem;
}}
.stButton > button[kind="primary"] {{
    background: {ACCENT};
    color: {BG_DARK};
    border: none;
}}
.stButton > button[kind="primary"]:hover {{
    opacity: 0.9;
    background: {ACCENT};
    color: {BG_DARK};
    transform: translateY(-1px);
}}

/* ── Text inputs ── */
.stTextInput > div > div > input,
.stTextArea > div > div > textarea {{
    background: {BG_CARD} !important;
    border: 1px solid {BORDER} !important;
    border-radius: 8px !important;
    color: {TEXT_PRIMARY} !important;
    font-size: 0.88rem !important;
    transition: border-color 0.2s !important;
}}
.stTextInput > div > div > input:focus,
.stTextArea > div > div > textarea:focus {{
    border-color: {ACCENT} !important;
}}

/* ── Dataframe ── */
[data-testid="stDataFrame"] {{
    border: 1px solid {BORDER};
    border-radius: 8px;
    overflow: hidden;
}}

/* ── File uploader ── */
[data-testid="stFileUploader"] {{
    border: 2px dashed {BORDER} !important;
    border-radius: 10px !important;
    background: {BG_CARD} !important;
    transition: border-color 0.2s !important;
}}
[data-testid="stFileUploader"]:hover {{
    border-color: {ACCENT}60 !important;
}}

/* ── Footer ── */
.app-footer {{
    margin-top: 48px;
    padding: 16px 0;
    border-top: 1px solid {BORDER};
    text-align: center;
    animation: fadeInUp 0.6s ease-out;
}}
.app-footer-text {{
    font-size: 0.72rem;
    color: {TEXT_MUTED};
    letter-spacing: 0.03em;
}}
.app-footer-text a {{
    color: {ACCENT};
    text-decoration: none;
}}

/* ── Header ── */
.app-header {{
    padding: 8px 0 20px 0;
    animation: fadeInUp 0.4s ease-out;
}}
.app-header-title {{
    font-size: 1.8rem;
    font-weight: 800;
    color: {TEXT_PRIMARY};
    letter-spacing: -0.03em;
    line-height: 1.15;
    display: flex;
    align-items: center;
    gap: 12px;
}}
.app-header-accent {{
    color: {ACCENT};
}}
.app-header-sub {{
    font-size: 0.82rem;
    color: {TEXT_SECONDARY};
    margin-top: 6px;
    font-weight: 400;
    letter-spacing: 0.01em;
}}
.api-badge {{
    display: inline-flex;
    align-items: center;
    gap: 6px;
    border-radius: 20px;
    padding: 5px 14px;
    font-size: 0.7rem;
    font-weight: 500;
    letter-spacing: 0.03em;
    transition: opacity 0.2s;
}}
.api-badge:hover {{ opacity: 0.8; }}
.api-badge-ok {{
    background: {POSITIVE}12;
    border: 1px solid {POSITIVE}30;
    color: {POSITIVE};
}}
.api-badge-warn {{
    background: {WARNING}12;
    border: 1px solid {WARNING}30;
    color: {WARNING};
}}

/* ── Sidebar branded ── */
.sidebar-brand {{
    padding: 12px 0 20px 0;
    border-bottom: 1px solid {BORDER};
    margin-bottom: 16px;
}}
.sidebar-brand-name {{
    font-size: 0.95rem;
    font-weight: 700;
    color: {TEXT_PRIMARY};
    display: flex;
    align-items: center;
    gap: 8px;
}}
.sidebar-brand-version {{
    font-size: 0.62rem;
    color: {TEXT_MUTED};
    font-family: 'JetBrains Mono', monospace;
    margin-top: 2px;
}}
.sidebar-section-title {{
    font-size: 0.62rem;
    font-weight: 600;
    color: {ACCENT};
    text-transform: uppercase;
    letter-spacing: 0.14em;
    margin: 20px 0 8px 0;
}}

/* ── Probability bar ── */
.prob-bar-container {{
    display: flex;
    align-items: center;
    gap: 10px;
    margin-bottom: 8px;
    animation: slideInLeft 0.5s ease-out;
}}
.prob-bar-label {{
    font-size: 0.75rem;
    color: {TEXT_SECONDARY};
    text-transform: capitalize;
    width: 70px;
    text-align: right;
    font-weight: 500;
}}
.prob-bar-track {{
    flex: 1;
    height: 10px;
    background: {BG_DARK};
    border-radius: 5px;
    overflow: hidden;
    border: 1px solid {BORDER};
}}
.prob-bar-fill {{
    height: 100%;
    border-radius: 5px;
    transition: width 0.6s ease-out;
}}
.prob-bar-value {{
    font-size: 0.72rem;
    color: {TEXT_PRIMARY};
    font-family: 'JetBrains Mono', monospace;
    width: 48px;
    font-weight: 500;
}}

/* ── Query search bar ── */
.search-container {{
    display: flex;
    align-items: center;
    gap: 8px;
    margin-bottom: 8px;
}}
</style>
"""


def initialize_session_state():
    if 'rag_pipeline' not in st.session_state:
        st.session_state.rag_pipeline = None
    if 'sentiment_analyzer' not in st.session_state:
        st.session_state.sentiment_analyzer = None
    if 'sample_data_loaded' not in st.session_state:
        st.session_state.sample_data_loaded = False


def load_sample_data():
    if st.session_state.sample_data_loaded:
        return
    st.session_state.rag_pipeline = FinancialRAGPipeline()
    st.session_state.sentiment_analyzer = FinancialSentimentAnalyzer()
    sample_chunks = [
        {'text': 'Apple reported Q4 2023 revenue of $94.8 billion, up 1% year-over-year, driven by strong iPhone sales.',
         'company': 'Apple', 'quarter': 'Q4 2023', 'source': 'earnings_call', 'type': 'earnings'},
        {'text': 'Tesla delivered 484,507 vehicles in Q3 2023, exceeding expectations and showing strong demand for electric vehicles.',
         'company': 'Tesla', 'quarter': 'Q3 2023', 'source': 'earnings_call', 'type': 'earnings'},
        {'text': 'Microsoft Azure revenue grew 29% year-over-year in the latest quarter, driven by increased cloud adoption.',
         'company': 'Microsoft', 'quarter': 'Q3 2023', 'source': 'earnings_call', 'type': 'earnings'},
        {'text': 'Amazon Web Services reported operating income of $7.0 billion, up 12% year-over-year.',
         'company': 'Amazon', 'quarter': 'Q3 2023', 'source': 'earnings_call', 'type': 'earnings'},
        {'text': 'The Federal Reserve raised interest rates by 0.25% to combat inflation, affecting tech stock valuations.',
         'company': 'Market', 'quarter': 'Q3 2023', 'source': 'news', 'type': 'news'}
    ]
    embedded = st.session_state.rag_pipeline.embedder.embed_document_chunks(sample_chunks)
    st.session_state.rag_pipeline.build_index(embedded)
    st.session_state.sample_data_loaded = True


def _get_index_stats():
    rag = st.session_state.rag_pipeline
    if not rag or not rag.chunks:
        return 0, [], [], []
    chunks = rag.chunks
    companies = sorted(set(c.get('company', '?') for c in chunks))
    quarters = sorted(set(c.get('quarter', '?') for c in chunks))
    sources = sorted(set(c.get('source', '?') for c in chunks))
    return len(chunks), companies, quarters, sources


def _sentiment_verdict(sentiment: str) -> str:
    s = sentiment.lower()
    icon_map = {'positive': ('verdict-positive', '+'), 'negative': ('verdict-negative', '-'), 'neutral': ('verdict-neutral', '~')}
    css_class, icon = icon_map.get(s, ('verdict-neutral', '~'))
    return f'''<div class="{css_class}">
  <div class="verdict-icon">{icon}</div>
  <div class="verdict-text">{s}</div>
</div>'''


def _prob_bar_html(label: str, value: float, color: str) -> str:
    pct = max(0, min(100, value * 100))
    return f'''<div class="prob-bar-container">
  <div class="prob-bar-label">{label}</div>
  <div class="prob-bar-track">
    <div class="prob-bar-fill" style="width:{pct}%;background:{color};"></div>
  </div>
  <div class="prob-bar-value">{pct:.1f}%</div>
</div>'''


def _make_gauge_chart(value: float, title: str) -> go.Figure:
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value * 100,
        number={'suffix': '%', 'font': {'size': 28, 'color': TEXT_PRIMARY, 'family': 'JetBrains Mono'}},
        title={'text': title, 'font': {'size': 13, 'color': TEXT_SECONDARY}},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 0, 'tickcolor': BG_DARK, 'dtick': 25,
                     'tickfont': {'size': 9, 'color': TEXT_MUTED}},
            'bar': {'color': ACCENT, 'thickness': 0.3},
            'bgcolor': BG_CARD,
            'borderwidth': 1,
            'bordercolor': BORDER,
            'steps': [
                {'range': [0, 33], 'color': f'{NEGATIVE}18'},
                {'range': [33, 66], 'color': f'{NEUTRAL}18'},
                {'range': [66, 100], 'color': f'{POSITIVE}18'},
            ],
        }
    ))
    fig.update_layout(
        height=200, margin=dict(l=20, r=20, t=40, b=10),
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        font={'color': TEXT_PRIMARY}
    )
    return fig


def _make_prob_chart(probs: dict) -> go.Figure:
    labels = [k.title() for k in probs.keys()]
    values = list(probs.values())
    colors = [POSITIVE if 'pos' in k else NEGATIVE if 'neg' in k else NEUTRAL for k in probs.keys()]

    fig = go.Figure(go.Bar(
        x=values, y=labels, orientation='h',
        marker=dict(color=colors, line=dict(width=0)),
        text=[f'{v:.1%}' for v in values],
        textposition='outside',
        textfont=dict(size=12, color=TEXT_PRIMARY, family='JetBrains Mono'),
    ))
    fig.update_layout(
        height=160, margin=dict(l=0, r=40, t=8, b=8),
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(range=[0, 1.15], showgrid=False, showticklabels=False, zeroline=False),
        yaxis=dict(showgrid=False, tickfont=dict(size=12, color=TEXT_SECONDARY)),
        bargap=0.35,
    )
    return fig


def _render_news_card(article: dict, idx: int):
    source = article.get('source', 'Unknown')
    published = article.get('publishedAt', '')
    if published and 'T' in published:
        published = published.split('T')[0]
    title = article.get('title', f'Article {idx}')
    desc = article.get('description', '')
    content = article.get('content', '')

    st.markdown(f"""<div class="news-card">
  <div class="news-card-title">{title}</div>
  <div class="news-card-meta">
    <span class="source-badge">{source}</span>
    <span>{published}</span>
  </div>
  <div class="news-card-desc">{desc}</div>
  {"<div class='news-card-desc' style='margin-top:8px;color:" + TEXT_MUTED + ";font-size:0.75rem;'>" + content[:350] + "...</div>" if content else ""}
</div>""", unsafe_allow_html=True)


def main():
    st.set_page_config(
        page_title="AI Financial Analyst",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

    initialize_session_state()
    load_sample_data()

    from config import OPENAI_API_KEY, NEWS_API_KEY
    openai_ok = bool(OPENAI_API_KEY and OPENAI_API_KEY.strip() not in ("", "your_openai_api_key_here", "OPENAI_API_KEY"))
    newsapi_ok = bool(NEWS_API_KEY and NEWS_API_KEY.strip() not in ("", "your_newsapi_key_here", "NEWS_API_KEY"))

    chunk_count, companies, quarters, sources = _get_index_stats()

    # ── Header ──────────────────────────────────────────────────────────────────
    h1, h2 = st.columns([3, 1])
    with h1:
        st.markdown(f'''<div class="app-header">
  <div class="app-header-title">
    <svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="{ACCENT}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="22 12 18 12 15 21 9 3 6 12 2 12"/></svg>
    <span>AI Financial <span class="app-header-accent">Analyst</span></span>
  </div>
  <div class="app-header-sub">Retrieval-Augmented Generation &middot; FinBERT Sentiment &middot; Real-time News Intelligence</div>
</div>''', unsafe_allow_html=True)
    with h2:
        badges = []
        for label, ok in [("OpenAI", openai_ok), ("NewsAPI", newsapi_ok)]:
            dot_color = POSITIVE if ok else WARNING
            cls = "api-badge api-badge-ok" if ok else "api-badge api-badge-warn"
            badges.append(f'<span class="{cls}"><span style="width:6px;height:6px;border-radius:50%;background:{dot_color};display:inline-block;"></span>{label}</span>')
        st.markdown(f'<div style="display:flex;justify-content:flex-end;gap:8px;padding-top:16px;">{"".join(badges)}</div>', unsafe_allow_html=True)

    # ── Hero dashboard row ──────────────────────────────────────────────────────
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f'''<div class="hero-stat">
  <div class="hero-stat-value">{chunk_count}</div>
  <div class="hero-stat-label">Indexed Chunks</div>
  <div class="hero-stat-detail">RAG knowledge base</div>
</div>''', unsafe_allow_html=True)
    with c2:
        st.markdown(f'''<div class="hero-stat">
  <div class="hero-stat-value">{len(companies)}</div>
  <div class="hero-stat-label">Companies</div>
  <div class="hero-stat-detail">{", ".join(companies[:3])}{"..." if len(companies) > 3 else ""}</div>
</div>''', unsafe_allow_html=True)
    with c3:
        st.markdown(f'''<div class="hero-stat">
  <div class="hero-stat-value">{len(quarters)}</div>
  <div class="hero-stat-label">Quarters</div>
  <div class="hero-stat-detail">{", ".join(quarters[:3])}</div>
</div>''', unsafe_allow_html=True)
    with c4:
        active = sum([openai_ok, newsapi_ok, chunk_count > 0])
        st.markdown(f'''<div class="hero-stat">
  <div class="hero-stat-value">{active}/3</div>
  <div class="hero-stat-label">Services Online</div>
  <div class="hero-stat-detail">RAG, LLM, News</div>
</div>''', unsafe_allow_html=True)

    st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)

    # ── Sidebar ──────────────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown(f'''<div class="sidebar-brand">
  <div class="sidebar-brand-name">
    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="{ACCENT}" stroke-width="2.5"><polyline points="22 12 18 12 15 21 9 3 6 12 2 12"/></svg>
    Financial Analyst
  </div>
  <div class="sidebar-brand-version">v2.0 &middot; RAG + FinBERT</div>
</div>''', unsafe_allow_html=True)

        st.markdown(f'<div class="sidebar-section-title">System Status</div>', unsafe_allow_html=True)

        idx_dot = 'dot-green' if chunk_count > 0 else 'dot-yellow'
        oai_dot = 'dot-green' if openai_ok else 'dot-yellow'
        nws_dot = 'dot-green' if newsapi_ok else 'dot-yellow'

        st.markdown(f'''
<div class="status-row"><div class="{idx_dot}"></div><span>RAG Index</span><span class="status-label" style="margin-left:auto">{chunk_count} chunks</span></div>
<div class="status-row"><div class="{oai_dot}"></div><span>OpenAI</span><span class="status-label" style="margin-left:auto">{"active" if openai_ok else "not set"}</span></div>
<div class="status-row"><div class="{nws_dot}"></div><span>NewsAPI</span><span class="status-label" style="margin-left:auto">{"active" if newsapi_ok else "not set"}</span></div>
''', unsafe_allow_html=True)

        if openai_ok:
            st.markdown(f'<div style="margin-top:10px;padding:8px 12px;background:{BG_ELEVATED};border-radius:6px;font-size:0.72rem;color:{TEXT_SECONDARY};border-left:3px solid {ACCENT}40;">Quota errors use enhanced fallback.</div>', unsafe_allow_html=True)

        st.markdown(f'<div class="sidebar-section-title">Coverage</div>', unsafe_allow_html=True)
        if companies:
            st.markdown(f'<div style="font-size:0.65rem;color:{TEXT_MUTED};margin-bottom:4px;text-transform:uppercase;letter-spacing:0.1em;">Companies</div>', unsafe_allow_html=True)
            st.markdown(" ".join(f'<span class="source-chip">{c}</span>' for c in companies), unsafe_allow_html=True)
        if quarters:
            st.markdown(f'<div style="font-size:0.65rem;color:{TEXT_MUTED};margin:12px 0 4px 0;text-transform:uppercase;letter-spacing:0.1em;">Quarters</div>', unsafe_allow_html=True)
            st.markdown(" ".join(f'<span class="source-chip">{q}</span>' for q in quarters), unsafe_allow_html=True)

        st.markdown(f'<div class="sidebar-section-title">Session</div>', unsafe_allow_html=True)
        st.markdown(f'<div style="font-size:0.72rem;color:{TEXT_MUTED};font-family:JetBrains Mono,monospace;">{datetime.now().strftime("%b %d, %Y %H:%M")}</div>', unsafe_allow_html=True)

    # ── Tabs ─────────────────────────────────────────────────────────────────────
    tab1, tab2, tab3, tab4 = st.tabs(["  Query  ", "  News  ", "  Sentiment  ", "  Data Sources  "])

    # ╔═══════════════════════════════════════════════════════════════════════════╗
    # ║  TAB 1 — QUERY                                                          ║
    # ╚═══════════════════════════════════════════════════════════════════════════╝
    with tab1:
        st.markdown('<div class="section-label">Ask anything about your financial data</div>', unsafe_allow_html=True)

        q_col1, q_col2 = st.columns([6, 1])
        with q_col1:
            query = st.text_input("", placeholder="What was Apple's revenue in Q4 2023?", label_visibility="collapsed",
                                  help="Ask about earnings, performance, trends, or any indexed financial data.")
        with q_col2:
            search_clicked = st.button("Search", type="primary", use_container_width=True, key="search_btn")

        if search_clicked:
            if query:
                with st.spinner("Retrieving and generating answer..."):
                    response = st.session_state.rag_pipeline.query(query, top_k=3)

                    st.markdown('<div class="section-label" style="margin-top:16px;">Answer</div>', unsafe_allow_html=True)
                    st.markdown(f'<div class="answer-box">{response["answer"]}</div>', unsafe_allow_html=True)

                    if response['sources']:
                        st.markdown('<div class="section-label">Retrieved Sources</div>', unsafe_allow_html=True)
                        rows = []
                        for src in response['sources']:
                            rows.append({
                                'Company': src.get('company', '?'),
                                'Quarter': src.get('quarter', '?'),
                                'Type': src.get('source', '?'),
                                'Similarity': round(src['similarity_score'], 3),
                                'Excerpt': src['text'][:140] + '...'
                            })
                        st.dataframe(
                            pd.DataFrame(rows),
                            use_container_width=True,
                            hide_index=True,
                            column_config={
                                "Similarity": st.column_config.ProgressColumn("Similarity", min_value=0, max_value=1, format="%.3f"),
                            }
                        )

                    st.markdown('<div class="section-label" style="margin-top:16px;">Answer Sentiment</div>', unsafe_allow_html=True)
                    sent = st.session_state.sentiment_analyzer.analyze_sentiment(response['answer'])
                    st.markdown(_sentiment_verdict(sent['sentiment']), unsafe_allow_html=True)

                    sc1, sc2 = st.columns([1, 2])
                    with sc1:
                        st.plotly_chart(_make_gauge_chart(sent['confidence'], 'Confidence'), use_container_width=True, config={'displayModeBar': False})
                    with sc2:
                        st.plotly_chart(_make_prob_chart(sent['class_probabilities']), use_container_width=True, config={'displayModeBar': False})
            else:
                st.warning("Please enter a question.")

    # ╔═══════════════════════════════════════════════════════════════════════════╗
    # ║  TAB 2 — NEWS                                                           ║
    # ╚═══════════════════════════════════════════════════════════════════════════╝
    with tab2:
        st.markdown('<div class="section-label">Live Financial News</div>', unsafe_allow_html=True)

        nc1, nc2 = st.columns([5, 1])
        with nc1:
            news_query = st.text_input("", placeholder="e.g., Apple earnings, Tesla deliveries, Fed interest rates...",
                                       label_visibility="collapsed", key="news_q")
        with nc2:
            fetch_clicked = st.button("Fetch", type="primary", use_container_width=True, key="fetch_btn")

        if fetch_clicked:
            if news_query:
                with st.spinner("Fetching articles..."):
                    articles = fetch_financial_news(news_query, page_size=10)
                    if articles:
                        st.markdown(f'<div style="font-size:0.78rem;color:{ACCENT};margin:8px 0 16px 0;font-weight:500;">{len(articles)} articles found</div>', unsafe_allow_html=True)
                        left, right = st.columns(2)
                        for i, art in enumerate(articles):
                            with (left if i % 2 == 0 else right):
                                _render_news_card(art, i + 1)
                    else:
                        st.warning("No articles found. Check your NewsAPI key.")
            else:
                st.warning("Enter a search query.")

        st.markdown('<div class="section-label" style="margin-top:32px;">Sample Articles</div>', unsafe_allow_html=True)
        sample_news = [
            {'title': 'Apple Reports Strong Q4 2023 Earnings', 'source': 'Financial Times',
             'publishedAt': '2023-10-26', 'description': 'Apple exceeded analyst expectations with revenue growth of 1% year-over-year, driven by record iPhone and Services revenue.'},
            {'title': 'Tesla Vehicle Deliveries Surge in Q3', 'source': 'Reuters',
             'publishedAt': '2023-10-18', 'description': 'Tesla delivered 484,507 vehicles globally in Q3 2023, beating estimates and reinforcing strong EV demand.'},
            {'title': 'Microsoft Azure Cloud Revenue Grows 29%', 'source': 'Bloomberg',
             'publishedAt': '2023-10-24', 'description': 'Microsoft reports strong cloud growth driven by enterprise AI adoption and increased Azure consumption.'},
            {'title': 'Federal Reserve Holds Rates Steady', 'source': 'CNBC',
             'publishedAt': '2023-11-01', 'description': 'The Fed pauses rate hikes while signaling potential further tightening if inflation remains elevated.'},
        ]
        sl, sr = st.columns(2)
        for i, art in enumerate(sample_news):
            with (sl if i % 2 == 0 else sr):
                _render_news_card(art, i + 1)

    # ╔═══════════════════════════════════════════════════════════════════════════╗
    # ║  TAB 3 — SENTIMENT                                                      ║
    # ╚═══════════════════════════════════════════════════════════════════════════╝
    with tab3:
        st.markdown('<div class="section-label">FinBERT Sentiment Analysis</div>', unsafe_allow_html=True)

        sentiment_text = st.text_area(
            "", placeholder="Paste any financial text here...\ne.g., The company reported strong quarterly results with revenue growth exceeding expectations.",
            height=120, label_visibility="collapsed"
        )

        if st.button("Analyze", type="primary", key="analyze_btn"):
            if sentiment_text:
                with st.spinner("Running FinBERT..."):
                    result = st.session_state.sentiment_analyzer.analyze_sentiment(sentiment_text)

                    st.markdown(_sentiment_verdict(result['sentiment']), unsafe_allow_html=True)

                    gc, pc = st.columns([1, 2])
                    with gc:
                        st.plotly_chart(
                            _make_gauge_chart(result['confidence'], 'Confidence'),
                            use_container_width=True, config={'displayModeBar': False}
                        )
                    with pc:
                        st.markdown(f'<div class="section-label" style="margin-top:0;">Class Probabilities</div>', unsafe_allow_html=True)
                        probs = result['class_probabilities']
                        color_map = {'positive': POSITIVE, 'negative': NEGATIVE, 'neutral': NEUTRAL}
                        for cls, prob in probs.items():
                            st.markdown(_prob_bar_html(cls, prob, color_map.get(cls, NEUTRAL)), unsafe_allow_html=True)

                    mc1, mc2 = st.columns(2)
                    with mc1:
                        st.metric("Model", result['model_used'])
                    with mc2:
                        st.metric("Sentiment", result['sentiment'].title())
            else:
                st.warning("Please enter text to analyze.")

        st.markdown('<div class="section-label" style="margin-top:36px;">Quick Samples</div>', unsafe_allow_html=True)
        samples = [
            ("The company reported strong quarterly results with revenue growth exceeding expectations.", "positive"),
            ("Investors are concerned about the declining profit margins and increased competition.", "negative"),
            ("The market outlook remains uncertain due to macroeconomic factors.", "neutral"),
        ]
        for text, expected in samples:
            with st.expander(f'{text[:75]}...'):
                if st.button("Analyze", key=f"sa_{hash(text)}"):
                    r = st.session_state.sentiment_analyzer.analyze_sentiment(text)
                    st.markdown(_sentiment_verdict(r['sentiment']), unsafe_allow_html=True)
                    st.plotly_chart(_make_prob_chart(r['class_probabilities']),
                                   use_container_width=True, config={'displayModeBar': False})

    # ╔═══════════════════════════════════════════════════════════════════════════╗
    # ║  TAB 4 — DATA SOURCES                                                   ║
    # ╚═══════════════════════════════════════════════════════════════════════════╝
    with tab4:
        # ── Pipeline diagram ──
        st.markdown('<div class="section-label">RAG Pipeline</div>', unsafe_allow_html=True)
        st.markdown(f'''<div class="pipeline">
  <div class="pipeline-step"><div class="pipeline-step-num">Step 1</div><div class="pipeline-step-title">Ingest</div><div class="pipeline-step-desc">PDFs, earnings calls, news articles</div></div>
  <div class="pipeline-step"><div class="pipeline-step-num">Step 2</div><div class="pipeline-step-title">Embed</div><div class="pipeline-step-desc">Sentence-transformers vectorization</div></div>
  <div class="pipeline-step"><div class="pipeline-step-num">Step 3</div><div class="pipeline-step-title">Index</div><div class="pipeline-step-desc">FAISS similarity search</div></div>
  <div class="pipeline-step"><div class="pipeline-step-num">Step 4</div><div class="pipeline-step-title">Generate</div><div class="pipeline-step-desc">OpenAI / local LLM answer</div></div>
  <div class="pipeline-step"><div class="pipeline-step-num">Step 5</div><div class="pipeline-step-title">Analyze</div><div class="pipeline-step-desc">FinBERT sentiment scoring</div></div>
</div>''', unsafe_allow_html=True)

        # ── Index stats ──
        st.markdown('<div class="section-label" style="margin-top:28px;">Index Statistics</div>', unsafe_allow_html=True)
        d1, d2, d3, d4 = st.columns(4)
        with d1:
            st.metric("Total Chunks", chunk_count)
        with d2:
            st.metric("Companies", len(companies))
        with d3:
            st.metric("Quarters", len(quarters))
        with d4:
            st.metric("Source Types", len(sources))

        # ── Source status ──
        st.markdown('<div class="section-label" style="margin-top:28px;">Data Source Status</div>', unsafe_allow_html=True)
        st.markdown(f'''
<div class="status-row"><div class="dot-green"></div><span>Earnings Calls</span><span class="status-label" style="margin-left:auto">{chunk_count} sample chunks loaded</span></div>
<div class="status-row"><div class="dot-yellow"></div><span>Financial Reports</span><span class="status-label" style="margin-left:auto">Upload PDFs below</span></div>
<div class="status-row"><div class="{"dot-green" if newsapi_ok else "dot-yellow"}"></div><span>News Articles</span><span class="status-label" style="margin-left:auto">{"Active" if newsapi_ok else "API key required"}</span></div>
<div class="status-row"><div class="dot-yellow"></div><span>Market Data</span><span class="status-label" style="margin-left:auto">Yahoo Finance required</span></div>
''', unsafe_allow_html=True)

        # ── Upload ──
        st.markdown('<div class="section-label" style="margin-top:28px;">Upload Financial Reports</div>', unsafe_allow_html=True)
        uploaded = st.file_uploader("Choose PDF files (10-K, 10-Q, annual reports)",
                                    type=['pdf'], accept_multiple_files=True)

        if uploaded:
            if st.button("Parse & Index PDFs", type="primary", key="parse_btn"):
                import tempfile
                new_chunks = []
                for f in uploaded:
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                        tmp.write(f.read())
                        tmp_path = tmp.name
                    try:
                        data = parse_pdf_report(tmp_path)
                        if data:
                            data['file_path'] = f.name
                            chunks = chunk_report(data)
                            new_chunks.extend(chunks)
                            st.success(f"Parsed **{f.name}**: {len(chunks)} chunks")
                    finally:
                        os.remove(tmp_path)
                if new_chunks:
                    rag = st.session_state.rag_pipeline
                    emb = rag.embedder.embed_document_chunks(new_chunks)
                    all_emb = list(rag.chunks) + emb
                    rag.build_index(all_emb)
                    st.success(f"Index rebuilt with **{len(all_emb)}** total chunks")

    # ── Footer ───────────────────────────────────────────────────────────────────
    st.markdown(f'''<div class="app-footer">
  <div class="app-footer-text">
    AI Financial Analyst &nbsp;&middot;&nbsp; Streamlit &nbsp;&middot;&nbsp; OpenAI &nbsp;&middot;&nbsp; FinBERT &nbsp;&middot;&nbsp; FAISS
    &nbsp;&middot;&nbsp; Configure API keys in <code style="color:{ACCENT};background:{BG_CARD};padding:1px 5px;border-radius:3px;font-size:0.68rem;">.env</code>
  </div>
</div>''', unsafe_allow_html=True)


if __name__ == "__main__":
    main()
