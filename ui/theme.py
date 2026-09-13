"""Centralized visual theme for the TransferGrid workbench."""

from __future__ import annotations

import streamlit as st


def inject_theme_css() -> None:
    """Inject the monochrome launcher theme and Streamlit overrides."""
    st.markdown(
        """
        <style>
        :root {
          --bg: #000000;
          --fg: #FFFFFF;
          --muted: #888888;
          --border: #333333;
          --border-hover: #555555;
          --console-fg: #CCCCCC;
          --status-fg: #666666;
        }

        html, body, .stApp, [data-testid="stAppViewContainer"] {
          background-color: var(--bg) !important;
          color: var(--fg) !important;
          font-family: "Courier New", Courier, monospace !important;
        }

        .stApp {
          background-image:
            radial-gradient(1px 1px at 12% 22%, rgba(255,255,255,.35), transparent),
            radial-gradient(1px 1px at 78% 18%, rgba(255,255,255,.28), transparent),
            radial-gradient(1px 1px at 34% 76%, rgba(255,255,255,.22), transparent),
            radial-gradient(1px 1px at 88% 64%, rgba(255,255,255,.22), transparent),
            radial-gradient(1px 1px at 56% 40%, rgba(255,255,255,.18), transparent);
        }

        html, body, [class*="st-"], h1, h2, h3, p, label, button, code, pre {
          font-family: "Courier New", Courier, monospace !important;
          border-radius: 0 !important;
        }

        [data-testid="stIconMaterial"],
        .material-symbols-rounded,
        .material-symbols-outlined {
          font-family: "Material Symbols Rounded" !important;
          font-weight: normal !important;
          font-style: normal !important;
          letter-spacing: normal !important;
          text-transform: none !important;
          white-space: nowrap !important;
          word-wrap: normal !important;
          direction: ltr !important;
          font-feature-settings: "liga" !important;
          -webkit-font-feature-settings: "liga" !important;
          -webkit-font-smoothing: antialiased !important;
        }

        /* Expander glyphs are unnecessary in this workbench and their
           font ligatures can leak as literal arrow_right/arrow_down text. */
        [data-testid="stExpander"] summary [data-testid="stIconMaterial"],
        [data-testid="stExpander"] summary [data-testid="stExpanderToggleIcon"] {
          display: none !important;
        }

        [data-testid="stExpander"] summary {
          padding-left: 16px !important;
        }

        [data-testid="stHeader"], [data-testid="stToolbar"],
        [data-testid="stSidebar"], #MainMenu, footer {
          display: none !important;
        }

        .block-container, [data-testid="stMainBlockContainer"] {
          max-width: 1200px !important;
          min-height: 100vh;
          padding: 32px 24px 24px !important;
        }

        .tg-header {
          position: relative;
          display: flex;
          justify-content: center;
          margin-bottom: 32px;
        }

        .tg-version {
          position: absolute;
          top: 0;
          right: 0;
          color: var(--muted);
          font-size: 12px;
          letter-spacing: .08em;
        }

        .tg-brand {
          display: flex;
          align-items: center;
          gap: 18px;
          text-align: left;
        }

        .tg-logo { width: 48px; height: 48px; flex: 0 0 48px; }

        .tg-title {
          margin: 0;
          color: var(--fg);
          font-size: clamp(32px, 3.2vw, 40px);
          font-weight: 700;
          letter-spacing: .1em;
          line-height: 1.1;
        }

        .tg-subtitle {
          margin-top: 8px;
          color: var(--muted);
          font-size: 12px;
          letter-spacing: .2em;
          line-height: 1.45;
        }

        [data-testid="stButton"] { margin-bottom: 6px; }

        [data-testid="stButton"] > button {
          position: relative;
          width: 100%;
          height: 44px;
          padding: 0 12px;
          border: 1px solid var(--border) !important;
          background: var(--bg) !important;
          color: var(--fg) !important;
          box-shadow: none !important;
          font-size: 16px !important;
          font-weight: 400 !important;
          letter-spacing: .05em;
          line-height: 1 !important;
          text-align: left !important;
        }

        [data-testid="stButton"] > button:hover {
          border-color: var(--border-hover) !important;
          color: var(--fg) !important;
        }

        [data-testid="stButton"] > button:focus-visible {
          outline: 1px solid var(--fg) !important;
          outline-offset: 2px !important;
        }

        button[data-testid="stBaseButton-primary"], button[kind="primary"] {
          border-color: var(--fg) !important;
        }

        button[data-testid="stBaseButton-primary"]::after,
        button[kind="primary"]::after {
          content: "▸";
          position: absolute;
          right: 12px;
          color: var(--fg);
          font-size: 14px;
        }

        [data-testid="stVerticalBlockBorderWrapper"] {
          border: 1px solid var(--border) !important;
          background: var(--bg) !important;
          border-radius: 0 !important;
          box-shadow: none !important;
        }

        .tg-panel-title {
          margin: 0 0 16px;
          color: var(--fg);
          font-size: 14px;
          font-weight: 400;
          letter-spacing: .06em;
        }

        .tg-env-name {
          margin: 0 0 16px;
          color: var(--fg);
          font-size: 18px;
          font-weight: 400;
          letter-spacing: .04em;
          line-height: 1.35;
        }

        .tg-description {
          margin: 0;
          color: var(--muted);
          font-size: 13px;
          line-height: 1.55;
        }

        .tg-divider {
          height: 1px;
          margin: 16px 0;
          background: var(--border);
        }

        .tg-meta {
          display: flex;
          flex-wrap: wrap;
          gap: 12px 24px;
          color: var(--muted);
          font-size: 12px;
          letter-spacing: .06em;
        }

        .tg-console {
          min-height: 88px;
          max-height: 120px;
          box-sizing: border-box;
          margin-top: 22px;
          padding: 12px 14px;
          border: 1px solid var(--border);
          background: var(--bg);
        }

        .tg-console-label {
          color: var(--muted);
          font-size: 11px;
          letter-spacing: .08em;
        }

        .tg-console-body {
          margin: 10px 0 0;
          color: var(--console-fg);
          font-size: 13px;
          line-height: 1.4;
          white-space: pre-wrap;
        }

        .tg-status {
          margin-top: 10px;
          padding: 0 4px;
          color: var(--status-fg);
          font-size: 11px;
          letter-spacing: .02em;
          line-height: 32px;
          white-space: nowrap;
          overflow: hidden;
          text-overflow: ellipsis;
        }

        [data-testid="stImage"] img {
          border-radius: 0 !important;
          image-rendering: pixelated;
        }

        @media (max-width: 899px) {
          .tg-header { justify-content: flex-start; }
          .tg-brand { gap: 12px; }
          .tg-version { position: static; margin-left: 12px; }
          .tg-title { font-size: 28px; }
          .tg-subtitle { font-size: 10px; letter-spacing: .12em; }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
