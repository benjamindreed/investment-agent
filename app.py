"""
Investment Opportunity Agent — Streamlit App
============================================
Screens large-cap stocks for:
  • Market cap > $5B
  • Trailing P/E ≤ 35
  • Today's volume 1.5–2× the 20-day average
  • Negative media sentiment in the last 90 days

Identified opportunities can be traded directly via a connected Robinhood account.

Run with:
    streamlit run app.py
"""
import os
import time
import json
from datetime import datetime

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Miami Vice colour palette
# ---------------------------------------------------------------------------

MIAMI_VICE_PALETTE = [
    ("Hot Pink",       "#FF6EC7"),
    ("Neon Cyan",      "#00E5FF"),
    ("Electric Purple","#B44FFF"),
    ("Coral",          "#FF6B35"),
    ("Ocean Teal",     "#00C9C8"),
    ("Magenta",        "#FF00CC"),
    ("Electric Blue",  "#00BFFF"),
    ("Sunset Gold",    "#FFB347"),
]

_DEFAULT_COLOR = "#FF6EC7"  # Hot Pink

from screener import screen_stocks
from sentiment import analyze_sentiment, filter_negative_coverage
from brokerage_client import BaseBrokerageClient, BROKERS, BROKER_LOGOS, UNSUPPORTED_BROKERS, make_client

# ---------------------------------------------------------------------------
# Bootstrap
# ---------------------------------------------------------------------------

load_dotenv()


def _secret(key: str, default: str = "") -> str:
    """Read from env var first, then Streamlit secrets (for cloud deployment)."""
    val = os.getenv(key, "")
    if not val:
        try:
            val = st.secrets.get(key, default)
        except Exception:
            pass
    return val or default


st.set_page_config(
    page_title="Investment Opportunity Agent",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Session state initialisation
# ---------------------------------------------------------------------------

if "broker_client" not in st.session_state:
    st.session_state.broker_client = None          # BaseBrokerageClient or None
if "selected_broker" not in st.session_state:
    st.session_state.selected_broker = "Robinhood"
if "scan_results" not in st.session_state:
    st.session_state.scan_results = None          # pd.DataFrame | None
if "sentiment_map" not in st.session_state:
    st.session_state.sentiment_map = {}           # ticker -> sentiment dict
if "last_scan_time" not in st.session_state:
    st.session_state.last_scan_time = None
if "order_feedback" not in st.session_state:
    st.session_state.order_feedback = None
if "theme_color" not in st.session_state:
    st.session_state.theme_color = _DEFAULT_COLOR


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fmt_market_cap(v: float) -> str:
    if v >= 1e12:
        return f"${v/1e12:.2f}T"
    if v >= 1e9:
        return f"${v/1e9:.2f}B"
    return f"${v/1e6:.1f}M"


def _sentiment_badge(avg: float) -> str:
    if avg < -0.2:
        return "🔴 Very Negative"
    if avg < -0.05:
        return "🟠 Negative"
    if avg < 0.05:
        return "🟡 Neutral"
    return "🟢 Positive"


def _inject_theme(color: str) -> None:
    """Inject CSS to apply the selected Miami Vice accent colour site-wide."""
    st.markdown(f"""
    <style>
    /* Primary action buttons */
    button[kind="primary"],
    button[data-testid="baseButton-primary"] {{
        background-color: {color} !important;
        border-color: {color} !important;
        color: #0A0F0A !important;
    }}
    button[kind="primary"]:hover,
    button[data-testid="baseButton-primary"]:hover {{
        opacity: 0.85;
        background-color: {color} !important;
    }}
    /* Links */
    a {{ color: {color} !important; }}
    /* Progress bar fill */
    div[data-testid="stProgressBar"] > div > div {{
        background-color: {color} !important;
    }}
    /* Slider active track & thumb */
    div[data-baseweb="slider"] [role="slider"] {{
        background-color: {color} !important;
        border-color: {color} !important;
    }}
    /* Toggle (on state) */
    div[data-baseweb="toggle"] div[data-checked="true"] {{
        background-color: {color} !important;
    }}
    /* Metric delta colour override */
    [data-testid="stMetricDelta"] svg {{ fill: {color} !important; }}
    /* Selected tab underline */
    button[data-baseweb="tab"][aria-selected="true"] {{
        border-bottom-color: {color} !important;
        color: {color} !important;
    }}
    </style>
    """, unsafe_allow_html=True)


def _make_volume_chart(ticker: str):
    import yfinance as yf
    try:
        hist = yf.Ticker(ticker).history(period="30d")
        if hist.empty:
            return None
        avg = hist["Volume"].iloc[-21:-1].mean()
        colors = [
            "#EF5350" if v > avg * 1.5 else "#42A5F5"
            for v in hist["Volume"]
        ]
        fig = go.Figure(
            go.Bar(
                x=hist.index,
                y=hist["Volume"],
                marker_color=colors,
                name="Volume",
            )
        )
        fig.add_hline(
            y=avg,
            line_dash="dash",
            line_color="orange",
            annotation_text="20-day avg",
            annotation_position="bottom right",
        )
        fig.update_layout(
            title=f"{ticker} — 30-Day Volume",
            height=260,
            margin=dict(l=0, r=0, t=36, b=0),
            xaxis_title=None,
            yaxis_title="Volume",
            showlegend=False,
        )
        return fig
    except Exception:
        return None


# Apply selected theme colour
_inject_theme(st.session_state.theme_color)

# ---------------------------------------------------------------------------
# Sidebar — configuration & Robinhood login
# ---------------------------------------------------------------------------

with st.sidebar:
    st.title("Investment Agent")
    st.caption("Contrarian large-cap screener")

    # --- Miami Vice theme picker ---
    st.subheader("Theme")
    _cols = st.columns(4)
    for _i, (_name, _color) in enumerate(MIAMI_VICE_PALETTE):
        with _cols[_i % 4]:
            _selected = _color == st.session_state.theme_color
            st.markdown(
                f'<div style="background:{_color};height:22px;border-radius:4px;'
                f'border:{"2px solid #fff" if _selected else "1px solid #444"};'
                f'margin-bottom:3px;"></div>',
                unsafe_allow_html=True,
            )
            if st.button(
                _name.split()[0],
                key=f"mv_{_color[1:]}",
                use_container_width=True,
                help=f"{_name}  {_color}",
            ):
                st.session_state.theme_color = _color
                st.rerun()

    st.divider()

    # --- Scan parameters ---
    st.subheader("Scan Parameters")

    # Market Cap
    col_t, col_l = st.columns([1, 3])
    with col_t:
        use_market_cap = st.toggle("", value=True, key="tog_mktcap", help="Enable market cap filter")
    with col_l:
        st.markdown("**Market Cap**")
    min_cap = st.number_input(
        "Min Market Cap ($B)",
        min_value=1.0, max_value=500.0, value=5.0, step=1.0,
        disabled=not use_market_cap,
    ) * 1e9
    if not use_market_cap:
        st.caption("_Filter off — all market caps included_")

    st.divider()

    # P/E Ratio
    col_t, col_l = st.columns([1, 3])
    with col_t:
        use_pe = st.toggle("", value=True, key="tog_pe", help="Enable P/E ratio filter")
    with col_l:
        st.markdown("**P/E Ratio**")
    max_pe = st.slider("Max P/E Ratio", 5, 100, 50, disabled=not use_pe)
    if not use_pe:
        st.caption("_Filter off — all P/E ratios included_")

    st.divider()

    # Volume
    col_t, col_l = st.columns([1, 3])
    with col_t:
        use_volume = st.toggle("", value=True, key="tog_vol", help="Enable volume ratio filter")
    with col_l:
        st.markdown("**Volume Spike**")
    vol_range = st.slider(
        "Volume Ratio Range (× 20-day avg)",
        0.5, 5.0, (1.2, 5.0), step=0.1,
        disabled=not use_volume,
    )
    min_vol, max_vol = vol_range
    if not use_volume:
        st.caption("_Filter off — all volume levels included_")

    st.divider()

    # Sentiment
    col_t, col_l = st.columns([1, 3])
    with col_t:
        use_sentiment = st.toggle("", value=True, key="tog_sent", help="Enable negative sentiment filter")
    with col_l:
        st.markdown("**Negative Sentiment**")
    sentiment_weeks = st.slider(
        "Sentiment Lookback (weeks)",
        min_value=1, max_value=13, value=5,
        disabled=not use_sentiment,
        help="How far back to search for negative news. 13 weeks ≈ 90 days.",
    )
    sentiment_days = sentiment_weeks * 7
    if use_sentiment:
        st.caption(f"{sentiment_weeks} week{'s' if sentiment_weeks != 1 else ''} = {sentiment_days} days")
    else:
        st.caption("_Filter off — sentiment analysis skipped_")

    st.divider()

    # 5-Year High Premium
    col_t, col_l = st.columns([1, 3])
    with col_t:
        use_5yr_high = st.toggle("", value=True, key="tog_5yr", help="Enable 5-year high premium filter")
    with col_l:
        st.markdown("**5-Year High Premium**")
    min_5yr_pct = st.slider(
        "5yr high at least X% above current price",
        min_value=0, max_value=200, value=10, step=5,
        disabled=not use_5yr_high,
        help="Finds stocks trading significantly below their 5-year peak.",
    )
    if use_5yr_high:
        st.caption(f"5yr high ≥ current price + **{min_5yr_pct}%**")
    else:
        st.caption("_Filter off — 5yr high still shown in results_")

    st.divider()

    news_api_key = st.text_input(
        "NewsAPI Key (optional)",
        value=_secret("NEWS_API_KEY"),
        type="password",
        help="Provides broader news coverage. Free at newsapi.org",
    )


    st.divider()

    # --- Brokerage login ---
    st.subheader("Brokerage")

    broker_options = list(BROKERS.keys()) + list(UNSUPPORTED_BROKERS.keys())
    broker_labels = [
        f"{BROKER_LOGOS.get(b, '⚪')} {b}" for b in list(BROKERS.keys())
    ] + [f"⚫ {b} (unavailable)" for b in UNSUPPORTED_BROKERS.keys()]

    prev_broker = st.session_state.selected_broker
    broker_idx = st.selectbox(
        "Select your brokerage",
        options=range(len(broker_options)),
        format_func=lambda i: broker_labels[i],
        index=list(BROKERS.keys()).index(prev_broker) if prev_broker in BROKERS else 0,
        label_visibility="collapsed",
    )
    selected_broker_name = broker_options[broker_idx]

    # Reset client when broker changes
    if selected_broker_name != st.session_state.selected_broker:
        if st.session_state.broker_client:
            st.session_state.broker_client.logout()
        st.session_state.broker_client = None
        st.session_state.selected_broker = selected_broker_name
        st.rerun()

    # Show unavailable notice
    if selected_broker_name in UNSUPPORTED_BROKERS:
        st.info(f"**{selected_broker_name}** does not offer a public trading API for individual accounts.\n\n_{UNSUPPORTED_BROKERS[selected_broker_name]}_")
        broker = None
    else:
        # Ensure a client exists for the selected broker
        if st.session_state.broker_client is None or \
                type(st.session_state.broker_client).__name__ != f"{selected_broker_name}Client":
            st.session_state.broker_client = make_client(selected_broker_name)
        broker: BaseBrokerageClient = st.session_state.broker_client

        if broker.logged_in:
            buying_power = broker.get_buying_power()
            st.success(f"Connected  |  ${buying_power:,.2f} available")
            if st.button("Log Out", use_container_width=True):
                broker.logout()
                st.rerun()

        elif selected_broker_name == "Robinhood":
            with st.form("rh_login"):
                rh_user = st.text_input("Email", value=_secret("ROBINHOOD_USERNAME"))
                rh_pass = st.text_input("Password", value=_secret("ROBINHOOD_PASSWORD"), type="password")
                rh_mfa  = st.text_input("MFA Code (if required)", value="")
                if st.form_submit_button("Connect Robinhood", use_container_width=True):
                    with st.spinner("Connecting..."):
                        ok, msg = broker.login(username=rh_user, password=rh_pass, mfa_code=rh_mfa)
                    if ok:
                        st.success(msg)
                        st.rerun()
                    else:
                        st.error(f"Login failed: {msg}")

        elif selected_broker_name == "Alpaca":
            with st.form("alpaca_login"):
                alpaca_key    = st.text_input("API Key",    value=_secret("ALPACA_API_KEY"),    type="password")
                alpaca_secret = st.text_input("API Secret", value=_secret("ALPACA_API_SECRET"), type="password")
                alpaca_paper  = st.checkbox("Paper trading account", value=True,
                                            help="Use a paper (simulated) account. Uncheck for live trading.")
                st.caption("Get API keys free at [alpaca.markets](https://alpaca.markets)")
                if st.form_submit_button("Connect Alpaca", use_container_width=True):
                    with st.spinner("Connecting..."):
                        ok, msg = broker.login(api_key=alpaca_key, api_secret=alpaca_secret, paper=alpaca_paper)
                    if ok:
                        st.success(msg)
                        st.rerun()
                    else:
                        st.error(f"Login failed: {msg}")

    st.divider()
    st.caption(
        "Data: Yahoo Finance · NewsAPI · Google News\n"
        "Trading: Robinhood (robin-stocks) · Alpaca (alpaca-py)\n\n"
        "_Not financial advice. Use at your own risk._"
    )


# ---------------------------------------------------------------------------
# Main content
# ---------------------------------------------------------------------------

st.title("Investment Opportunity Agent")
_active = []
if use_market_cap:  _active.append(f"Mkt cap > ${min_cap/1e9:.0f}B")
if use_pe:          _active.append(f"P/E ≤ {max_pe}")
if use_volume:      _active.append(f"Vol {min_vol:.1f}–{max_vol:.1f}×")
if use_5yr_high:    _active.append(f"5yr high ≥ +{min_5yr_pct}%")
if use_sentiment:   _active.append(f"Neg. sentiment ({sentiment_days}d / {sentiment_weeks}w)")
st.caption("  •  ".join(_active) if _active else "No filters active — showing full universe")

# --- Scan button ---
col_scan, col_status = st.columns([2, 5])
with col_scan:
    run_scan = st.button("Run Full Scan", type="primary", use_container_width=True)
with col_status:
    if st.session_state.last_scan_time:
        st.info(f"Last scan: {st.session_state.last_scan_time.strftime('%Y-%m-%d %H:%M:%S')}")

if run_scan:
    progress_bar = st.progress(0.0)
    status_text = st.empty()

    def ui_progress(step: int, total_steps: int, msg: str):
        progress_bar.progress(step / total_steps)
        status_text.write(f"**{msg}**")

    # --- Step 1+2: Fundamental screen ---
    with st.spinner("Running stock screener..."):
        df = screen_stocks(
            min_market_cap=min_cap       if use_market_cap else 0,
            max_pe=max_pe                if use_pe         else 99999,
            min_vol_ratio=min_vol        if use_volume     else 0,
            max_vol_ratio=max_vol        if use_volume     else 99999,
            min_5yr_high_pct=min_5yr_pct if use_5yr_high  else 0,
            progress_callback=ui_progress,
        )

    if df.empty:
        active_filters = [f for f, on in [
            ("market cap", use_market_cap), ("P/E", use_pe),
            ("volume", use_volume), ("5yr high", use_5yr_high),
        ] if on]
        label = ", ".join(active_filters) if active_filters else "no active"
        st.warning(f"No stocks passed the {label} filter(s).")
        progress_bar.empty()
        status_text.empty()
    else:
        # --- Step 3: Sentiment filter ---
        sentiment_map: dict[str, dict] = {}

        if use_sentiment:
            status_text.write(f"**Running sentiment analysis on {len(df)} candidates...**")
            progress_bar.progress(0.75)

            tickers_info = df[["ticker", "name"]].to_dict("records")
            total_sent = len(tickers_info)
            passing_tickers: set[str] = set()

            sent_bar = st.progress(0.0)
            sent_status = st.empty()

            for idx, item in enumerate(tickers_info):
                sent_status.write(
                    f"Sentiment {idx+1}/{total_sent}: analysing **{item['ticker']}**"
                )
                sent_bar.progress((idx + 1) / total_sent)
                result = analyze_sentiment(
                    item["ticker"], item["name"], news_api_key,
                    days=sentiment_days,
                )
                sentiment_map[item["ticker"]] = result
                if result["has_negative_coverage"]:
                    passing_tickers.add(item["ticker"])
                time.sleep(0.15)

            sent_bar.empty()
            sent_status.empty()
            final_df = df[df["ticker"].isin(passing_tickers)].copy()
        else:
            # Sentiment filter off — all fundamentals candidates pass
            final_df = df.copy()
            passing_tickers = set(df["ticker"].tolist())

        if not final_df.empty:
            final_df["sentiment_score"] = final_df["ticker"].map(
                lambda t: sentiment_map.get(t, {}).get("avg_compound", float("nan"))
            )
            final_df["articles"] = final_df["ticker"].map(
                lambda t: sentiment_map.get(t, {}).get("article_count", 0)
            )

        st.session_state.scan_results = final_df
        st.session_state.sentiment_map = sentiment_map
        st.session_state.last_scan_time = datetime.now()

        progress_bar.progress(1.0)
        status_text.write("**Scan complete.**")
        time.sleep(0.5)
        progress_bar.empty()
        status_text.empty()
        st.rerun()


# ---------------------------------------------------------------------------
# Results display
# ---------------------------------------------------------------------------

df = st.session_state.scan_results  # pd.DataFrame or None
sentiment_map: dict = st.session_state.sentiment_map

if df is None:
    st.info("Configure your parameters in the sidebar and click **Run Full Scan** to begin.")

elif df.empty:
    n_active = sum([use_market_cap, use_pe, use_volume, use_5yr_high, use_sentiment])
    st.warning(f"No opportunities matched the {n_active} active filter{'s' if n_active != 1 else ''} in this scan.")

else:
    st.success(f"{len(df)} opportunity{'s' if len(df) != 1 else ''} identified")

    # --- Summary table ---
    display_df = df.copy()
    display_df["market_cap"] = display_df["market_cap"].apply(_fmt_market_cap)
    display_df["volume_ratio"] = display_df["vol_ratio"].apply(lambda x: f"{x:.2f}×")

    table_cols = ["ticker", "name", "sector", "price", "market_cap", "pe_ratio", "volume_ratio"]
    col_rename = {
        "ticker": "Ticker", "name": "Company", "sector": "Sector",
        "price": "Price ($)", "market_cap": "Mkt Cap",
        "pe_ratio": "P/E", "volume_ratio": "Vol Ratio",
    }

    if "5yr_high" in display_df.columns:
        display_df["5yr_high_disp"] = display_df.apply(
            lambda r: (
                f"${r['5yr_high']:,.2f} (+{r['5yr_high_pct_above']:.0f}%)"
                if pd.notna(r.get("5yr_high")) and pd.notna(r.get("5yr_high_pct_above"))
                else "—"
            ), axis=1
        )
        table_cols.append("5yr_high_disp")
        col_rename["5yr_high_disp"] = "5yr High (↑%)"

    if "sentiment_score" in display_df.columns:
        display_df["sentiment"] = display_df["sentiment_score"].apply(
            lambda v: _sentiment_badge(v) if pd.notna(v) else "—"
        )
        table_cols += ["articles", "sentiment"]
        col_rename.update({"articles": "Articles", "sentiment": "Sentiment"})

    st.dataframe(
        display_df[table_cols].rename(columns=col_rename),
        use_container_width=True,
        hide_index=True,
    )

    st.divider()

    # --- Per-ticker detail + trade ---
    st.subheader("Opportunity Detail & Trade")

    selected_ticker = st.selectbox(
        "Select a ticker to inspect",
        options=df["ticker"].tolist(),
        format_func=lambda t: f"{t} — {df.loc[df['ticker']==t, 'name'].iloc[0]}",
    )

    if selected_ticker:
        row = df.loc[df["ticker"] == selected_ticker].iloc[0]
        sent_data = sentiment_map.get(selected_ticker, {})

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Price", f"${row['price']:,.2f}")
        col2.metric("P/E Ratio", f"{row['pe_ratio']:.1f}")
        col3.metric(
            "Market Cap", _fmt_market_cap(row["market_cap"])
        )
        col4.metric("Volume Ratio", f"{row['vol_ratio']:.2f}×")

        col5, col6, col7 = st.columns(3)
        col5.metric("Sector", row["sector"])
        col6.metric("52W High", f"${row['52w_high']:,.2f}" if row.get("52w_high") else "N/A")
        col6.metric("52W Low", f"${row['52w_low']:,.2f}" if row.get("52w_low") else "N/A")
        if pd.notna(row.get("5yr_high")) and pd.notna(row.get("5yr_high_pct_above")):
            col6.metric(
                "5yr High",
                f"${row['5yr_high']:,.2f}",
                delta=f"+{row['5yr_high_pct_above']:.1f}% above today",
                delta_color="inverse",
            )
        col7.metric(
            "Sentiment Score",
            f"{sent_data.get('avg_compound', 0):.3f}",
            help="VADER compound: -1 (most negative) to +1 (most positive)",
        )
        col7.metric(
            "Negative Articles",
            f"{sent_data.get('negative_count', 0)} / {sent_data.get('article_count', 0)}",
        )

        # Volume chart
        vol_fig = _make_volume_chart(selected_ticker)
        if vol_fig:
            st.plotly_chart(vol_fig, use_container_width=True)

        # News articles
        with st.expander(f"Recent News ({sent_data.get('article_count', 0)} articles)"):
            articles = sent_data.get("articles", [])
            if not articles:
                st.write("No articles found.")
            for art in articles[:15]:
                compound = art["compound"]
                badge = "🔴" if compound < -0.2 else "🟠" if compound < -0.05 else "🟡" if compound < 0.05 else "🟢"
                url = art.get("url", "")
                title = art["title"]
                link = f"[{title}]({url})" if url else title
                st.markdown(
                    f"{badge} **{art['source']}** · score: `{compound:.3f}`  \n{link}  \n"
                    f"<small>{art.get('published_at', '')}</small>",
                    unsafe_allow_html=True,
                )
                st.divider()

        # --- Trade panel ---
        st.subheader("Place Order")

        broker = st.session_state.broker_client
        if not broker or not broker.logged_in:
            st.warning("Connect a brokerage account in the sidebar to trade.")
        else:
            buying_power = broker.get_buying_power()
            st.caption(f"Buying power available: **${buying_power:,.2f}**")

            order_tab1, order_tab2, order_tab3 = st.tabs(
                ["Market Buy (shares)", "Market Buy (dollars)", "Limit Buy"]
            )

            # --- Tab 1: market buy by shares ---
            with order_tab1:
                shares_qty = st.number_input(
                    "Shares to buy",
                    min_value=0.001, max_value=10000.0,
                    value=1.0, step=0.1,
                    key="mkt_shares",
                )
                est_cost = shares_qty * row["price"]
                st.caption(f"Estimated cost: **${est_cost:,.2f}**")

                if st.button(
                    f"Buy {shares_qty:.3f} shares of {selected_ticker}",
                    key="btn_mkt_shares",
                    type="primary",
                    disabled=est_cost > buying_power,
                ):
                    with st.spinner("Placing order..."):
                        result = broker.place_market_buy(selected_ticker, shares_qty)
                    st.session_state.order_feedback = result

            # --- Tab 2: market buy by dollars ---
            with order_tab2:
                dollar_amt = st.number_input(
                    "Dollar amount to invest ($)",
                    min_value=1.0, max_value=float(buying_power) if buying_power else 100000.0,
                    value=min(100.0, buying_power) if buying_power else 100.0,
                    step=10.0,
                    key="mkt_dollars",
                )
                est_shares = dollar_amt / row["price"] if row["price"] > 0 else 0
                st.caption(f"Estimated shares: **{est_shares:.4f}**")

                if st.button(
                    f"Invest ${dollar_amt:,.2f} in {selected_ticker}",
                    key="btn_mkt_dollars",
                    type="primary",
                    disabled=dollar_amt > buying_power,
                ):
                    with st.spinner("Placing order..."):
                        result = broker.place_dollar_buy(selected_ticker, dollar_amt)
                    st.session_state.order_feedback = result

            # --- Tab 3: limit buy ---
            with order_tab3:
                lmt_price = st.number_input(
                    "Limit price ($)",
                    min_value=0.01,
                    value=round(row["price"] * 0.98, 2),
                    step=0.01,
                    key="lmt_price",
                    help="Order fills only if the price drops to this level",
                )
                lmt_shares = st.number_input(
                    "Shares",
                    min_value=0.001, max_value=10000.0,
                    value=1.0, step=0.1,
                    key="lmt_shares",
                )
                est_lmt_cost = lmt_shares * lmt_price
                st.caption(f"Max cost: **${est_lmt_cost:,.2f}** (GTC order)")

                if st.button(
                    f"Place limit buy for {lmt_shares:.3f} {selected_ticker} @ ${lmt_price:.2f}",
                    key="btn_lmt",
                    type="primary",
                    disabled=est_lmt_cost > buying_power,
                ):
                    with st.spinner("Placing order..."):
                        result = broker.place_limit_buy(selected_ticker, lmt_shares, lmt_price)
                    st.session_state.order_feedback = result

            # --- Order feedback ---
            if st.session_state.order_feedback:
                fb = st.session_state.order_feedback
                if "error" in fb:
                    st.error(f"Order failed: {fb['error']}")
                else:
                    st.success(
                        f"Order placed!  ID: `{fb.get('order_id', 'N/A')}`  "
                        f"State: **{fb.get('state', 'N/A')}**"
                    )
                if st.button("Dismiss", key="dismiss_feedback"):
                    st.session_state.order_feedback = None
                    st.rerun()

    st.divider()

    # --- Holdings snapshot ---
    broker = st.session_state.broker_client
    if broker and broker.logged_in:
        with st.expander(f"My {broker.broker_name} Holdings"):
            holdings = broker.get_holdings()
            if holdings:
                h_df = pd.DataFrame(holdings)
                h_df["average_buy_price"] = h_df["average_buy_price"].apply(
                    lambda x: f"${x:,.2f}"
                )
                h_df["equity"] = h_df["equity"].apply(lambda x: f"${x:,.2f}")
                h_df["percent_change"] = h_df["percent_change"].apply(
                    lambda x: f"{x:+.2f}%"
                )
                st.dataframe(
                    h_df.rename(columns={
                        "ticker": "Ticker",
                        "name": "Company",
                        "quantity": "Shares",
                        "average_buy_price": "Avg Cost",
                        "equity": "Equity",
                        "percent_change": "Return",
                    }),
                    use_container_width=True,
                    hide_index=True,
                )
            else:
                st.write("No holdings found.")

    # --- Export results ---
    with st.expander("Export Scan Results"):
        csv = df.to_csv(index=False)
        st.download_button(
            "Download CSV",
            data=csv,
            file_name=f"investment_opportunities_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
            mime="text/csv",
        )
        json_data = json.dumps(sentiment_map, indent=2)
        st.download_button(
            "Download Sentiment Data (JSON)",
            data=json_data,
            file_name=f"sentiment_{datetime.now().strftime('%Y%m%d_%H%M')}.json",
            mime="application/json",
        )
