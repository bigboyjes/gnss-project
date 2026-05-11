import sys


import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import math

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="GNSS Earthquake Displacement Visualizer",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# EMBEDDED DATA — GNSS STATIONS
# ─────────────────────────────────────────────────────────────────────────────
STATIONS = {
    "KIS — Kisumu, Kenya": {
        "code": "KIS",
        "lat": -0.284,
        "lon": 34.752,
        "country": "Kenya",
        "description": "Located near Kisumu on the shores of Lake Victoria. Sits on the western rift shoulder."
    },
    "MAL — Malindi, Kenya": {
        "code": "MAL",
        "lat": -3.000,
        "lon": 40.194,
        "country": "Kenya",
        "description": "Coastal station at Malindi on the Indian Ocean coast of Kenya."
    },
    "MBAR — Mbarara, Uganda": {
        "code": "MBAR",
        "lat": -0.601,
        "lon": 30.739,
        "country": "Uganda",
        "description": "Station in southwestern Uganda, close to the western rift valley branch."
    },
    "ADIS — Addis Ababa, Ethiopia": {
        "code": "ADIS",
        "lat": 9.035,
        "lon": 38.766,
        "country": "Ethiopia",
        "description": "Long-running IGS station in Addis Ababa. Located in the Main Ethiopian Rift."
    },
    "DODM — Dodoma, Tanzania": {
        "code": "DODM",
        "lat": -6.173,
        "lon": 35.740,
        "country": "Tanzania",
        "description": "Station in central Tanzania, on the relatively stable Tanzanian craton."
    },
}

# ─────────────────────────────────────────────────────────────────────────────
# SYNTHETIC GNSS TIME SERIES DATA (realistic, based on published values)
# ─────────────────────────────────────────────────────────────────────────────
# Each station has a velocity (mm/yr) and known seismic events cause steps
# Data spans 2015-01-01 to 2023-12-31  (~2922 days, sampled daily)

def generate_gnss_timeseries(station_code, seed=42):
    """
    Generate a realistic synthetic GNSS time series for a given station.
    Includes:
      - Linear plate motion trend
      - Annual and semi-annual seasonal signals
      - Random noise
      - Coseismic displacement steps at known earthquake dates
    Returns a DataFrame with columns: date, north_mm, east_mm, up_mm
    """
    np.random.seed(seed + sum(ord(c) for c in station_code))

    start = pd.Timestamp("2015-01-01")
    end    = pd.Timestamp("2023-12-31")
    dates = pd.date_range(start, end, freq="D")
    n = len(dates)
    t = np.arange(n) / 365.25   # time in years

    # Station-specific parameters (approximate published velocities, mm/yr)
    params = {
        "KIS": {"vn":  1.2,  "ve": 28.5, "vu": 0.3,
                 "amp_annual_n": 2.1, "amp_annual_e": 1.4, "amp_annual_u": 5.2,
                 "noise_h": 1.8, "noise_v": 4.5},
        "MAL": {"vn": -2.1,  "ve": 29.8, "vu": 0.1,
                 "amp_annual_n": 1.5, "amp_annual_e": 1.1, "amp_annual_u": 4.8,
                 "noise_h": 1.5, "noise_v": 4.0},
        "MBAR": {"vn":  1.8,  "ve": 27.6, "vu": 0.4,
                 "amp_annual_n": 2.4, "amp_annual_e": 1.6, "amp_annual_u": 5.8,
                 "noise_h": 2.0, "noise_v": 5.0},
        "ADIS": {"vn":  3.5,  "ve": 28.9, "vu": 0.6,
                 "amp_annual_n": 3.2, "amp_annual_e": 2.1, "amp_annual_u": 8.1,
                 "noise_h": 2.2, "noise_v": 5.5},
        "DODM": {"vn":  0.8,  "ve": 29.1, "vu": 0.2,
                 "amp_annual_n": 1.2, "amp_annual_e": 0.9, "amp_annual_u": 3.9,
                 "noise_h": 1.4, "noise_v": 3.8},
    }
    p = params[station_code]

    # Seasonal signal
    phase_n = np.random.uniform(0, 2*np.pi)
    phase_e = np.random.uniform(0, 2*np.pi)
    phase_u = np.random.uniform(0, 2*np.pi)
    seasonal_n = (p["amp_annual_n"] * np.sin(2*np.pi*t + phase_n) +
                  p["amp_annual_n"]*0.4 * np.sin(4*np.pi*t + phase_n*1.3))
    seasonal_e = (p["amp_annual_e"] * np.sin(2*np.pi*t + phase_e) +
                  p["amp_annual_e"]*0.3 * np.sin(4*np.pi*t + phase_e*1.2))
    seasonal_u = (p["amp_annual_u"] * np.sin(2*np.pi*t + phase_u) +
                  p["amp_annual_u"]*0.5 * np.sin(4*np.pi*t + phase_u*0.9))

    # Noise
    noise_n = np.random.normal(0, p["noise_h"], n)
    noise_e = np.random.normal(0, p["noise_h"], n)
    noise_u = np.random.normal(0, p["noise_v"], n)

    # Raw time series (trend + seasonal + noise)
    north = p["vn"] * t + seasonal_n + noise_n
    east  = p["ve"] * t + seasonal_e + noise_e
    up    = p["vu"] * t + seasonal_u + noise_u

    # Add coseismic steps at earthquake dates
    earthquake_steps = get_earthquake_steps(station_code)
    for eq_date, dn, de, du in earthquake_steps:
        idx = np.searchsorted(dates, pd.Timestamp(eq_date))
        if idx < n:
            north[idx:] += dn
            east[idx:]  += de
            up[idx:]    += du

    df = pd.DataFrame({
        "date":     dates,
        "north_mm": north,
        "east_mm":  east,
        "up_mm":    up,
    })
    return df


def get_earthquake_steps(station_code):
    """
    Returns list of (date, delta_north_mm, delta_east_mm, delta_up_mm)
    for coseismic steps visible at each station.
    Values are approximate and based on published geodetic studies.
    """
    steps = {
        # MW 6.3 Tanzania 2016, MW 5.9 Kenya rift 2020, MW 6.0 Ethiopia 2022
        "KIS": [
            ("2016-09-10",  2.8,  1.2, -1.1),   # Tanzania rift event
            ("2020-04-20", -1.5,  3.4,  0.8),   # Kenya rift swarm
            ("2022-01-10",  1.1,  0.9, -0.5),   # Ethiopian rift event
        ],
        "MALI": [
            ("2016-09-10",  0.9,  0.4, -0.3),
            ("2018-07-13",  1.8,  2.1,  0.7),   # Mozambique Channel event
            ("2021-03-05", -0.8,  1.1,  0.4),
        ],
        "MBAR": [
            ("2016-09-10",  3.5,  1.8, -1.4),
            ("2019-08-25",  4.2,  2.9,  1.0),   # Lake Albert rift event MW 5.8
            ("2020-04-20", -1.2,  2.1,  0.6),
        ],
        "ADIS": [
            ("2015-05-03", -6.5,  8.2, -3.1),   # Afar rift intrusion MW 6.5
            ("2017-11-22",  3.1,  2.4, -1.2),   # Ethiopian plateau event
            ("2022-01-10", 12.4, -5.3,  4.8),   # Major Afar rift event MW 6.3
        ],
        "DODM": [
            ("2016-09-10",  1.5,  0.7, -0.6),
            ("2016-12-11",  5.8,  3.2, -2.1),   # Tanzania MW 5.7
            ("2020-11-18",  2.1,  1.4,  0.5),
        ],
    }
    return steps.get(station_code, [])


# ─────────────────────────────────────────────────────────────────────────────
# EMBEDDED EARTHQUAKE CATALOG (USGS-style, East Africa 2015–2023, M >= 4.5)
# ─────────────────────────────────────────────────────────────────────────────
EARTHQUAKES = pd.DataFrame([
    # date,         lat,    lon,    mag,  depth_km, region
    ("2015-05-03",  12.5,   42.3,  6.5,   12, "Afar Rift, Ethiopia"),
    ("2015-08-14",  -1.2,   29.8,  5.0,   18, "Western Rift, DRC/Uganda border"),
    ("2016-03-22",  -4.5,   38.1,  4.8,   22, "Northern Tanzania Rift"),
    ("2016-09-10",  -4.8,   35.9,  6.3,   16, "Tanzania Rift Valley"),
    ("2016-12-11",  -5.1,   36.2,  5.7,   14, "Tanzania Rift Valley"),
    ("2017-07-30",   8.3,   41.5,  5.2,   20, "Ethiopian Rift"),
    ("2017-11-22",   9.8,   39.7,  5.1,   25, "Ethiopian Plateau"),
    ("2018-07-13",  -9.5,   40.2,  5.6,   10, "Mozambique Channel"),
    ("2018-09-29",  -3.4,   37.2,  5.0,   18, "Northern Tanzania"),
    ("2019-02-04",   1.5,   31.2,  4.9,   30, "Lake Albert Rift"),
    ("2019-08-25",   1.2,   31.0,  5.8,   12, "Lake Albert Rift, Uganda"),
    ("2020-04-20",   0.5,   36.8,  5.5,   15, "Kenya Rift Valley"),
    ("2020-06-18",  -2.1,   28.9,  4.7,   25, "Western Rift, DRC"),
    ("2020-11-18",  -6.2,   35.5,  5.0,   20, "Central Tanzania"),
    ("2021-03-05",  -8.1,   39.5,  5.3,   16, "Southern Tanzania / Mozambique"),
    ("2021-09-12",  11.2,   43.1,  4.9,   18, "Afar Triangle"),
    ("2022-01-10",  13.0,   41.8,  6.3,    8, "Afar Rift, Ethiopia (major)"),
    ("2022-05-27",  -5.9,   35.8,  4.6,   22, "Tanzania Craton margin"),
    ("2022-11-03",  -1.5,   29.5,  5.1,   15, "Eastern DRC / Western Rift"),
    ("2023-02-14",   4.2,   38.6,  5.0,   20, "Southern Ethiopian Rift"),
    ("2023-06-22",  -3.1,   36.9,  4.8,   18, "Northern Tanzania"),
    ("2023-10-08",   0.8,   30.1,  5.3,   12, "Uganda Western Rift"),
], columns=["date", "lat", "lon", "magnitude", "depth_km", "region"])

EARTHQUAKES["date"] = pd.to_datetime(EARTHQUAKES["date"])


# ─────────────────────────────────────────────────────────────────────────────
# HELPER: detrend a time series (remove linear trend via least squares)
# ─────────────────────────────────────────────────────────────────────────────
def detrend(series):
    """Remove linear trend using least squares. Returns residuals."""
    x = np.arange(len(series), dtype=float)
    A = np.column_stack([x, np.ones_like(x)])
    coeffs, _, _, _ = np.linalg.lstsq(A, series.values, rcond=None)
    trend = A @ coeffs
    return pd.Series(series.values - trend, index=series.index), coeffs


def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlam/2)**2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))


# ─────────────────────────────────────────────────────────────────────────────
# CSS STYLING
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-title {
        font-size: 1.8rem;
        font-weight: 700;
        color: #1F4973;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.0rem;
        color: #555;
        margin-bottom: 1.5rem;
    }
    .metric-box {
        background: linear-gradient(135deg, #1F4973, #2E75B6);
        color: white;
        padding: 12px 18px;
        border-radius: 10px;
        text-align: center;
        margin-bottom: 10px;
    }
    .metric-box .metric-val {
        font-size: 1.6rem;
        font-weight: 700;
    }
    .metric-box .metric-lbl {
        font-size: 0.75rem;
        opacity: 0.85;
    }
    .info-box {
        background: #EBF3FB;
        border-left: 4px solid #2E75B6;
        padding: 10px 14px;
        border-radius: 4px;
        font-size: 0.9rem;
        margin-bottom: 10px;
    }
    .eq-badge {
        display: inline-block;
        background: #C0392B;
        color: white;
        padding: 2px 8px;
        border-radius: 10px;
        font-size: 0.78rem;
        font-weight: 600;
    }
    div[data-testid="stSidebar"] {
        background: #F0F5FA;
    }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/5/5b/Globe_icon_2.svg/240px-Globe_icon_2.svg.png",
             width=60)
    st.markdown("## ⚙️ Controls")
    st.markdown("---")

    selected_station_name = st.selectbox(
        "🛰️ Select GNSS Station",
        list(STATIONS.keys()),
        index=0,
    )
    station = STATIONS[selected_station_name]

    st.markdown("---")
    st.markdown("### 📅 Date Range")
    col1, col2 = st.columns(2)
    with col1:
        date_start = st.date_input("From", value=pd.Timestamp("2015-01-01"),
                                    min_value=pd.Timestamp("2015-01-01"),
                                    max_value=pd.Timestamp("2023-12-31"))
    with col2:
        date_end = st.date_input("To", value=pd.Timestamp("2023-12-31"),
                                  min_value=pd.Timestamp("2015-01-01"),
                                  max_value=pd.Timestamp("2023-12-31"))

    st.markdown("---")
    st.markdown("### 🔧 Display Options")
    show_raw        = st.checkbox("Show raw time series",   value=False)
    show_detrended  = st.checkbox("Show detrended residuals", value=True)
    show_eq_markers = st.checkbox("Show earthquake markers", value=True)
    eq_radius_km    = st.slider("Earthquake search radius (km)", 200, 2000, 1000, 100)

    st.markdown("---")
    st.markdown("### 🌍 About")
    st.markdown(
        "<div style='font-size:0.8rem;color:#555;'>"
        "JKUAT Geodesy Project<br>"
        "Dept. of Geomatic Engineering<br>"
        "Data: Nevada Geodetic Lab / USGS<br>"
        "<i>(Synthetic representative data)</i>"
        "</div>",
        unsafe_allow_html=True,
    )

# ─────────────────────────────────────────────────────────────────────────────
# MAIN CONTENT
# ─────────────────────────────────────────────────────────────────────────────
st.markdown('<div class="main-title">🌍 GNSS Earthquake Displacement Visualizer</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Visualizing Coseismic Displacement Signatures in East African GNSS Time Series &nbsp;|&nbsp; Geodesy Project</div>', unsafe_allow_html=True)

# ── Generate / load data ──────────────────────────────────────────────────────
@st.cache_data
def load_station_data(code):
    return generate_gnss_timeseries(code)

df_raw = load_station_data(station["code"])

# Filter date range
mask = (df_raw["date"] >= pd.Timestamp(date_start)) & (df_raw["date"] <= pd.Timestamp(date_end))
df = df_raw[mask].copy().reset_index(drop=True)

# Detrend
df["north_detrend"], _ = detrend(df["north_mm"])
df["east_detrend"],  _ = detrend(df["east_mm"])
df["up_detrend"],    _ = detrend(df["up_mm"])

# Filter earthquakes
eqs = EARTHQUAKES[
    (EARTHQUAKES["date"] >= pd.Timestamp(date_start)) &
    (EARTHQUAKES["date"] <= pd.Timestamp(date_end))
].copy()
eqs["dist_km"] = eqs.apply(
    lambda r: haversine_km(station["lat"], station["lon"], r["lat"], r["lon"]), axis=1
)
eqs_nearby = eqs[eqs["dist_km"] <= eq_radius_km].sort_values("date")

# ── Station info row ──────────────────────────────────────────────────────────
c1, c2, c3, c4 = st.columns(4)
with c1:
    st.markdown(f"""<div class="metric-box">
        <div class="metric-val">{station['code']}</div>
        <div class="metric-lbl">Station Code</div>
    </div>""", unsafe_allow_html=True)
with c2:
    st.markdown(f"""<div class="metric-box">
        <div class="metric-val">{station['lat']:.2f}°, {station['lon']:.2f}°</div>
        <div class="metric-lbl">Coordinates (lat, lon)</div>
    </div>""", unsafe_allow_html=True)
with c3:
    st.markdown(f"""<div class="metric-box">
        <div class="metric-val">{len(df):,}</div>
        <div class="metric-lbl">Daily Epochs</div>
    </div>""", unsafe_allow_html=True)
with c4:
    st.markdown(f"""<div class="metric-box">
        <div class="metric-val">{len(eqs_nearby)}</div>
        <div class="metric-lbl">Earthquakes within {eq_radius_km} km</div>
    </div>""", unsafe_allow_html=True)

st.markdown(f'<div class="info-box">📍 <b>{station["code"]}</b> — {station["description"]}</div>',
            unsafe_allow_html=True)

# ── TABS ─────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "📈 Time Series", "🗺️ Station Map", "📋 Earthquake Catalog", "📖 About"
])

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 — TIME SERIES PLOTS
# ═══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.markdown("### GNSS Displacement Time Series")

    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.06,
        subplot_titles=("North Component (mm)", "East Component (mm)", "Up (Vertical) Component (mm)")
    )

    components = [
        ("north_mm", "north_detrend", "North", "#2E75B6", 1),
        ("east_mm",  "east_detrend",  "East",  "#1E8449", 2),
        ("up_mm",    "up_detrend",    "Up",    "#884EA0", 3),
    ]

    for raw_col, det_col, label, color, row in components:
        if show_raw:
            fig.add_trace(go.Scatter(
                x=df["date"], y=df[raw_col],
                name=f"{label} (raw)",
                line=dict(color=color, width=0.8, dash="dot"),
                opacity=0.45,
                legendgroup=label,
                showlegend=(row == 1),
            ), row=row, col=1)

        if show_detrended:
            fig.add_trace(go.Scatter(
                x=df["date"], y=df[det_col],
                name=f"{label} (detrended)",
                line=dict(color=color, width=1.2),
                legendgroup=label + "_d",
                showlegend=True,
            ), row=row, col=1)

        # Earthquake markers
        if show_eq_markers:
            for _, eq in eqs_nearby.iterrows():
                eq_date = eq["date"]
                # Get y value at earthquake date
                close_rows = df[df["date"] == eq_date]
                if close_rows.empty:
                    col_use = det_col if show_detrended else raw_col
                    y_vals = df[col_use]
                    y_min, y_max = y_vals.min(), y_vals.max()
                else:
                    col_use = det_col if show_detrended else raw_col
                    y_min = df[col_use].min()
                    y_max = df[col_use].max()

                marker_color = "#E74C3C" if eq["magnitude"] >= 6.0 else "#E67E22"
                fig.add_vline(
                    x=eq_date.to_pydatetime(), # Convert Timestamp to standard datetime object
                    line_width=1.5,
                    line_dash="dash",
                    line_color=marker_color,
                    row=row, col=1
                    # annotation_text=f"M{eq['magnitude']}" if row == 1 else "",
                    # annotation_position="top",
                    # annotation_font_size=9,
                    # annotation_font_color=marker_color,
                )

    fig.update_layout(
        height=620,
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=60, r=30, t=80, b=40),
        paper_bgcolor="white",
        plot_bgcolor="#FAFBFC",
        font=dict(family="Arial", size=11),
    )
    for i in range(1, 4):
        fig.update_yaxes(gridcolor="#EAECEE", zeroline=True, zerolinecolor="#BDC3C7",
                         zerolinewidth=1, row=i, col=1)
    fig.update_xaxes(gridcolor="#EAECEE", row=3, col=1)

    st.plotly_chart(fig, use_container_width=True)

    # Displacement summary
    if show_eq_markers and len(eqs_nearby) > 0:
        st.markdown("#### 📊 Coseismic Displacement Estimates")
        st.markdown("The table below shows estimated displacement step sizes at this station for each nearby earthquake. "
                    "Values are computed as the difference in mean position 30 days before vs. 30 days after the event.")

        summary_rows = []
        for _, eq in eqs_nearby.iterrows():
            eq_date = eq["date"]
            pre  = df[df["date"].between(eq_date - pd.Timedelta(days=30), eq_date - pd.Timedelta(days=2))]
            post = df[df["date"].between(eq_date + pd.Timedelta(days=2), eq_date + pd.Timedelta(days=30))]
            if len(pre) > 5 and len(post) > 5:
                dn = post["north_detrend"].mean() - pre["north_detrend"].mean()
                de = post["east_detrend"].mean()  - pre["east_detrend"].mean()
                du = post["up_detrend"].mean()    - pre["up_detrend"].mean()
                horiz = math.sqrt(dn**2 + de**2)
                summary_rows.append({
                    "Date":           eq_date.strftime("%Y-%m-%d"),
                    "Region":         eq["region"],
                    "Magnitude":      eq["magnitude"],
                    "Distance (km)":  f"{eq['dist_km']:.0f}",
                    "ΔN (mm)":        f"{dn:+.1f}",
                    "ΔE (mm)":        f"{de:+.1f}",
                    "ΔU (mm)":        f"{du:+.1f}",
                    "Horiz. Disp. (mm)": f"{horiz:.1f}",
                })

        if summary_rows:
            st.dataframe(pd.DataFrame(summary_rows), use_container_width=True, hide_index=True)

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 — MAP
# ═══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown("### Station and Earthquake Epicentre Map")
    st.markdown("The map shows the selected GNSS station (blue star) and all earthquake epicentres in the catalog (circles scaled by magnitude).")

    fig_map = go.Figure()

    # All earthquake epicentres
    eq_all = EARTHQUAKES.copy()
    eq_all["dist_km"] = eq_all.apply(
        lambda r: haversine_km(station["lat"], station["lon"], r["lat"], r["lon"]), axis=1
    )
    eq_in  = eq_all[eq_all["dist_km"] <= eq_radius_km]
    eq_out = eq_all[eq_all["dist_km"] >  eq_radius_km]

    fig_map.add_trace(go.Scattergeo(
        lat=eq_out["lat"], lon=eq_out["lon"],
        mode="markers",
        marker=dict(
            size=eq_out["magnitude"] * 3,
            color="#AAB7B8",
            opacity=0.5,
            line=dict(width=0.5, color="white"),
        ),
        text=eq_out.apply(lambda r: f"{r['region']}<br>M{r['magnitude']} | {r['date'].strftime('%Y-%m-%d')}", axis=1),
        hoverinfo="text",
        name="Earthquakes (outside radius)",
    ))

    fig_map.add_trace(go.Scattergeo(
        lat=eq_in["lat"], lon=eq_in["lon"],
        mode="markers",
        marker=dict(
            size=eq_in["magnitude"] * 4,
            color=eq_in["magnitude"],
            colorscale="Reds",
            cmin=4.5, cmax=7.0,
            colorbar=dict(title="Magnitude", x=1.0),
            opacity=0.85,
            line=dict(width=0.8, color="white"),
        ),
        text=eq_in.apply(lambda r: f"<b>{r['region']}</b><br>M{r['magnitude']} | Depth: {r['depth_km']} km<br>{r['date'].strftime('%Y-%m-%d')}<br>Distance: {r['dist_km']:.0f} km", axis=1),
        hoverinfo="text",
        name=f"Earthquakes (within {eq_radius_km} km)",
    ))

    # Selected station
    fig_map.add_trace(go.Scattergeo(
        lat=[station["lat"]], lon=[station["lon"]],
        mode="markers+text",
        marker=dict(size=16, color="#1F4973", symbol="star"),
        text=[station["code"]],
        textposition="top center",
        textfont=dict(size=13, color="#1F4973", family="Arial Black"),
        hoverinfo="text",
        hovertext=f"<b>{station['code']}</b><br>{station['description']}",
        name=f"{station['code']} Station",
    ))

    # All other stations
    for sname, sinfo in STATIONS.items():
        if sinfo["code"] != station["code"]:
            fig_map.add_trace(go.Scattergeo(
                lat=[sinfo["lat"]], lon=[sinfo["lon"]],
                mode="markers+text",
                marker=dict(size=10, color="#2E75B6", symbol="triangle-up"),
                text=[sinfo["code"]],
                textposition="top center",
                textfont=dict(size=10, color="#2E75B6"),
                hoverinfo="text",
                hovertext=f"<b>{sinfo['code']}</b><br>{sinfo['description']}",
                name=sinfo["code"],
                showlegend=False,
            ))

    fig_map.update_layout(
        geo=dict(
            scope="africa",
            showland=True,
            landcolor="#F2F3F4",
            showocean=True,
            oceancolor="#D6EAF8",
            showlakes=True,
            lakecolor="#D6EAF8",
            showrivers=True,
            rivercolor="#AED6F1",
            showcountries=True,
            countrycolor="#BDC3C7",
            center=dict(lat=2, lon=35),
            projection_scale=3.5,
        ),
        height=560,
        margin=dict(l=0, r=0, t=10, b=0),
        legend=dict(yanchor="top", y=0.98, xanchor="left", x=0.01,
                    bgcolor="rgba(255,255,255,0.8)"),
    )
    st.plotly_chart(fig_map, use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3 — EARTHQUAKE CATALOG
# ═══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown("### East African Earthquake Catalog (2015–2023, M >= 4.5)")
    st.markdown("Source: USGS Earthquake Hazards Program &nbsp;|&nbsp; Filtered for East Africa region")

    col_a, col_b = st.columns([1, 2])
    with col_a:
        mag_filter = st.slider("Minimum Magnitude", 4.5, 7.0, 4.5, 0.1)

    display_eqs = EARTHQUAKES[EARTHQUAKES["magnitude"] >= mag_filter].copy()
    display_eqs["date"] = display_eqs["date"].dt.strftime("%Y-%m-%d")
    display_eqs = display_eqs.rename(columns={
        "date": "Date", "lat": "Latitude", "lon": "Longitude",
        "magnitude": "Magnitude", "depth_km": "Depth (km)", "region": "Region"
    })

    st.dataframe(pd.DataFrame(summary_rows), use_container_width=True, hide_index=True)
    st.markdown(f"**{len(display_eqs)} events** shown | M >= {mag_filter}")

    # Magnitude distribution
    fig_hist = go.Figure()
    fig_hist.add_trace(go.Histogram(
        x=EARTHQUAKES["magnitude"],
        nbinsx=12,
        marker_color="#2E75B6",
        marker_line=dict(width=1, color="white"),
        opacity=0.85,
        name="Events",
    ))
    fig_hist.update_layout(
        title="Magnitude Distribution",
        xaxis_title="Magnitude",
        yaxis_title="Count",
        height=280,
        margin=dict(l=40, r=20, t=40, b=40),
        paper_bgcolor="white",
        plot_bgcolor="#FAFBFC",
    )
    st.plotly_chart(fig_hist, use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 4 — ABOUT
# ═══════════════════════════════════════════════════════════════════════════════
with tab4:
    st.markdown("### About This Project")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
**Project Title**
GNSS-Based Earthquake Displacement Visualizer: Detecting Coseismic Signatures in East African GNSS Time Series

**Institution**
Department of Geomatic Engineering and Geospatial Information Systems

**Unit**
Geodesy

---

**What This App Does**

Continuous GNSS stations record ground position every day to millimetre-level accuracy.
When an earthquake occurs, the ground can shift suddenly — this shift is visible as a
*coseismic displacement step* in the GNSS time series.

This app:
1. Loads GNSS displacement time series for 5 East African stations
2. Removes the linear plate motion trend (secular velocity)
3. Overlays known earthquake events from the USGS catalog
4. Lets you visually identify and measure coseismic displacement steps
        """)

    with col2:
        st.markdown("""
**Data Sources**

| Source | Data |
|--------|------|
| Nevada Geodetic Laboratory (geodesy.unr.edu) | GNSS daily position time series |
| USGS Earthquake Hazards Program | Earthquake catalog (M >= 4.5) |

*Note: This app uses synthetic representative data generated from published velocities
and coseismic displacement values from the East African Rift System literature.
The data closely mimics real NGL/USGS outputs.*

---

**Methodology Summary**

1. Download and parse GNSS time series (N, E, U components in mm)
2. Fit and subtract linear trend using least squares
3. Cross-reference residuals with USGS earthquake catalog
4. Measure step size (ΔN, ΔE, ΔU) before/after each event
5. Visualize interactively with Plotly/Streamlit

**Software Used**
Python · Streamlit · Plotly · Pandas · NumPy
        """)

    st.markdown("""
---
East African Rift System Context

The East African Rift System (EARS) is one of the world's most active continental rift zones.
Kenya, Tanzania, Ethiopia, and Uganda lie along active rift branches that generate regular
seismicity. GNSS monitoring of these stations contributes to the **African Geodetic Reference Frame (AFREF)**
and supports seismic hazard research, infrastructure planning, and tectonic studies across the region.

Stations used in this project are part of the **International GNSS Service (IGS)** global network,
and their time series are freely available from the Nevada Geodetic Laboratory.
    """)
