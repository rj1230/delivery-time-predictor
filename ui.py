"""Delivery Time Predictor: a single-file, multi-page Streamlit app.

Run:  streamlit run ui.py

Expected files next to this script (or in the working directory), all of which
your training script already produces:

    xgb_best.pkl          tuned XGBRegressor
    standard_scaler.pkl   StandardScaler fitted on the training split
    Food delivery.csv     the raw dataset the model was trained on

Nothing else is required. The feature column order is read from the scaler
(``feature_names_in_``), category options and speed thresholds come from the
data, and held-out metrics are recomputed by reproducing the training split
(random_state=42, test_size=0.20). If a ``feature_columns.pkl`` exists it is
used as an override.

Pages: Predict, Batch, Insights, Performance, Guide.

Sections: 1 Config, 2 UI helpers, 3 Data, 4 Artifacts, 5 Preprocessing,
6 Evaluation, 7 Charts, 8 Pages, 9 Entry point.
"""

from __future__ import annotations

import logging
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from html import escape
from pathlib import Path
from typing import Any, Callable, Iterator, Sequence

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)


# ===========================================================================
# 1. CONFIG
# ===========================================================================

APP_TITLE = "Delivery Time Predictor"
APP_ICON = "🛵"

BASE_DIR = Path(__file__).resolve().parent

MODEL_FILE = "xgb_best.pkl"
SCALER_FILE = "standard_scaler.pkl"
RAW_DATA_FILE = "Food delivery.csv"
CLEANED_DATA_FILE = "cleaned_food_delivery.csv"  # optional fallback
FEATURE_COLUMNS_FILE = "feature_columns.pkl"  # optional override

RANDOM_STATE = 42  # must match the training script
TEST_SIZE = 0.20
HISTORY_LIMIT = 25  # predictions kept in the session history

TARGET = "Time_taken(min)"
MAX_DISTANCE_KM = 100.0  # training dropped distance_km >= 100
TRAFFIC_MAP = {"Low": 0, "Medium": 1, "High": 2, "Jam": 3}
CATEGORICAL_COLS = [
    "Type_of_order",
    "Type_of_vehicle",
    "Festival",
    "City",
    "Weatherconditions",
]

MODEL_FEATURES = [
    "Delivery_person_Age",
    "Delivery_person_Ratings",
    "distance_km",
    "order_hour",
    "Weatherconditions",
    "Road_traffic_density",
    "Vehicle_condition",
    "Type_of_order",
    "Type_of_vehicle",
    "multiple_deliveries",
    "Festival",
    "City",
]
NUMERIC_IMPUTE_COLS = [
    "Delivery_person_Age",
    "Delivery_person_Ratings",
    "multiple_deliveries",
]
CATEGORICAL_IMPUTE_COLS = [
    "Weatherconditions",
    "Road_traffic_density",
    "Festival",
    "City",
]
RAW_COORD_COLS = [
    "Restaurant_latitude",
    "Restaurant_longitude",
    "Delivery_location_latitude",
    "Delivery_location_longitude",
]
MISSING_MARKERS = ["NaN", "nan", "NAN", "NA", "N/A", "null", "NULL", "None", ""]

# (low, high) inclusive bounds for numeric inputs.
NUMERIC_RULES: dict[str, tuple[float | None, float | None]] = {
    "Delivery_person_Age": (15, 80),
    "Delivery_person_Ratings": (0, 5),
    "order_hour": (0, 23),
    "Vehicle_condition": (0, 3),
    "multiple_deliveries": (0, 3),
}
WHOLE_NUMBER_COLS = [
    "Delivery_person_Age",
    "order_hour",
    "Vehicle_condition",
    "multiple_deliveries",
]

SPEED_PERCENTILES = (33.0, 67.0)
SPEED_ORDER = ["Fast", "Normal", "Slow"]

PALETTE = {
    "primary": "#58A6C8",  # muted sky
    "accent": "#E3A857",  # amber
    "violet": "#9A8CE0",
    "teal": "#4FC3B0",
    "danger": "#D86B5A",  # clay red
    "success": "#4CAF8A",  # sage green
    "warning": "#D9A441",
    "neutral": "#7D8896",
}
CHART_COLORS = ["#58A6C8", "#E3A857", "#9A8CE0", "#4FC3B0", "#E27D6A", "#7D8896"]
SPEED_TONES = {"Fast": "success", "Normal": "warning", "Slow": "danger"}
TRAFFIC_COLORS = {
    "Low": "#4CAF8A",
    "Medium": "#D9A441",
    "High": "#DB8A55",
    "Jam": "#D86B5A",
}
SEQUENTIAL_SCALE = [
    [0.0, "#1C2442"],
    [0.45, "#3B86AA"],
    [0.75, "#6FCBB8"],
    [1.0, "#F0C97A"],
]
DIVERGING_SCALE = [[0.0, "#58A6C8"], [0.5, "#1A2028"], [1.0, "#D86B5A"]]


class UserError(Exception):
    """An error whose message is safe and useful to show to the end user."""


# ===========================================================================
# 2. UI HELPERS
# ===========================================================================

_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');

:root {
    --bg: #0D1117; --surface: #131A22; --line: #222B36; --line-soft: #1A222C;
    --text: #E6EDF3; --muted: #8B98A8; --accent: #58A6C8;
}
.stApp { background: var(--bg); color: var(--text); }
.stApp, .stApp button, .stApp input, .stApp textarea, .stApp [data-baseweb] {
    font-family: 'Plus Jakarta Sans', system-ui, -apple-system, 'Segoe UI', sans-serif;
}
#MainMenu, footer, .stDeployButton, [data-testid="stDecoration"],
[data-testid="stStatusWidget"] { display: none !important; }
header[data-testid="stHeader"] { background: transparent; }
.block-container { padding-top: 2rem; padding-bottom: 3.5rem; max-width: 1200px; }
h1, h2, h3 { letter-spacing: -0.015em; }

/* page header */
.pg-title { font-size: 1.85rem; font-weight: 700; line-height: 1.15; letter-spacing: -0.025em; margin: .6rem 0 0 0; }
.pg-sub { color: var(--muted); font-size: 1rem; margin: .4rem 0 1.6rem 0; max-width: 62ch; }
.sec-label { font-size: .9rem; font-weight: 650; color: var(--text); margin: 1.3rem 0 .5rem 0;
             padding-bottom: .4rem; border-bottom: 1px solid var(--line); }
.sec-label:first-child { margin-top: .2rem; }

/* cards: flat, hairline border, no shadow */
[data-testid="stVerticalBlockBorderWrapper"] { border-radius: 8px !important;
    border-color: var(--line) !important; background: var(--surface); }
.card-title { font-size: .98rem; font-weight: 650; margin: 0 0 .1rem 0; }
.card-sub { font-size: .83rem; color: var(--muted); margin: 0 0 .4rem 0; }

/* KPI tiles */
.kpi { border: 1px solid var(--line); border-radius: 8px; padding: .95rem 1.1rem;
       height: 100%; background: var(--surface); }
.kpi-label { font-size: .82rem; font-weight: 500; color: var(--muted); display: flex; align-items: center; gap: .45rem; }
.kpi-label::before { content: ""; width: .42rem; height: .42rem; border-radius: 50%;
                     background: var(--kpi-accent, #7D8896); flex: none; }
.kpi-value { font-size: 1.75rem; font-weight: 700; line-height: 1.15; margin-top: .3rem;
             font-variant-numeric: tabular-nums; letter-spacing: -0.02em; }
.kpi-unit { font-size: .92rem; font-weight: 500; color: var(--muted); margin-left: .3rem; }
.kpi-sub { font-size: .82rem; color: var(--muted); margin-top: .3rem; }

/* result ticket */
.ticket { border: 1px solid var(--line); border-left: 3px solid var(--kpi-accent, #58A6C8);
          border-radius: 8px; padding: 1.4rem 1.5rem 1.3rem 1.5rem;
          background: color-mix(in srgb, var(--kpi-accent, #58A6C8) 8%, var(--surface)); }
.ticket-label { font-size: .88rem; font-weight: 500; color: var(--muted); }
.ticket-value { font-size: 4.2rem; font-weight: 700; line-height: 1.05; letter-spacing: -0.045em;
                font-variant-numeric: tabular-nums; margin-top: .2rem; }
.ticket-unit { font-size: 1.2rem; font-weight: 500; color: var(--muted); margin-left: .4rem; letter-spacing: 0; }
.ticket-row { font-size: .88rem; color: var(--muted); margin-top: .35rem; }

.badge { display: inline-block; padding: .18rem .7rem; border-radius: 999px; margin-top: .75rem;
         font-size: .8rem; font-weight: 600; color: var(--badge-color, #7D8896);
         background: color-mix(in srgb, var(--badge-color, #7D8896) 16%, transparent);
         border: 1px solid color-mix(in srgb, var(--badge-color, #7D8896) 45%, transparent); }

/* pace track */
.pace { margin-top: 1.2rem; }
.pace-bar { position: relative; display: flex; height: .45rem; border-radius: 999px; overflow: visible; opacity: .9; }
.pace-bar span { display: block; height: 100%; }
.pace-bar span:first-child { border-radius: 999px 0 0 999px; }
.pace-bar span:nth-child(3) { border-radius: 0 999px 999px 0; }
.pace-pin { position: absolute; top: -.45rem; width: .2rem; height: 1.35rem; border-radius: 2px;
            background: var(--text); transform: translateX(-50%); }
.pace-scale { display: flex; justify-content: space-between; font-size: .76rem; color: var(--muted); margin-top: .6rem; }

.callout { border-radius: 8px; padding: .75rem 1rem; margin: .4rem 0 .8rem 0;
           border: 1px solid var(--line); border-left: 3px solid var(--c); background: var(--surface); }
.empty { text-align: center; padding: 2.6rem 1rem; color: var(--muted);
         border: 1px dashed var(--line); border-radius: 8px; }
.empty b { display: block; font-size: 1rem; color: var(--text); margin: .5rem 0 .15rem 0; }

/* controls */
button[data-testid^="stBaseButton-primary"] { background-color: var(--accent); border-color: var(--accent);
    color: #0B1015; border-radius: 6px; font-weight: 650; }
button[data-testid^="stBaseButton-primary"]:hover { background-color: #7DBCD8; border-color: #7DBCD8; color: #0B1015; }
button[data-testid^="stBaseButton-secondary"], .stDownloadButton button { border-radius: 6px; font-weight: 550;
    border-color: var(--line); background: var(--surface); }
button:focus-visible { outline: 2px solid rgba(88,166,200,.65) !important; outline-offset: 2px; }
[data-testid="stForm"] { border-color: var(--line); border-radius: 8px; background: var(--surface); }
[data-testid="stExpander"] { border-color: var(--line); border-radius: 8px; }
hr { border-color: var(--line); }

/* Deploy button off (the toolbar also holds the top navigation, so keep it) */
[data-testid="stAppDeployButton"], .stAppDeployButton { display: none !important; }

/* brand strip across the top */
.stApp::before { content: ""; position: fixed; top: 0; left: 0; right: 0; height: 3px; z-index: 1000000;
    background: linear-gradient(90deg, #58A6C8, #9A8CE0 55%, #E3A857); }

/* selected options in multiselects: soft tint instead of solid blocks */
[data-baseweb="tag"] { background-color: rgba(88,166,200,.16) !important;
    border: 1px solid rgba(88,166,200,.38) !important; border-radius: 6px !important; }
[data-baseweb="tag"] span { color: #BFE3F2 !important; }

/* number inputs: no stepper buttons */
[data-testid="stNumberInputStepUp"], [data-testid="stNumberInputStepDown"] { display: none; }

/* form: even rhythm, quiet labels, one input style */
[data-testid="stForm"] { padding: 1.1rem 1.3rem 1.3rem 1.3rem; }
[data-testid="stForm"] [data-testid="stVerticalBlock"] { gap: .7rem; }
[data-testid="stWidgetLabel"] p { font-size: .82rem; font-weight: 500; color: var(--muted); }
[data-testid="stTooltipIcon"] { opacity: .4; }
[data-baseweb="select"] > div, [data-baseweb="input"], [data-baseweb="base-input"] {
    background-color: #0F151C !important; border-color: var(--line) !important; border-radius: 6px !important; }
[data-baseweb="select"] > div, [data-baseweb="base-input"] { min-height: 2.5rem; }
[data-testid="stRadio"] label p { font-size: .9rem; color: var(--text); }
[data-testid="stTabs"] [data-baseweb="tab-list"] { gap: .5rem; border-bottom: 1px solid var(--line); }
</style>
"""


def inject_css() -> None:
    """Inject the global stylesheet (hides Streamlit chrome, styles cards/KPIs)."""
    st.markdown(_CSS, unsafe_allow_html=True)


def page_header(title: str, subtitle: str = "") -> None:
    """Render a page title with an optional one-line description."""
    st.markdown(f'<div class="pg-title">{escape(title)}</div>', unsafe_allow_html=True)
    if subtitle:
        st.markdown(
            f'<div class="pg-sub">{escape(subtitle)}</div>', unsafe_allow_html=True
        )


def section_label(text: str) -> None:
    """Heading with a hairline rule above a group of inputs."""
    st.markdown(f'<div class="sec-label">{escape(text)}</div>', unsafe_allow_html=True)


def card(title: str | None = None, subtitle: str | None = None):
    """Return a bordered container (use as ``with card("Title"):``)."""
    box = st.container(border=True)
    if title:
        box.markdown(
            f'<div class="card-title">{escape(title)}</div>', unsafe_allow_html=True
        )
    if subtitle:
        box.markdown(
            f'<div class="card-sub">{escape(subtitle)}</div>', unsafe_allow_html=True
        )
    return box


def badge(text: str, tone: str = "neutral") -> str:
    """HTML for a coloured pill; the text is escaped."""
    color = PALETTE.get(tone, PALETTE["neutral"])
    return f'<span class="badge" style="--badge-color:{color}">{escape(text)}</span>'


def kpi_card(
    label: str,
    value: Any,
    unit: str = "",
    sub: str = "",
    tone: str = "primary",
) -> None:
    """Render one KPI tile. ``value`` is pre-formatted by the caller."""
    unit_html = f'<span class="kpi-unit">{escape(unit)}</span>' if unit else ""
    sub_html = f'<div class="kpi-sub">{escape(sub)}</div>' if sub else ""
    st.markdown(
        f'<div class="kpi" style="--kpi-accent:{PALETTE.get(tone, PALETTE["neutral"])}">'
        f'<div class="kpi-label">{escape(label)}</div>'
        f'<div class="kpi-value">{escape(str(value))}{unit_html}</div>{sub_html}</div>',
        unsafe_allow_html=True,
    )


@dataclass(frozen=True)
class Metric:
    """Declarative KPI for :func:`kpi_row`."""

    label: str
    value: Any
    unit: str = ""
    sub: str = ""
    tone: str = "primary"


def kpi_row(metrics: Sequence[Metric]) -> None:
    """Render metrics as equal-width KPI tiles in one row."""
    for col, m in zip(st.columns(len(metrics)), metrics):
        with col:
            kpi_card(m.label, m.value, m.unit, m.sub, m.tone)


def pace_track(minutes: float, thresholds: tuple[float, float], upper: float) -> str:
    """HTML bar showing where ``minutes`` falls between the Fast/Normal/Slow bands."""
    lo, hi = thresholds
    upper = max(upper, minutes * 1.1, hi * 1.25)
    w_fast = lo / upper * 100
    w_norm = (hi - lo) / upper * 100
    w_slow = max(0.0, 100 - w_fast - w_norm)
    pos = min(minutes / upper * 100, 98.5)
    return (
        '<div class="pace"><div class="pace-bar">'
        f'<span style="width:{w_fast:.2f}%;background:{PALETTE["success"]}"></span>'
        f'<span style="width:{w_norm:.2f}%;background:{PALETTE["warning"]}"></span>'
        f'<span style="width:{w_slow:.2f}%;background:{PALETTE["danger"]}"></span>'
        f'<i class="pace-pin" style="left:{pos:.2f}%"></i></div>'
        f'<div class="pace-scale"><span>Fast up to {lo:.0f} min</span>'
        f"<span>Slow after {hi:.0f} min</span></div></div>"
    )


def result_ticket(
    minutes: float,
    category: str,
    rmse: float | None,
    thresholds: tuple[float, float],
    upper: float,
) -> None:
    """The headline prediction: big number, speed badge, error range, pace track."""
    tone = SPEED_TONES[category]
    rows = ""
    if rmse:
        low, high = max(1.0, minutes - rmse), minutes + rmse
        rows = (
            f'<div class="ticket-row">Likely range {low:.0f} to {high:.0f} min</div>'
            f'<div class="ticket-row">Typical model error is {rmse:.1f} min (test RMSE)</div>'
        )
    st.markdown(
        f'<div class="ticket" style="--kpi-accent:{PALETTE[tone]}">'
        '<div class="ticket-label">Estimated delivery time</div>'
        f'<div class="ticket-value">{minutes:.0f}<span class="ticket-unit">min</span></div>'
        f"{rows}{badge(category, tone)}{pace_track(minutes, thresholds, upper)}</div>",
        unsafe_allow_html=True,
    )


def callout(message: str, kind: str = "info", title: str = "") -> None:
    """Tinted message box. kind: info | success | warning | error."""
    color = {
        "info": "primary",
        "success": "success",
        "warning": "warning",
        "error": "danger",
    }[kind]
    lead = f"<b>{escape(title)}</b> " if title else ""
    st.markdown(
        f'<div class="callout" style="--c:{PALETTE[color]}">{lead}{escape(message)}</div>',
        unsafe_allow_html=True,
    )


def empty_state(title: str, message: str = "", icon: str = "🗂️") -> None:
    """Placeholder for 'nothing to show yet': says what to do next."""
    body = f"<div>{escape(message)}</div>" if message else ""
    st.markdown(
        f'<div class="empty"><b>{escape(title)}</b>{body}</div>',
        unsafe_allow_html=True,
    )


def show_chart(fig: go.Figure) -> None:
    """Render a Plotly figure full-width (compatible with old and new Streamlit)."""
    try:
        st.plotly_chart(fig, width="stretch")
    except Exception:  # noqa: BLE001 - older Streamlit versions
        st.plotly_chart(fig, use_container_width=True)


def show_table(df: pd.DataFrame) -> None:
    """Render a DataFrame full-width without the index."""
    try:
        st.dataframe(df, width="stretch", hide_index=True)
    except Exception:  # noqa: BLE001
        st.dataframe(df, use_container_width=True, hide_index=True)


@contextmanager
def error_boundary(action: str, stop: bool = True) -> Iterator[None]:
    """Show friendly errors instead of tracebacks.

    ``UserError`` messages are shown verbatim. Any other exception is logged with
    its traceback and shown as a generic message with a short reference id.
    Streamlit's ``st.stop``/``st.rerun`` signals derive from BaseException and
    pass straight through.
    """
    try:
        yield
    except UserError as exc:
        st.error(str(exc), icon="⚠️")
        if stop:
            st.stop()
    except Exception:  # noqa: BLE001 - last line of defence for the UI
        ref = uuid.uuid4().hex[:8]
        logger.exception("Unhandled UI error [%s] while trying to %s", ref, action)
        st.error(
            f"We couldn't {action}. Check your inputs and try again. "
            f"If it keeps happening, share reference `{ref}` with the maintainer.",
            icon="🚫",
        )
        if stop:
            st.stop()


# ===========================================================================
# 3. DATA (replicates the training-time cleaning exactly)
# ===========================================================================


def _locate(name: str) -> Path:
    """Find ``name`` next to this script, else in the working directory."""
    for folder in (BASE_DIR, Path.cwd()):
        if (folder / name).exists():
            return folder / name
    return BASE_DIR / name


def _clean_text_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Strip whitespace and standardise missing markers (same as training)."""
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(
            df[col]
        ) or pd.api.types.is_datetime64_any_dtype(df[col]):
            continue
        df[col] = df[col].astype(str).str.strip().replace(MISSING_MARKERS, np.nan)
    return df


def _clean_features(df: pd.DataFrame) -> pd.DataFrame:
    """Fix prefixed weather/target values and numeric columns (same as training)."""
    df = df.copy()
    df["Weatherconditions"] = (
        df["Weatherconditions"]
        .apply(lambda v: v.split(" ")[-1] if pd.notnull(v) else v)
        .replace(
            MISSING_MARKERS, np.nan
        )  # "conditions NaN" becomes missing, then imputed
    )
    df[TARGET] = df[TARGET].apply(
        lambda v: int(str(v).split(" ")[-1]) if pd.notnull(v) else np.nan
    )
    for col in NUMERIC_IMPUTE_COLS:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["Delivery_person_Ratings"] = df["Delivery_person_Ratings"].clip(upper=5)
    return df


def _impute(df: pd.DataFrame) -> pd.DataFrame:
    """Median for numeric, mode for categorical (same as training)."""
    df = df.copy()
    for col in NUMERIC_IMPUTE_COLS:
        df[col] = df[col].fillna(df[col].median())
    for col in CATEGORICAL_IMPUTE_COLS:
        df[col] = df[col].fillna(df[col].mode()[0])
    return df


def haversine_km(lat1, lon1, lat2, lon2):
    """Great-circle distance in km. Works on scalars and pandas Series."""
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    a = (
        np.sin((lat2 - lat1) / 2) ** 2
        + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    )
    return 6371 * 2 * np.arcsin(np.sqrt(a))


def _engineer(df: pd.DataFrame) -> pd.DataFrame:
    """Add distance_km and order_hour; drop impossible distances (same as training)."""
    df = df.copy()
    df["distance_km"] = haversine_km(
        df["Restaurant_latitude"],
        df["Restaurant_longitude"],
        df["Delivery_location_latitude"],
        df["Delivery_location_longitude"],
    )
    df = df[df["distance_km"] < MAX_DISTANCE_KM].reset_index(drop=True)
    df["order_hour"] = pd.to_datetime(
        df["Time_Orderd"], format="%H:%M:%S", errors="coerce"
    ).dt.hour
    df["order_hour"] = df["order_hour"].fillna(df["order_hour"].median())
    return df


def _data_source() -> tuple[str, float, str]:
    """Return (path, mtime, kind) for the dataset; kind is 'raw' or 'cleaned'."""
    raw, cleaned = _locate(RAW_DATA_FILE), _locate(CLEANED_DATA_FILE)
    if raw.exists():
        return str(raw), raw.stat().st_mtime, "raw"
    if cleaned.exists():
        return str(cleaned), cleaned.stat().st_mtime, "cleaned"
    raise UserError(
        f"Dataset not found. Place `{RAW_DATA_FILE}` next to ui.py "
        "(the same file used for training)."
    )


@st.cache_data(show_spinner="Loading data…")
def _load_data(path: str, mtime: float, kind: str) -> pd.DataFrame:
    """Load and prepare the dataset (cached; ``mtime`` busts the cache on change)."""
    df = pd.read_csv(path)
    if kind == "raw":
        needed = [
            "Weatherconditions",
            "Road_traffic_density",
            "Type_of_order",
            "Type_of_vehicle",
            "Festival",
            "City",
            "Vehicle_condition",
            "Time_Orderd",
            TARGET,
            *NUMERIC_IMPUTE_COLS,
            *RAW_COORD_COLS,
        ]
        missing = [c for c in needed if c not in df.columns]
        if missing:
            raise UserError(
                f"`{Path(path).name}` is missing columns: {', '.join(missing)}"
            )
        return _engineer(_impute(_clean_features(_clean_text_columns(df))))
    needed = MODEL_FEATURES + [TARGET]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise UserError(f"`{Path(path).name}` is missing columns: {', '.join(missing)}")
    return df


def get_data() -> pd.DataFrame:
    """Return the prepared dataset (engineered, ready for EDA and evaluation)."""
    return _load_data(*_data_source())


def category_options(df: pd.DataFrame) -> dict[str, list[str]]:
    """Valid values per categorical column, read from the data (never hardcoded)."""
    opts = {c: sorted(df[c].dropna().astype(str).unique()) for c in CATEGORICAL_COLS}
    opts["Road_traffic_density"] = [
        t for t in TRAFFIC_MAP if t in set(df["Road_traffic_density"])
    ] or list(TRAFFIC_MAP)
    return opts


def ui_options(opts: dict[str, list[str]]) -> dict[str, list[str]]:
    """Dropdown options: hides the artefact category ``"NaN"`` (kept valid for batch)."""
    shown = dict(opts)
    shown["Weatherconditions"] = [w for w in opts["Weatherconditions"] if w != "NaN"]
    return shown


def speed_thresholds(df: pd.DataFrame) -> tuple[float, float]:
    """Fast/Normal/Slow cut-offs: the 33rd and 67th percentiles of delivery time."""
    lo, hi = np.percentile(df[TARGET], SPEED_PERCENTILES)
    return float(lo), float(hi)


def speed_category(minutes: float, thresholds: tuple[float, float]) -> str:
    """Map predicted minutes to Fast / Normal / Slow."""
    if minutes <= thresholds[0]:
        return "Fast"
    return "Slow" if minutes > thresholds[1] else "Normal"


# ===========================================================================
# 4. ARTIFACTS
# ===========================================================================


@dataclass(frozen=True)
class Artifacts:
    """The trained model, its scaler and the exact training column order."""

    model: Any
    scaler: Any
    columns: tuple[str, ...]


@st.cache_resource(show_spinner="Loading model…")
def load_artifacts() -> Artifacts:
    """Load model + scaler once per server process."""
    model_path, scaler_path = _locate(MODEL_FILE), _locate(SCALER_FILE)
    missing = [p.name for p in (model_path, scaler_path) if not p.exists()]
    if missing:
        raise UserError(
            f"Missing model file(s): {', '.join(missing)}. "
            "Place them next to ui.py (they are written by the training script)."
        )
    model, scaler = joblib.load(model_path), joblib.load(scaler_path)
    override = _locate(FEATURE_COLUMNS_FILE)
    if override.exists():
        columns = list(joblib.load(override))
    elif hasattr(scaler, "feature_names_in_"):
        columns = list(scaler.feature_names_in_)
    else:
        raise UserError(
            "The scaler has no stored feature names. Save `feature_columns.pkl` "
            "(list(X_train.columns)) next to ui.py."
        )
    return Artifacts(model=model, scaler=scaler, columns=tuple(columns))


# ===========================================================================
# 5. PREPROCESSING (the single inference transform)
# ===========================================================================

RAW_INPUT_COLS = MODEL_FEATURES  # the 12 raw inputs, in training order


def validate_rows(
    raw: pd.DataFrame, opts: dict[str, list[str]]
) -> tuple[pd.DataFrame, dict[int, list[str]]]:
    """Validate and normalise raw input rows.

    Args:
        raw: Frame containing the 12 raw input columns.
        opts: Valid category values from :func:`category_options`.

    Returns:
        ``(clean, problems)``: ``clean`` has coerced types and canonical category
        spelling for *all* rows; ``problems`` maps row index to its error messages.
        Callers should drop rows listed in ``problems`` before scoring.
    """
    df = raw[RAW_INPUT_COLS].copy().reset_index(drop=True)
    problems: dict[int, list[str]] = {}

    def flag(mask: pd.Series, message: str) -> None:
        for i in df.index[mask.fillna(True)]:
            problems.setdefault(int(i), []).append(message)

    for col in [*NUMERIC_RULES, "distance_km"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["Delivery_person_Ratings"] = df["Delivery_person_Ratings"].clip(
        upper=5
    )  # as in training

    for col, (lo, hi) in NUMERIC_RULES.items():
        flag(
            df[col].isna() | (df[col] < lo) | (df[col] > hi),
            f"{col} must be a number from {lo} to {hi}",
        )
    for col in WHOLE_NUMBER_COLS:
        flag(df[col].notna() & (df[col] % 1 != 0), f"{col} must be a whole number")
    flag(
        df["distance_km"].isna() | (df["distance_km"] < 0),
        "distance_km must be a number ≥ 0",
    )
    flag(
        df["distance_km"] >= MAX_DISTANCE_KM,
        f"distance_km must be below {MAX_DISTANCE_KM:.0f} km (training excluded longer routes)",
    )

    for col in [*CATEGORICAL_COLS, "Road_traffic_density"]:
        lookup = {v.lower(): v for v in opts[col]}
        canonical = df[col].astype(str).str.strip().str.lower().map(lookup)
        flag(canonical.isna(), f"{col} must be one of: {', '.join(opts[col])}")
        df[col] = canonical
    return df, problems


def build_features(clean: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    """Turn validated raw rows into the model's exact feature matrix.

    Steps: ordinal traffic map, one-hot encode the five categoricals, then
    reindex to the training columns with ``fill_value=0``. A column dropped by
    ``drop_first=True`` at training time is simply absent from ``columns`` and
    so falls away here. One-hot is done without ``drop_first`` because
    ``get_dummies(drop_first=True)`` on a single row would erase its only dummy.
    """
    X = clean[RAW_INPUT_COLS].copy()
    X["Road_traffic_density"] = X["Road_traffic_density"].map(TRAFFIC_MAP)
    parts = [X.drop(columns=CATEGORICAL_COLS)]
    parts += [
        pd.get_dummies(X[c].astype(str), prefix=c, dtype=int) for c in CATEGORICAL_COLS
    ]
    return (
        pd.concat(parts, axis=1)
        .reindex(columns=list(columns), fill_value=0)
        .astype(float)
    )


def predict_minutes(clean: pd.DataFrame, art: Artifacts) -> np.ndarray:
    """Predict delivery minutes for validated rows: features → scaler → model."""
    X = build_features(clean, art.columns)
    return np.clip(art.model.predict(art.scaler.transform(X)), 1, None)


def distance_sweep(
    clean_row: pd.DataFrame,
    art: Artifacts,
    traffic_levels: Sequence[str],
    distances: np.ndarray,
) -> pd.DataFrame:
    """Predicted minutes for one order across distances and traffic levels."""
    frames = []
    for level in traffic_levels:
        rows = clean_row.loc[clean_row.index.repeat(len(distances))].reset_index(
            drop=True
        )
        rows["distance_km"] = distances
        rows["Road_traffic_density"] = level
        frames.append(
            pd.DataFrame(
                {
                    "distance": distances,
                    "traffic": level,
                    "minutes": predict_minutes(rows, art),
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


def _parent(col: str) -> str:
    """Map a one-hot column back to its source feature (``City_Urban`` → ``City``)."""
    return next((c for c in CATEGORICAL_COLS if col.startswith(c + "_")), col)


def feature_label(name: str) -> str:
    """Human-readable feature name."""
    return {
        "distance_km": "Distance",
        "order_hour": "Order hour",
        "Weatherconditions": "Weather",
        "Road_traffic_density": "Traffic",
        "Delivery_person_Age": "Rider age",
        "Delivery_person_Ratings": "Rider rating",
        "Vehicle_condition": "Vehicle condition",
        "Type_of_order": "Order type",
        "Type_of_vehicle": "Vehicle type",
        "multiple_deliveries": "Multiple deliveries",
        "Festival": "Festival",
        "City": "City",
    }.get(name, name.replace("_", " "))


def explain_prediction(clean_row: pd.DataFrame, art: Artifacts) -> pd.Series:
    """Per-feature contribution (minutes) for one prediction.

    Uses XGBoost's native tree contributions (no SHAP dependency) and groups
    one-hot dummies under their parent feature. Falls back to global feature
    importances (scaled to sum to 1) if native contributions are unavailable.
    """
    X = art.scaler.transform(build_features(clean_row, art.columns))
    try:
        booster = art.model.get_booster()
        dmatrix = xgb.DMatrix(X, feature_names=booster.feature_names)
        contribs = booster.predict(dmatrix, pred_contribs=True)[0][:-1]
        s = pd.Series(contribs, index=list(art.columns))
    except Exception:  # noqa: BLE001
        logger.warning(
            "Native contributions unavailable; using global importances", exc_info=True
        )
        s = pd.Series(art.model.feature_importances_, index=list(art.columns))
    s = s.groupby(_parent).sum()
    s.index = [feature_label(i) for i in s.index]
    return s


# ===========================================================================
# 6. EVALUATION (reproduces the training split to get held-out metrics)
# ===========================================================================


def _adjusted_r2(r2: float, n: int, p: int) -> float:
    return 1 - (1 - r2) * (n - 1) / (n - p - 1)


@st.cache_data(show_spinner="Evaluating the model on the held-out split…")
def _evaluate(path: str, mtime: float, kind: str, model_mtime: float) -> dict[str, Any]:
    """Recompute train/test metrics by re-creating the training split."""
    art, df = load_artifacts(), _load_data(path, mtime, kind)
    X = df[MODEL_FEATURES].copy()
    X["Road_traffic_density"] = X["Road_traffic_density"].map(TRAFFIC_MAP)
    X = pd.get_dummies(X, columns=CATEGORICAL_COLS, drop_first=True, dtype=int)
    if set(X.columns) != set(art.columns):
        diff = sorted(set(X.columns) ^ set(art.columns))[:6]
        raise UserError(
            "The dataset does not produce the same features the model was trained on "
            f"(differences: {', '.join(diff)}). Use the exact CSV from training."
        )
    X = X[list(art.columns)]
    y = df[TARGET]
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    p_tr = art.model.predict(art.scaler.transform(X_tr))
    p_te = art.model.predict(art.scaler.transform(X_te))
    r2_tr, r2_te = r2_score(y_tr, p_tr), r2_score(y_te, p_te)
    return {
        "rmse_train": float(np.sqrt(mean_squared_error(y_tr, p_tr))),
        "rmse_test": float(np.sqrt(mean_squared_error(y_te, p_te))),
        "mae_test": float(mean_absolute_error(y_te, p_te)),
        "within_10": float(np.mean(np.abs(y_te.to_numpy() - p_te) <= 10)),
        "r2_train": float(r2_tr),
        "r2_test": float(r2_te),
        "adj_r2_test": float(_adjusted_r2(r2_te, len(y_te), X_te.shape[1])),
        "n_train": len(y_tr),
        "n_test": len(y_te),
        "y_test": y_te.to_numpy(),
        "pred_test": p_te,
        "dist_test": X_te["distance_km"].to_numpy(),
    }


def get_evaluation() -> dict[str, Any]:
    """Held-out metrics and predictions (cached)."""
    return _evaluate(*_data_source(), _locate(MODEL_FILE).stat().st_mtime)


# ===========================================================================
# 7. CHARTS (all Plotly, one palette)
# ===========================================================================


def _style(fig: go.Figure, height: int = 330, legend: bool = False) -> go.Figure:
    fig.update_layout(
        height=height,
        margin=dict(l=8, r=8, t=16, b=8),
        showlegend=legend,
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        colorway=CHART_COLORS,
        font=dict(
            size=13, color="#AAB3BF", family="Plus Jakarta Sans, system-ui, sans-serif"
        ),
        hoverlabel=dict(bgcolor="#1B212A", bordercolor="#2A323D", font_color="#E6E9EE"),
    )
    fig.update_xaxes(
        gridcolor="rgba(255,255,255,.06)",
        zeroline=False,
        linecolor="rgba(255,255,255,.12)",
    )
    fig.update_yaxes(
        gridcolor="rgba(255,255,255,.06)",
        zeroline=False,
        linecolor="rgba(255,255,255,.12)",
    )
    return fig


def chart_contributions(contrib: pd.Series, top: int = 8) -> go.Figure:
    """Horizontal bars: positive (slower) in chili, negative (faster) in herb green."""
    s = contrib.reindex(contrib.abs().sort_values(ascending=False).index)[:top][::-1]
    fig = go.Figure(
        go.Bar(
            x=s.values,
            y=s.index,
            orientation="h",
            marker_color=[
                PALETTE["danger"] if v >= 0 else PALETTE["success"] for v in s.values
            ],
            text=[f"{v:+.1f}" for v in s.values],
            textposition="outside",
            cliponaxis=False,
        )
    )
    fig.update_xaxes(
        title="Effect on predicted minutes, compared with the model average"
    )
    return _style(fig, height=340)


def chart_sweep(sweep: pd.DataFrame, distance: float, minutes: float) -> go.Figure:
    """Predicted minutes against distance, one line per traffic level."""
    fig = go.Figure()
    for level, g in sweep.groupby("traffic", sort=False):
        fig.add_trace(
            go.Scatter(
                x=g["distance"],
                y=g["minutes"],
                mode="lines",
                name=str(level),
                line=dict(
                    color=TRAFFIC_COLORS.get(str(level), PALETTE["neutral"]), width=2.5
                ),
            )
        )
    fig.add_trace(
        go.Scatter(
            x=[distance],
            y=[minutes],
            mode="markers",
            name="This order",
            marker=dict(size=11, color="#0E1116", line=dict(width=3, color="#E6E9EE")),
        )
    )
    fig.update_xaxes(title="Distance (km)")
    fig.update_yaxes(title="Predicted time (min)")
    fig.update_layout(legend=dict(orientation="h", y=1.12, x=0))
    return _style(fig, height=340, legend=True)


def chart_distribution(df: pd.DataFrame) -> go.Figure:
    fig = px.histogram(
        df, x=TARGET, nbins=30, color_discrete_sequence=[PALETTE["primary"]]
    )
    fig.update_layout(bargap=0.04)
    fig.update_xaxes(title="Delivery time (min)")
    fig.update_yaxes(title="Orders")
    return _style(fig)


def chart_avg_by(
    df: pd.DataFrame, col: str, order: list[str] | None = None, color: str | None = None
) -> go.Figure:
    g = df.groupby(col, observed=True)[TARGET].mean().reset_index()
    if order:
        g[col] = pd.Categorical(
            g[col], categories=[o for o in order if o in set(g[col])], ordered=True
        )
        g = g.sort_values(col)
    else:
        g = g.sort_values(TARGET, ascending=False)
    fig = px.bar(
        g,
        x=col,
        y=TARGET,
        text_auto=".1f",
        color_discrete_sequence=[color or PALETTE["primary"]],
    )
    if col == "Road_traffic_density":
        fig.update_traces(
            marker_color=[
                TRAFFIC_COLORS.get(str(c), PALETTE["neutral"]) for c in g[col]
            ]
        )
    fig.update_xaxes(title=feature_label(col))
    fig.update_yaxes(title="Average time (min)")
    return _style(fig)


def chart_by_hour(df: pd.DataFrame) -> go.Figure:
    g = df.groupby("order_hour")[TARGET].mean().reset_index()
    fig = px.bar(
        g, x="order_hour", y=TARGET, color_discrete_sequence=[PALETTE["accent"]]
    )
    fig.update_xaxes(title="Order hour", dtick=1)
    fig.update_yaxes(title="Average time (min)")
    return _style(fig)


def chart_hour_traffic(df: pd.DataFrame) -> go.Figure:
    """Average delivery time by traffic level and hour of day."""
    levels = [t for t in TRAFFIC_MAP if t in set(df["Road_traffic_density"])]
    pivot = df.pivot_table(
        index="Road_traffic_density",
        columns="order_hour",
        values=TARGET,
        aggfunc="mean",
    ).reindex(levels)
    fig = px.imshow(
        pivot,
        aspect="auto",
        color_continuous_scale=SEQUENTIAL_SCALE,
        labels=dict(x="Order hour", y="Traffic", color="Avg min"),
    )
    fig.update_xaxes(dtick=1)
    return _style(fig, height=280)


def chart_distance_scatter(df: pd.DataFrame, max_points: int = 3000) -> go.Figure:
    sample = df.sample(min(len(df), max_points), random_state=RANDOM_STATE)
    fig = px.scatter(
        sample,
        x="distance_km",
        y=TARGET,
        opacity=0.45,
        color_discrete_sequence=[PALETTE["teal"]],
    )
    fig.update_xaxes(title="Distance (km)")
    fig.update_yaxes(title="Delivery time (min)")
    return _style(fig)


def chart_corr(df: pd.DataFrame) -> go.Figure:
    cols = [
        "Delivery_person_Age",
        "Delivery_person_Ratings",
        "multiple_deliveries",
        "Vehicle_condition",
        "distance_km",
        "order_hour",
        TARGET,
    ]
    corr = df[cols].corr().rename(index=feature_label, columns=feature_label)
    corr = corr.rename(
        index={TARGET: "Delivery time"}, columns={TARGET: "Delivery time"}
    )
    fig = px.imshow(
        corr,
        text_auto=".2f",
        zmin=-1,
        zmax=1,
        color_continuous_scale=DIVERGING_SCALE,
        aspect="auto",
    )
    fig.update_layout(coloraxis_showscale=False)
    return _style(fig, height=420)


def chart_category_counts(scored: pd.DataFrame) -> go.Figure:
    counts = (
        scored["speed_category"]
        .value_counts()
        .reindex(SPEED_ORDER, fill_value=0)
        .rename_axis("Speed")
        .reset_index(name="Orders")
    )
    fig = px.bar(
        counts,
        x="Speed",
        y="Orders",
        color="Speed",
        text="Orders",
        color_discrete_map={k: PALETTE[v] for k, v in SPEED_TONES.items()},
    )
    return _style(fig, height=300)


def chart_prediction_hist(scored: pd.DataFrame) -> go.Figure:
    fig = px.histogram(
        scored,
        x="predicted_minutes",
        nbins=30,
        color_discrete_sequence=[PALETTE["primary"]],
    )
    fig.update_layout(bargap=0.04)
    fig.update_xaxes(title="Predicted time (min)")
    fig.update_yaxes(title="Orders")
    return _style(fig, height=300)


def chart_actual_vs_pred(
    y: np.ndarray, pred: np.ndarray, max_points: int = 3000
) -> go.Figure:
    idx = np.random.default_rng(RANDOM_STATE).choice(
        len(y), min(len(y), max_points), replace=False
    )
    lo, hi = float(min(y.min(), pred.min())), float(max(y.max(), pred.max()))
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=y[idx],
            y=pred[idx],
            mode="markers",
            name="Orders",
            marker=dict(color=PALETTE["primary"], opacity=0.4, size=5),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[lo, hi],
            y=[lo, hi],
            mode="lines",
            name="Perfect (y = x)",
            line=dict(color=PALETTE["accent"], dash="dash"),
        )
    )
    fig.update_xaxes(title="Actual (min)")
    fig.update_yaxes(title="Predicted (min)")
    return _style(fig, legend=True)


def chart_residuals(
    pred: np.ndarray, resid: np.ndarray, max_points: int = 3000
) -> go.Figure:
    idx = np.random.default_rng(RANDOM_STATE).choice(
        len(pred), min(len(pred), max_points), replace=False
    )
    fig = go.Figure(
        go.Scatter(
            x=pred[idx],
            y=resid[idx],
            mode="markers",
            marker=dict(color=PALETTE["primary"], opacity=0.4, size=5),
        )
    )
    fig.add_hline(y=0, line_dash="dash", line_color=PALETTE["accent"])
    fig.update_xaxes(title="Predicted (min)")
    fig.update_yaxes(title="Residual: actual minus predicted (min)")
    return _style(fig)


def chart_residual_hist(resid: np.ndarray) -> go.Figure:
    fig = px.histogram(x=resid, nbins=40, color_discrete_sequence=[PALETTE["teal"]])
    fig.update_xaxes(title="Residual (min)")
    fig.update_yaxes(title="Orders")
    fig.update_layout(bargap=0.04)
    return _style(fig)


def chart_error_by_distance(
    y: np.ndarray, pred: np.ndarray, dist: np.ndarray
) -> go.Figure:
    """Mean absolute error per distance band: shows where the model is weakest."""
    bins = [0, 3, 6, 10, 15, 25, MAX_DISTANCE_KM]
    labels = ["0 to 3", "3 to 6", "6 to 10", "10 to 15", "15 to 25", "25+"]
    band = pd.cut(pd.Series(dist), bins=bins, labels=labels, include_lowest=True)
    g = (
        pd.DataFrame({"band": band, "err": np.abs(y - pred)})
        .groupby("band", observed=True)["err"]
        .agg(["mean", "size"])
        .reset_index()
    )
    fig = px.bar(
        g,
        x="band",
        y="mean",
        text=g["mean"].round(1),
        hover_data={"size": True},
        color_discrete_sequence=[PALETTE["violet"]],
    )
    fig.update_xaxes(title="Distance band (km)")
    fig.update_yaxes(title="Mean absolute error (min)")
    return _style(fig)


def chart_importance(art: Artifacts, top: int = 15) -> go.Figure:
    s = pd.Series(art.model.feature_importances_, index=list(art.columns)).nlargest(
        top
    )[::-1]
    fig = go.Figure(
        go.Bar(
            x=s.values,
            y=[i.replace("_", " ") for i in s.index],
            orientation="h",
            marker_color=PALETTE["primary"],
        )
    )
    fig.update_xaxes(title="Importance")
    return _style(fig, height=470)


# ===========================================================================
# 8. PAGES
# ===========================================================================


def _default_index(options: list[str], series: pd.Series) -> int:
    """Index of the most common value in ``series`` within ``options`` (else 0)."""
    mode = series.mode()
    return options.index(mode.iloc[0]) if len(mode) and mode.iloc[0] in options else 0


def _rmse_or_none() -> float | None:
    """Test RMSE for the ± range, or None if the held-out split can't be rebuilt."""
    try:
        return get_evaluation()["rmse_test"]
    except UserError:
        return None


def page_predict() -> None:
    """Single-order prediction: compact form on top, result and drivers right below."""
    page_header(
        "Predict delivery time",
        "Describe the order and get an estimate in minutes, with what is driving it.",
    )
    with error_boundary("load the model and data"):
        art, df = load_artifacts(), get_data()
        opts_all = category_options(df)
        opts = ui_options(opts_all)
        thresholds = speed_thresholds(df)
        upper = float(df[TARGET].quantile(0.99))

    mode = st.radio(
        "Route input",
        ["Enter distance", "Enter coordinates"],
        horizontal=True,
        label_visibility="collapsed",
    )
    with st.form("predict_form"):
        c1, c2, c3, c4 = st.columns(4)
        age = c1.number_input(
            "Rider age", 15, 80, int(df["Delivery_person_Age"].median()), step=1
        )
        rating = c2.number_input(
            "Rider rating",
            1.0,
            5.0,
            float(round(df["Delivery_person_Ratings"].median(), 1)),
            0.1,
            format="%.1f",
        )
        vtype = c3.selectbox(
            "Vehicle type",
            opts["Type_of_vehicle"],
            index=_default_index(opts["Type_of_vehicle"], df["Type_of_vehicle"]),
        )
        vcond = c4.selectbox(
            "Vehicle condition",
            [0, 1, 2, 3],
            index=1,
            help="Condition score used in training, from 0 to 3.",
        )

        c1, c2, c3, c4 = st.columns(4)
        hour = c1.number_input(
            "Order hour",
            0,
            23,
            int(df["order_hour"].median()),
            step=1,
            help="Hour the order was placed, on a 24-hour clock.",
        )
        multi = c2.selectbox(
            "Multiple deliveries",
            [0, 1, 2, 3],
            help="Other orders the rider is carrying at the same time.",
        )
        otype = c3.selectbox(
            "Order type",
            opts["Type_of_order"],
            index=_default_index(opts["Type_of_order"], df["Type_of_order"]),
        )
        city = c4.selectbox(
            "City type", opts["City"], index=_default_index(opts["City"], df["City"])
        )

        c1, c2, c3, c4 = st.columns(4)
        weather = c1.selectbox(
            "Weather",
            opts["Weatherconditions"],
            index=_default_index(opts["Weatherconditions"], df["Weatherconditions"]),
        )
        traffic = c2.selectbox(
            "Road traffic",
            opts["Road_traffic_density"],
            index=_default_index(
                opts["Road_traffic_density"], df["Road_traffic_density"]
            ),
        )
        festival = c3.selectbox(
            "Festival",
            opts["Festival"],
            index=_default_index(opts["Festival"], df["Festival"]),
        )
        if mode == "Enter distance":
            distance = c4.number_input(
                "Distance (km)",
                0.0,
                99.9,
                5.0,
                0.1,
                help=f"Must be below {MAX_DISTANCE_KM:.0f} km.",
            )
            coords = None
        else:
            c4.text_input("Distance (km)", "From coordinates", disabled=True)
            r1, r2, d1, d2 = st.columns(4)
            coords = (
                r1.number_input("Restaurant lat", -90.0, 90.0, 12.9716, format="%.5f"),
                r2.number_input(
                    "Restaurant lon", -180.0, 180.0, 77.5946, format="%.5f"
                ),
                d1.number_input("Delivery lat", -90.0, 90.0, 12.9910, format="%.5f"),
                d2.number_input("Delivery lon", -180.0, 180.0, 77.6100, format="%.5f"),
            )
            distance = None
        submitted = st.form_submit_button("Predict delivery time", type="primary")

    if submitted:
        with error_boundary("make the prediction", stop=False):
            if coords is not None:
                distance = float(haversine_km(*coords))
                if not (
                    6 <= coords[0] <= 38
                    and 68 <= coords[1] <= 98
                    and 6 <= coords[2] <= 38
                    and 68 <= coords[3] <= 98
                ):
                    st.warning(
                        "These coordinates look outside India. The model was trained on Indian cities."
                    )
            raw = pd.DataFrame(
                [
                    {
                        "Delivery_person_Age": age,
                        "Delivery_person_Ratings": rating,
                        "distance_km": distance,
                        "order_hour": hour,
                        "Weatherconditions": weather,
                        "Road_traffic_density": traffic,
                        "Vehicle_condition": vcond,
                        "Type_of_order": otype,
                        "Type_of_vehicle": vtype,
                        "multiple_deliveries": multi,
                        "Festival": festival,
                        "City": city,
                    }
                ]
            )
            clean, problems = validate_rows(raw, opts_all)
            if problems:
                raise UserError("; ".join(problems[0]))
            minutes = float(predict_minutes(clean, art)[0])
            category = speed_category(minutes, thresholds)
            sweep_max = float(min(MAX_DISTANCE_KM - 0.1, max(15.0, distance * 2)))
            sweep = distance_sweep(
                clean,
                art,
                opts_all["Road_traffic_density"],
                np.linspace(0.5, sweep_max, 25),
            )
            st.session_state["prediction"] = {
                "minutes": minutes,
                "distance": distance,
                "rmse": _rmse_or_none(),
                "category": category,
                "contrib": explain_prediction(clean, art),
                "sweep": sweep,
            }
            history = st.session_state.setdefault("history", [])
            history.insert(
                0,
                {
                    "Time": datetime.now().strftime("%H:%M:%S"),
                    "Predicted (min)": round(minutes, 1),
                    "Speed": category,
                    "Distance (km)": round(distance, 2),
                    "Traffic": traffic,
                    "Weather": weather,
                    "Order hour": hour,
                    "Multiple deliveries": multi,
                    "Festival": festival,
                    "City": city,
                },
            )
            del history[HISTORY_LIMIT:]

    result = st.session_state.get("prediction")
    st.write("")
    if not result:
        empty_state(
            "No prediction yet",
            "Fill in the order details and select Predict delivery time.",
        )
        return

    left, right = st.columns([1, 1.5], gap="large")
    with left:
        result_ticket(
            result["minutes"], result["category"], result["rmse"], thresholds, upper
        )
        st.caption(f"Route distance used: {result['distance']:.2f} km")
    with (
        right,
        card(
            "What is driving this estimate",
            "Red bars add minutes, green bars take minutes off.",
        ),
    ):
        show_chart(chart_contributions(result["contrib"]))

    with card(
        "How distance and traffic change the time",
        "The same order, re-scored at other distances and traffic levels.",
    ):
        show_chart(chart_sweep(result["sweep"], result["distance"], result["minutes"]))

    history = st.session_state.get("history", [])
    if history:
        with st.expander(f"Prediction history ({len(history)})"):
            hist_df = pd.DataFrame(history)
            show_table(hist_df)
            b1, b2, _ = st.columns([1, 1, 3])
            b1.download_button(
                "Download history",
                hist_df.to_csv(index=False).encode(),
                "prediction_history.csv",
                "text/csv",
            )
            if b2.button("Clear history"):
                st.session_state["history"] = []
                st.rerun()


def _template_frame(opts: dict[str, list[str]]) -> pd.DataFrame:
    """Example rows used as the downloadable template."""
    first = {c: opts[c][0] for c in [*CATEGORICAL_COLS, "Road_traffic_density"]}
    rows = [
        {
            "Delivery_person_Age": 28,
            "Delivery_person_Ratings": 4.7,
            "distance_km": 3.2,
            "order_hour": 13,
            "Vehicle_condition": 1,
            "multiple_deliveries": 0,
            **first,
            "Road_traffic_density": opts["Road_traffic_density"][0],
        },
        {
            "Delivery_person_Age": 35,
            "Delivery_person_Ratings": 4.2,
            "distance_km": 9.8,
            "order_hour": 20,
            "Vehicle_condition": 2,
            "multiple_deliveries": 1,
            **first,
            "Road_traffic_density": opts["Road_traffic_density"][-1],
        },
    ]
    return pd.DataFrame(rows)[RAW_INPUT_COLS]


def page_batch() -> None:
    """CSV upload → validation → scoring → preview and download."""
    page_header("Batch predict", "Score many orders at once from a CSV file.")
    with error_boundary("load the model and data"):
        art, df = load_artifacts(), get_data()
        opts_all = category_options(df)
        thresholds = speed_thresholds(df)

    with card("1. Get the template", "Your file needs these columns, in any order."):
        top_l, top_r = st.columns([3, 1])
        with top_l:
            st.markdown(", ".join(f"`{c}`" for c in RAW_INPUT_COLS))
            st.caption(
                "You can leave out `distance_km` if the file has the four coordinate columns: "
                + ", ".join(f"`{c}`" for c in RAW_COORD_COLS)
                + "."
            )
        with top_r:
            st.download_button(
                "Download template",
                _template_frame(opts_all).to_csv(index=False).encode(),
                "delivery_template.csv",
                "text/csv",
            )

    upload = st.file_uploader("2. Upload your CSV", type=["csv"])
    if upload is None:
        empty_state("No file yet", "Upload a CSV to see predictions.", "📄")
        return

    with error_boundary("process this file"):
        try:
            raw = pd.read_csv(upload)
        except Exception as exc:  # noqa: BLE001
            raise UserError(
                "That file couldn't be read as a CSV. Check the delimiter and encoding."
            ) from exc
        raw.columns = [str(c).strip() for c in raw.columns]
        raw = raw.reset_index(drop=True)
        if raw.empty:
            raise UserError("The file has no data rows.")
        if "distance_km" not in raw.columns and all(
            c in raw.columns for c in RAW_COORD_COLS
        ):
            raw["distance_km"] = haversine_km(
                *(pd.to_numeric(raw[c], errors="coerce") for c in RAW_COORD_COLS)
            )
        missing = [c for c in RAW_INPUT_COLS if c not in raw.columns]
        if missing:
            raise UserError(
                f"Missing columns: {', '.join(missing)}. Download the template for the exact layout."
            )

        with st.spinner("Validating and scoring…"):
            clean, problems = validate_rows(raw, opts_all)
            ok = clean.loc[~clean.index.isin(problems)]
            if ok.empty:
                scored = pd.DataFrame()
            else:
                minutes = predict_minutes(ok, art)
                scored = raw.loc[ok.index].copy()
                scored["predicted_minutes"] = np.round(minutes, 1)
                scored["speed_category"] = [
                    speed_category(m, thresholds) for m in minutes
                ]

    st.write("")
    kpi_row(
        [
            Metric("Rows scored", f"{len(scored):,}", tone="success"),
            Metric(
                "Rows rejected",
                f"{len(problems):,}",
                tone="danger" if problems else "neutral",
            ),
            Metric(
                "Mean prediction",
                f"{scored['predicted_minutes'].mean():.1f}" if len(scored) else "–",
                "min",
                tone="accent",
            ),
            Metric(
                "Slow orders",
                f"{(scored['speed_category'] == 'Slow').mean():.0%}"
                if len(scored)
                else "–",
                tone="danger",
            ),
        ]
    )

    if problems:
        rejected = pd.DataFrame(
            [
                {"csv_row": i + 2, "issues": " | ".join(m)}
                for i, m in sorted(problems.items())
            ]
        )
        callout(
            f"{len(problems)} row(s) were skipped. The remaining rows were scored.",
            "warning",
            "Some rows need fixing.",
        )
        with st.expander(f"Rejected rows ({len(problems)})", expanded=len(scored) == 0):
            show_table(rejected.head(200))
            st.download_button(
                "Download rejection report",
                rejected.to_csv(index=False).encode(),
                "rejected_rows.csv",
                "text/csv",
            )
    if scored.empty:
        st.error(
            "No valid rows to score. Fix the issues above and upload again.", icon="⚠️"
        )
        return

    st.write("")
    a, b = st.columns(2)
    with a, card("Speed mix", "Share of orders in each speed band."):
        show_chart(chart_category_counts(scored))
    with b, card("Predicted times"):
        show_chart(chart_prediction_hist(scored))

    section_label("Preview of the first 50 rows")
    show_table(scored.head(50))
    st.download_button(
        "Download predictions",
        scored.to_csv(index=False).encode(),
        "delivery_predictions.csv",
        "text/csv",
        type="primary",
    )


def page_insights() -> None:
    """Filterable exploratory charts over the training data."""
    page_header("Data insights", "What drives delivery time in the historical orders.")
    with error_boundary("load the data"):
        df = get_data()

    traffic_all = [t for t in TRAFFIC_MAP if t in set(df["Road_traffic_density"])]
    with card("Filters", "Every chart and number below follows these choices."):
        f1, f2, f3, f4 = st.columns(4)
        sel = {
            "City": f1.multiselect(
                "City", sorted(df["City"].unique()), default=sorted(df["City"].unique())
            ),
            "Weatherconditions": f2.multiselect(
                "Weather",
                sorted(df["Weatherconditions"].unique()),
                default=sorted(df["Weatherconditions"].unique()),
            ),
            "Road_traffic_density": f3.multiselect(
                "Traffic", traffic_all, default=traffic_all
            ),
            "Festival": f4.multiselect(
                "Festival",
                sorted(df["Festival"].unique()),
                default=sorted(df["Festival"].unique()),
            ),
        }
    view = df
    for col, values in sel.items():
        view = view[view[col].isin(values)]
    if view.empty:
        empty_state(
            "No orders match these filters", "Widen a filter to see data again.", "🔎"
        )
        return

    st.write("")
    kpi_row(
        [
            Metric("Orders", f"{len(view):,}"),
            Metric(
                "Average delivery time",
                f"{view[TARGET].mean():.1f}",
                "min",
                tone="accent",
            ),
            Metric(
                "Orders in traffic jams",
                f"{(view['Road_traffic_density'] == 'Jam').mean():.0%}",
                tone="danger",
            ),
            Metric(
                "Average distance",
                f"{view['distance_km'].mean():.1f}",
                "km",
                tone="violet",
            ),
        ]
    )
    st.write("")

    tab_overview, tab_conditions, tab_route = st.tabs(
        ["Overview", "Conditions", "Rider and route"]
    )
    with tab_overview:
        a, b = st.columns(2)
        with a, card("Delivery time distribution"):
            show_chart(chart_distribution(view))
        with b, card("Average time by order hour"):
            show_chart(chart_by_hour(view))
        with card("Average time by traffic and hour", "Darker cells are slower."):
            show_chart(chart_hour_traffic(view))
    with tab_conditions:
        a, b = st.columns(2)
        with a, card("Average time by traffic"):
            show_chart(chart_avg_by(view, "Road_traffic_density", list(TRAFFIC_MAP)))
        with b, card("Average time by weather"):
            show_chart(chart_avg_by(view, "Weatherconditions", color=PALETTE["violet"]))
        a, b = st.columns(2)
        with a, card("Average time by city type"):
            show_chart(chart_avg_by(view, "City", color=PALETTE["teal"]))
        with b, card("Average time by festival"):
            show_chart(chart_avg_by(view, "Festival", color=PALETTE["accent"]))
    with tab_route:
        a, b = st.columns(2)
        with a, card("Average time by vehicle type"):
            show_chart(chart_avg_by(view, "Type_of_vehicle"))
        with b, card("Average time by order type"):
            show_chart(chart_avg_by(view, "Type_of_order", color=PALETTE["violet"]))
        a, b = st.columns(2)
        with (
            a,
            card("Distance and delivery time", "Random sample of up to 3,000 orders."),
        ):
            show_chart(chart_distance_scatter(view))
        with b, card("Correlation between numeric features"):
            show_chart(chart_corr(view))


def page_performance() -> None:
    """Held-out metrics, diagnostic plots and importances."""
    page_header(
        "Model performance",
        "How well the deployed XGBoost model holds up on orders it has not seen.",
    )
    with error_boundary("evaluate the model"):
        art, ev = load_artifacts(), get_evaluation()

    gap = ev["r2_train"] - ev["r2_test"]
    kpi_row(
        [
            Metric(
                "Test RMSE",
                f"{ev['rmse_test']:.2f}",
                "min",
                f"Train {ev['rmse_train']:.2f}",
            ),
            Metric(
                "Test MAE",
                f"{ev['mae_test']:.2f}",
                "min",
                "Average miss per order",
                "accent",
            ),
            Metric(
                "Within 10 minutes",
                f"{ev['within_10']:.0%}",
                "",
                "Share of test orders",
                "success",
            ),
            Metric(
                "Test R²",
                f"{ev['r2_test']:.3f}",
                "",
                f"Train {ev['r2_train']:.3f}",
                "accent",
            ),
        ]
    )
    st.write("")
    kpi_row(
        [
            Metric(
                "Adjusted R²",
                f"{ev['adj_r2_test']:.3f}",
                "",
                f"{len(art.columns)} features",
                "accent",
            ),
            Metric(
                "Train and test R² gap",
                f"{gap:.3f}",
                "",
                "Lower means less overfitting",
                "success" if gap < 0.05 else "warning" if gap < 0.10 else "danger",
            ),
            Metric("Training orders", f"{ev['n_train']:,}", tone="neutral"),
            Metric("Held-out orders", f"{ev['n_test']:,}", tone="neutral"),
        ]
    )
    st.caption(
        f"Recomputed by reproducing the training split (test_size={TEST_SIZE}, random_state={RANDOM_STATE})."
    )
    st.write("")

    y, p = ev["y_test"], ev["pred_test"]
    resid = y - p
    a, b = st.columns(2)
    with (
        a,
        card(
            "Actual and predicted", "Points on the dashed line are perfect predictions."
        ),
    ):
        show_chart(chart_actual_vs_pred(y, p))
    with (
        b,
        card(
            "Residuals and predicted",
            "A flat, patternless band around 0 is what you want.",
        ),
    ):
        show_chart(chart_residuals(p, resid))
    a, b = st.columns(2)
    with a, card("Residual distribution"):
        show_chart(chart_residual_hist(resid))
    with b, card("Error by distance band", "Where the model misses by the most."):
        show_chart(chart_error_by_distance(y, p, ev["dist_test"]))
    with card("Top 15 features by importance"):
        show_chart(chart_importance(art))
    with st.expander("Best hyperparameters"):
        keys = [
            "n_estimators",
            "max_depth",
            "learning_rate",
            "subsample",
            "colsample_bytree",
            "reg_alpha",
            "reg_lambda",
        ]
        params = art.model.get_params()
        st.json({k: params[k] for k in keys if k in params})


def _file_status() -> pd.DataFrame:
    rows = []
    for name, role, required in [
        (MODEL_FILE, "Trained XGBoost model", True),
        (SCALER_FILE, "Standard scaler", True),
        (RAW_DATA_FILE, "Training dataset", True),
        (CLEANED_DATA_FILE, "Cleaned dataset (fallback)", False),
        (FEATURE_COLUMNS_FILE, "Feature order override", False),
    ]:
        path = _locate(name)
        if path.exists():
            stat = path.stat()
            status = "Found"
            size = f"{stat.st_size / 1024:,.0f} KB"
            modified = datetime.fromtimestamp(stat.st_mtime).strftime("%d %b %Y, %H:%M")
        else:
            status = "Missing" if required else "Not used"
            size, modified = "", ""
        rows.append(
            {
                "File": name,
                "Role": role,
                "Status": status,
                "Size": size,
                "Modified": modified,
            }
        )
    return pd.DataFrame(rows)


def page_guide() -> None:
    """How the app works, accepted input ranges and file health."""
    page_header(
        "Guide",
        "How a prediction is made, what inputs are accepted and which files the app needs.",
    )
    with error_boundary("load the model and data"):
        art, df = load_artifacts(), get_data()
        opts_all = category_options(df)
        thresholds = speed_thresholds(df)

    kpi_row(
        [
            Metric("Orders in dataset", f"{len(df):,}"),
            Metric("Model features", f"{len(art.columns)}", tone="accent"),
            Metric(
                "Average delivery time",
                f"{df[TARGET].mean():.1f}",
                "min",
                tone="accent",
            ),
            Metric(
                "Fast and slow cut-offs",
                f"{thresholds[0]:.0f} and {thresholds[1]:.0f}",
                "min",
                tone="neutral",
            ),
        ]
    )
    st.write("")

    a, b = st.columns(2)
    with a, card("How a prediction is made"):
        st.markdown(
            "1. Your inputs are checked against the ranges on this page.\n"
            "2. Traffic becomes a 0 to 3 score, and the other categories are one-hot encoded.\n"
            "3. Columns are put in the exact order used in training.\n"
            "4. The scaler standardises them, and the XGBoost model returns minutes.\n"
            "5. The result is placed in Fast, Normal or Slow using the 33rd and 67th percentiles "
            "of past delivery times.\n\n"
            "The likely range is the prediction plus or minus the model's test RMSE."
        )
    with b, card("Accepted numeric inputs"):
        rules = pd.DataFrame(
            [
                {"Field": feature_label(k), "Lowest": lo, "Highest": hi}
                for k, (lo, hi) in NUMERIC_RULES.items()
            ]
            + [
                {
                    "Field": "Distance (km)",
                    "Lowest": 0,
                    "Highest": f"under {MAX_DISTANCE_KM:.0f}",
                }
            ]
        )
        show_table(rules.astype(str))

    with card(
        "Accepted categories",
        "Read from the training data, so they always match the model.",
    ):
        cats = pd.DataFrame(
            [
                {"Field": feature_label(k), "Allowed values": ", ".join(v)}
                for k, v in opts_all.items()
            ]
        )
        show_table(cats)

    with card("Files"):
        files = _file_status()
        show_table(files)
        if (files["Status"] == "Missing").any():
            st.error("A required file is missing. Place it next to ui.py.", icon="⚠️")

    if st.button("Reset session (clears prediction and history)"):
        for key in ("prediction", "history"):
            st.session_state.pop(key, None)
        st.rerun()


# ===========================================================================
# 9. ENTRY POINT
# ===========================================================================


def main() -> None:
    """Configure the page and run the selected page from top navigation."""
    st.set_page_config(
        page_title=APP_TITLE,
        page_icon=APP_ICON,
        layout="wide",
        initial_sidebar_state="collapsed",
    )
    inject_css()
    pages: list[Callable[[], None]] = [
        page_predict,
        page_batch,
        page_insights,
        page_performance,
        page_guide,
    ]
    meta = [
        ("Predict", ":material/speed:", "predict"),
        ("Batch", ":material/upload_file:", "batch"),
        ("Insights", ":material/bar_chart:", "insights"),
        ("Performance", ":material/monitoring:", "performance"),
        ("Guide", ":material/menu_book:", "guide"),
    ]
    nav_pages = [
        st.Page(fn, title=t, icon=i, url_path=u, default=(u == "predict"))
        for fn, (t, i, u) in zip(pages, meta)
    ]
    try:
        nav = st.navigation(nav_pages, position="top")
    except TypeError:  # Streamlit < 1.46 has no top navigation
        nav = st.navigation(nav_pages)
    nav.run()


if __name__ == "__main__":
    main()
