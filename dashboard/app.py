from pathlib import Path
import unicodedata

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
import xgboost as xgb

from config.theme import (
    AMERICA_YELLOW,
    TEXT_DARK,
)


# ============================================================
# CONFIGURACIÓN
# ============================================================

st.set_page_config(
    page_title="CoachLens | Club América",
    page_icon="CA",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# RUTAS
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

DASHBOARD_DIR = ROOT / "dashboard"
DATA_DIR = DASHBOARD_DIR / "data"

CSS_PATH = (
    DASHBOARD_DIR
    / "styles"
    / "america.css"
)


# ------------------------------------------------------------
# VARONIL
# ------------------------------------------------------------

MALE_SIMULATOR_PATH = (
    DATA_DIR
    / "simulator_context.csv"
)

MALE_MATCH_HISTORY_PATH = (
    DATA_DIR
    / "match_history.csv"
)

MALE_MODEL_PATH = (
    ROOT
    / "outputs"
    / "models"
    / "blend_xgboost_model.json"
)

MALE_COACH_PATH = (
    ROOT
    / "outputs"
    / "coach_engine"
    / "coach_engine_all_test.csv"
)

MALE_SHAP_PATH = (
    ROOT
    / "outputs"
    / "models"
    / "interpretability"
    / "shap_global_importance.csv"
)


# ------------------------------------------------------------
# FEMENIL
# ------------------------------------------------------------

FEMALE_SIMULATOR_PATH = (
    DATA_DIR
    / "female_simulator_context.csv"
)

FEMALE_MATCH_HISTORY_PATH = (
    DATA_DIR
    / "female_match_history.csv"
)

FEMALE_CHANGE_HISTORY_PATH = (
    DATA_DIR
    / "female_change_history.csv"
)

FEMALE_METRICS_PATH = (
    DATA_DIR
    / "female_change_metrics.csv"
)

FEMALE_FEATURES_PATH = (
    DATA_DIR
    / "female_change_model_features.csv"
)

FEMALE_SELECTION_PATH = (
    DATA_DIR
    / "female_change_selection.csv"
)

FEMALE_MODEL_PATH = (
    ROOT
    / "outputs"
    / "models"
    / "female"
    / "change_model"
    / "female_change_model.json"
)


# ============================================================
# CSS
# ============================================================

def load_css(path):

    if not path.exists():
        return

    with open(
        path,
        "r",
        encoding="utf-8",
    ) as file:

        css = file.read()

    st.html(
        f"""
        <style>
        {css}
        </style>
        """
    )


load_css(CSS_PATH)


# ============================================================
# ESTADOS
# ============================================================

STATE_LEVELS = [
    "CONTROL",
    "PRESSURE",
    "TRANSITION",
    "PROTECTION",
    "URGENCY",
]


STATE_DISPLAY = {
    "CONTROL": "CONTROL",
    "PRESSURE": "PRESIÓN",
    "TRANSITION": "TRANSICIÓN",
    "PROTECTION": "PROTECCIÓN",
    "URGENCY": "URGENCIA",
}


STATE_REVERSE = {
    display: internal
    for internal, display
    in STATE_DISPLAY.items()
}


# ============================================================
# LOADERS
# ============================================================

@st.cache_data
def load_csv(path):

    if not path.exists():
        return pd.DataFrame()

    return pd.read_csv(path)


@st.cache_resource
def load_model(path):

    booster = xgb.Booster()

    booster.load_model(
        str(path)
    )

    return booster


# ============================================================
# HELPERS
# ============================================================

def normalize_text(value):

    value = str(
        value
    ).strip().lower()

    return "".join(
        char
        for char
        in unicodedata.normalize(
            "NFD",
            value,
        )
        if unicodedata.category(
            char
        ) != "Mn"
    )


def normalize_match_id(value):

    value = str(value)

    if value.endswith(".0"):
        value = value[:-2]

    return value


def state_display(value):

    value = str(
        value
    ).upper()

    return STATE_DISPLAY.get(
        value,
        value,
    )


def safe_percent(value):

    try:

        value = float(value)

        if not np.isfinite(value):
            return "N/D"

        if abs(value) <= 1:
            value *= 100

        return f"{value:.1f}%"

    except Exception:

        return "N/D"


def percentage_points(value):

    try:

        value = float(value)

        if not np.isfinite(value):
            return "N/D"

        return f"{value * 100:+.1f} pp"

    except Exception:

        return "N/D"


def metric_from_table(
    data,
    metric_name,
):

    if data.empty:
        return np.nan

    if not all(
        column in data.columns
        for column in [
            "metric",
            "value",
        ]
    ):

        return np.nan

    result = data[
        data["metric"]
        .astype(str)
        .eq(metric_name)
    ]

    if result.empty:
        return np.nan

    value = pd.to_numeric(
        result["value"],
        errors="coerce",
    ).iloc[0]

    return value


def first_existing_column(
    data,
    candidates,
):

    for column in candidates:

        if column in data.columns:
            return column

    return None


# ============================================================
# NOMBRES AMIGABLES DE VARIABLES
# ============================================================

def friendly_feature_name(feature):

    mapping = {
        "game_state_num":
            "Estado del marcador",

        "minute":
            "Minuto del partido",

        "current_state_control":
            "Estar actualmente en control",

        "current_state_pressure":
            "Estar actualmente en presión",

        "current_state_transition":
            "Estar actualmente en transición",

        "current_state_protection":
            "Estar actualmente en protección",

        "current_state_urgency":
            "Estar actualmente en urgencia",

        "america_support_share_10_last5":
            "Apoyos cercanos",

        "america_nearest_opponent_distance_last5":
            "Distancia al rival más cercano",

        "america_pressure_share_5_last5":
            "Intensidad de presión",

        "recovery_height_diff_last5":
            "Altura de recuperación",

        "pressure_diff_last5":
            "Diferencia de presión",

        "pressures_diff_last5":
            "Diferencia de presión",

        "entries_box_diff_last5":
            "Entradas al área",

        "entries_final_third_diff_last5":
            "Entradas al último tercio",

        "progressive_pass_diff_last5":
            "Pases progresivos",

        "progressive_passes_diff_last5":
            "Pases progresivos",

        "progressive_carries_diff_last5":
            "Conducciones progresivas",

        "america_field_tilt_last5":
            "Dominio territorial",

        "america_possession_context_last5":
            "Control con balón",

        "shots_diff_last5":
            "Diferencia de tiros",
    }

    if feature in mapping:

        return mapping[
            feature
        ]

    result = (
        str(feature)
        .replace(
            "america_",
            ""
        )
        .replace(
            "rival_",
            "rival "
        )
        .replace(
            "_last5",
            ""
        )
        .replace(
            "_last3",
            " reciente"
        )
        .replace(
            "_diff",
            ""
        )
        .replace(
            "_trend",
            " tendencia"
        )
        .replace(
            "_",
            " "
        )
        .strip()
    )

    return result.capitalize()


# ============================================================
# PARTIDOS JUGADOS
# ============================================================

def played_matches_only(data):

    if data.empty:
        return data.copy()

    result = data.copy()

    if "played" in result.columns:

        played = (
            result["played"]
            .astype(str)
            .str.lower()
            .isin(
                [
                    "true",
                    "1",
                    "t",
                ]
            )
        )

        result = result[
            played
        ].copy()

    if (
        "home_score" in result.columns
        and
        "away_score" in result.columns
    ):

        home_score = pd.to_numeric(
            result["home_score"],
            errors="coerce",
        )

        away_score = pd.to_numeric(
            result["away_score"],
            errors="coerce",
        )

        result = result[
            home_score.notna()
            &
            away_score.notna()
        ].copy()

    if "match_date" in result.columns:

        result["match_date"] = pd.to_datetime(
            result["match_date"],
            errors="coerce",
        )

        today = (
            pd.Timestamp
            .today()
            .normalize()
        )

        result = result[
            result["match_date"].notna()
            &
            (
                result["match_date"]
                <= today
            )
        ].copy()

    return result


# ============================================================
# RIVALES
# ============================================================

def get_historical_opponents(
    matches,
):

    data = played_matches_only(
        matches
    )

    if data.empty:
        return []

    if "opponent" not in data.columns:
        return []

    return sorted(
        data["opponent"]
        .dropna()
        .astype(str)
        .str.strip()
        .unique()
        .tolist()
    )


def count_matches(
    matches,
    opponent,
):

    data = played_matches_only(
        matches
    )

    if data.empty:
        return 0

    if "opponent" not in data.columns:
        return 0

    mask = (
        data["opponent"]
        .astype(str)
        .map(normalize_text)
        .eq(
            normalize_text(
                opponent
            )
        )
    )

    result = data[
        mask
    ]

    if "match_id" in result.columns:

        return int(
            result["match_id"]
            .nunique()
        )

    return int(
        len(result)
    )


def get_played_match_ids(
    matches,
):

    data = played_matches_only(
        matches
    )

    if (
        data.empty
        or
        "match_id" not in data.columns
    ):

        return set()

    return set(
        data["match_id"]
        .map(
            normalize_match_id
        )
        .tolist()
    )


# ============================================================
# ESCALA ROBUSTA
# ============================================================

def robust_scale(series):

    numeric = pd.to_numeric(
        series,
        errors="coerce",
    )

    numeric = numeric[
        np.isfinite(
            numeric
        )
    ]

    if numeric.empty:
        return 1.0

    q1 = numeric.quantile(
        0.25
    )

    q3 = numeric.quantile(
        0.75
    )

    scale = q3 - q1

    if (
        not np.isfinite(scale)
        or
        scale == 0
    ):

        scale = numeric.std()

    if (
        not np.isfinite(scale)
        or
        scale == 0
    ):

        scale = 1.0

    return float(scale)


# ============================================================
# CONTEXTO HISTÓRICO
# ============================================================

def build_context_from_history(
    data,
    opponent,
    venue,
    minute,
    score_state,
    current_state,
    feature_names,
    medians=None,
):

    pool = data.copy()

    if pool.empty:

        raise ValueError(
            "No existen contextos históricos disponibles."
        )

    medians = medians or {}

    pool["minute"] = pd.to_numeric(
        pool["minute"],
        errors="coerce",
    )

    pool["game_state_num"] = pd.to_numeric(
        pool["game_state_num"],
        errors="coerce",
    )

    pool = pool.dropna(
        subset=[
            "minute",
            "game_state_num",
        ]
    )

    if pool.empty:

        raise ValueError(
            "No existen contextos históricos utilizables."
        )

    components = pd.DataFrame(
        index=pool.index
    )

    components["minute"] = (
        (
            pool["minute"]
            -
            float(minute)
        ).abs()
        /
        robust_scale(
            pool["minute"]
        )
    )

    components["score"] = (
        (
            pool["game_state_num"]
            -
            float(score_state)
        ).abs()
        /
        robust_scale(
            pool["game_state_num"]
        )
    )

    if "venue" in pool.columns:

        components["venue"] = (
            pool["venue"]
            .astype(str)
            .str.upper()
            .ne(
                str(venue).upper()
            )
            .astype(float)
        )

    if "opponent" in pool.columns:

        components["opponent"] = (
            pool["opponent"]
            .astype(str)
            .map(normalize_text)
            .ne(
                normalize_text(
                    opponent
                )
            )
            .astype(float)
        )

    if "current_tactical_state" in pool.columns:

        components["state"] = (
            pool[
                "current_tactical_state"
            ]
            .astype(str)
            .str.upper()
            .ne(
                str(
                    current_state
                ).upper()
            )
            .astype(float)
        )

    pool["_distance"] = components.mean(
        axis=1
    )

    distances = (
        pool["_distance"]
        .replace(
            [
                np.inf,
                -np.inf,
            ],
            np.nan,
        )
        .dropna()
    )

    if distances.empty:

        raise ValueError(
            "No fue posible calcular similitud."
        )

    cutoff = float(
        distances.quantile(
            0.25
        )
    )

    selected = pool[
        pool["_distance"]
        <= cutoff
    ].copy()

    if selected.empty:

        selected = (
            pool
            .sort_values(
                "_distance"
            )
            .head(1)
            .copy()
        )

    distance_scale = robust_scale(
        distances
    )

    selected["_weight"] = np.exp(
        -
        selected["_distance"]
        /
        distance_scale
    )

    if (
        not np.isfinite(
            selected["_weight"]
        ).all()
        or
        selected["_weight"].sum()
        <= 0
    ):

        selected["_weight"] = 1.0

    selected["_weight"] = (
        selected["_weight"]
        /
        selected["_weight"].sum()
    )

    feature_row = {}

    for feature in feature_names:

        if feature == "minute":

            feature_row[
                feature
            ] = float(
                minute
            )

            continue

        if feature == "game_state_num":

            feature_row[
                feature
            ] = float(
                score_state
            )

            continue

        if feature.startswith(
            "current_state_"
        ):

            expected_state = (
                feature
                .replace(
                    "current_state_",
                    ""
                )
                .upper()
            )

            feature_row[
                feature
            ] = float(
                expected_state
                ==
                str(
                    current_state
                ).upper()
            )

            continue

        if feature not in selected.columns:

            feature_row[
                feature
            ] = float(
                medians.get(
                    feature,
                    0.0,
                )
            )

            continue

        values = pd.to_numeric(
            selected[
                feature
            ],
            errors="coerce",
        )

        valid = (
            values.notna()
            &
            np.isfinite(
                values
            )
        )

        if valid.sum() == 0:

            feature_row[
                feature
            ] = float(
                medians.get(
                    feature,
                    0.0,
                )
            )

            continue

        weights = (
            selected.loc[
                valid,
                "_weight",
            ]
            .to_numpy(
                dtype=float
            )
        )

        weights = (
            weights
            /
            weights.sum()
        )

        feature_row[
            feature
        ] = float(
            np.sum(
                values.loc[
                    valid
                ]
                .to_numpy(
                    dtype=float
                )
                *
                weights
            )
        )

    mean_distance = float(
        selected["_distance"]
        .mean()
    )

    similarity = float(
        (
            distances
            >= mean_distance
        ).mean()
        *
        100
    )

    return (
        pd.DataFrame(
            [
                feature_row
            ]
        ),
        selected,
        similarity,
    )


# ============================================================
# TRANSICIÓN HISTÓRICA FEMENIL
# ============================================================

def calculate_historical_transition(
    neighbors,
    current_state,
):

    empty_result = {
        "state": None,
        "share": np.nan,
        "observations": 0,
        "distribution": pd.DataFrame(),
    }

    if neighbors.empty:

        return empty_result

    target_column = (
        "future_tactical_state_5min"
    )

    if target_column not in neighbors.columns:

        return empty_result

    data = neighbors.copy()

    data[
        target_column
    ] = (
        data[
            target_column
        ]
        .astype(str)
        .str.upper()
        .str.strip()
    )

    current_state = str(
        current_state
    ).upper()

    transition_data = data[
        data[
            target_column
        ]
        .ne(current_state)
        &
        data[
            target_column
        ]
        .isin(
            STATE_LEVELS
        )
    ].copy()

    if transition_data.empty:

        return empty_result

    if "_weight" not in transition_data.columns:

        transition_data[
            "_weight"
        ] = 1.0

    transition_data[
        "_weight"
    ] = pd.to_numeric(
        transition_data[
            "_weight"
        ],
        errors="coerce",
    ).fillna(
        0.0
    )

    if (
        transition_data[
            "_weight"
        ].sum()
        <= 0
    ):

        transition_data[
            "_weight"
        ] = 1.0

    distribution = (
        transition_data
        .groupby(
            target_column,
            as_index=False,
        )
        .agg(
            weight=(
                "_weight",
                "sum",
            ),
            observations=(
                target_column,
                "size",
            ),
        )
    )

    total_weight = (
        distribution[
            "weight"
        ].sum()
    )

    distribution[
        "share"
    ] = (
        distribution[
            "weight"
        ]
        /
        total_weight
    )

    distribution = (
        distribution
        .sort_values(
            "share",
            ascending=False,
        )
        .reset_index(
            drop=True
        )
    )

    top_row = distribution.iloc[
        0
    ]

    return {
        "state":
            str(
                top_row[
                    target_column
                ]
            ),

        "share":
            float(
                top_row[
                    "share"
                ]
            ),

        "observations":
            int(
                transition_data.shape[0]
            ),

        "distribution":
            distribution,
    }


# ============================================================
# MÉTRICAS VARONIL
# ============================================================

def calculate_male_metrics(
    data,
):

    result = {
        "top1": np.nan,
        "top2": np.nan,
        "baseline": np.nan,
        "balanced": np.nan,
        "n": 0,
    }

    if data.empty:
        return result

    required = [
        "actual_future_state",
        "predicted_state",
    ]

    if not all(
        column in data.columns
        for column in required
    ):

        return result

    evaluation = data.dropna(
        subset=required
    ).copy()

    if evaluation.empty:
        return result

    actual = (
        evaluation[
            "actual_future_state"
        ]
        .astype(str)
        .str.upper()
    )

    predicted = (
        evaluation[
            "predicted_state"
        ]
        .astype(str)
        .str.upper()
    )

    result["n"] = len(
        evaluation
    )

    result["top1"] = float(
        (
            actual
            ==
            predicted
        ).mean()
    )

    if "second_state" in evaluation.columns:

        second = (
            evaluation[
                "second_state"
            ]
            .astype(str)
            .str.upper()
        )

        result["top2"] = float(
            (
                (
                    actual
                    ==
                    predicted
                )
                |
                (
                    actual
                    ==
                    second
                )
            ).mean()
        )

    if "current_state" in evaluation.columns:

        current = (
            evaluation[
                "current_state"
            ]
            .astype(str)
            .str.upper()
        )

        result["baseline"] = float(
            (
                actual
                ==
                current
            ).mean()
        )

    recalls = []

    for state in STATE_LEVELS:

        mask = actual.eq(
            state
        )

        if mask.sum() == 0:
            continue

        recalls.append(
            float(
                (
                    predicted[
                        mask
                    ]
                    ==
                    state
                ).mean()
            )
        )

    if recalls:

        result["balanced"] = float(
            np.mean(
                recalls
            )
        )

    return result


# ============================================================
# ACCIONES
# ============================================================

def get_coach_action(
    coach_data,
    predicted_state,
):

    if coach_data.empty:

        return {
            "title":
                "Sin recomendación disponible",

            "text":
                "No hay una recomendación histórica "
                "asociada a este escenario.",
        }

    state_column = first_existing_column(
        coach_data,
        [
            "predicted_state",
            "future_state",
            "prediction",
        ],
    )

    filtered = coach_data.copy()

    if state_column is not None:

        state_rows = filtered[
            filtered[
                state_column
            ]
            .astype(str)
            .str.upper()
            .eq(
                str(
                    predicted_state
                ).upper()
            )
        ]

        if not state_rows.empty:

            filtered = state_rows

    title_column = first_existing_column(
        filtered,
        [
            "action_title",
            "recommendation_title",
            "coach_action_title",
            "decision_title",
        ],
    )

    text_column = first_existing_column(
        filtered,
        [
            "action_text",
            "recommendation_text",
            "coach_action_text",
            "decision_text",
            "recommended_action",
            "coach_action",
            "recommendation",
        ],
    )

    title = None
    text = None

    if title_column is not None:

        values = (
            filtered[
                title_column
            ]
            .dropna()
            .astype(str)
            .str.strip()
        )

        values = values[
            values.ne("")
        ]

        if not values.empty:

            title = str(
                values
                .value_counts()
                .index[0]
            )

    if text_column is not None:

        values = (
            filtered[
                text_column
            ]
            .dropna()
            .astype(str)
            .str.strip()
        )

        values = values[
            values.ne("")
        ]

        if not values.empty:

            text = str(
                values
                .value_counts()
                .index[0]
            )

    if title is None:

        title = (
            f"Respuesta ante "
            f"{state_display(predicted_state).lower()}"
        )

    if text is None:

        text = (
            "No existe una recomendación textual "
            "registrada para este estado."
        )

    return {
        "title":
            title,

        "text":
            text,
    }


# ============================================================
# IMPORTANCIA XGBOOST
# ============================================================

def get_xgb_importance(
    model,
    top_n=8,
):

    if model is None:

        return pd.DataFrame()

    score = model.get_score(
        importance_type="gain"
    )

    if not score:

        return pd.DataFrame()

    result = pd.DataFrame(
        {
            "feature":
                list(
                    score.keys()
                ),

            "importance":
                list(
                    score.values()
                ),
        }
    )

    result[
        "label"
    ] = result[
        "feature"
    ].map(
        friendly_feature_name
    )

    return (
        result
        .sort_values(
            "importance",
            ascending=False,
        )
        .head(
            top_n
        )
        .sort_values(
            "importance",
            ascending=True,
        )
    )


# ============================================================
# DATOS
# ============================================================

male_simulator = load_csv(
    MALE_SIMULATOR_PATH
)

male_match_history = load_csv(
    MALE_MATCH_HISTORY_PATH
)

male_coach = load_csv(
    MALE_COACH_PATH
)

male_shap = load_csv(
    MALE_SHAP_PATH
)


female_simulator = load_csv(
    FEMALE_SIMULATOR_PATH
)

female_match_history = load_csv(
    FEMALE_MATCH_HISTORY_PATH
)

female_change_history = load_csv(
    FEMALE_CHANGE_HISTORY_PATH
)

female_metrics = load_csv(
    FEMALE_METRICS_PATH
)

female_features_table = load_csv(
    FEMALE_FEATURES_PATH
)

female_selection = load_csv(
    FEMALE_SELECTION_PATH
)


# ============================================================
# MODELOS
# ============================================================

male_model = (
    load_model(
        MALE_MODEL_PATH
    )
    if MALE_MODEL_PATH.exists()
    else None
)

female_model = (
    load_model(
        FEMALE_MODEL_PATH
    )
    if FEMALE_MODEL_PATH.exists()
    else None
)


# ============================================================
# FEATURES
# ============================================================

male_feature_names = (
    list(
        male_model.feature_names
    )
    if (
        male_model is not None
        and
        male_model.feature_names
    )
    else []
)

female_feature_names = (
    list(
        female_model.feature_names
    )
    if (
        female_model is not None
        and
        female_model.feature_names
    )
    else []
)


# ============================================================
# MEDIANAS FEMENIL
# ============================================================

female_medians = {}

if (
    not female_features_table.empty
    and
    all(
        column in
        female_features_table.columns
        for column in [
            "feature",
            "train_median",
        ]
    )
):

    for _, row in female_features_table.iterrows():

        value = pd.to_numeric(
            row[
                "train_median"
            ],
            errors="coerce",
        )

        if np.isfinite(
            value
        ):

            female_medians[
                str(
                    row[
                        "feature"
                    ]
                )
            ] = float(
                value
            )


# ============================================================
# THRESHOLD FEMENIL
# ============================================================

female_threshold = np.nan

if (
    not female_selection.empty
    and
    "selected_threshold"
    in female_selection.columns
):

    female_threshold = pd.to_numeric(
        female_selection[
            "selected_threshold"
        ],
        errors="coerce",
    ).iloc[0]


if not np.isfinite(
    female_threshold
):

    female_threshold = metric_from_table(
        female_metrics,
        "selected_threshold",
    )


# ============================================================
# MÉTRICAS VARONIL
# ============================================================

male_evidence = calculate_male_metrics(
    male_coach
)


# ============================================================
# SESSION
# ============================================================

if "section" not in st.session_state:

    st.session_state[
        "section"
    ] = "Varonil"


# ============================================================
# SIDEBAR BRAND
# ============================================================

with st.sidebar:

    st.html(
        """
        <div class="sidebar-brand">

            <div class="sidebar-logo-row">

                <div class="sidebar-circle">
                    CA
                </div>

                <div>

                    <div class="sidebar-brand-name">
                        CoachLens
                    </div>

                    <div class="sidebar-brand-subtitle">
                        INTELIGENCIA TÁCTICA
                    </div>

                </div>

            </div>

        </div>
        """
    )

    st.divider()


# ============================================================
# SIDEBAR VARONIL
# ============================================================

if st.session_state["section"] == "Varonil":

    opponents = get_historical_opponents(
        male_match_history
    )

    with st.sidebar:

        st.html(
            """
            <div class="sidebar-section-title">
                ESCENARIO VARONIL
            </div>
            """
        )

        venue_display = st.selectbox(
            "Condición",
            [
                "Local",
                "Visitante",
            ],
            key="male_venue",
        )

        venue = (
            "HOME"
            if venue_display == "Local"
            else
            "AWAY"
        )

        opponent = st.selectbox(
            "Rival",
            opponents,
            key="male_opponent",
        )

        minute = st.slider(
            "Minuto",
            1,
            90,
            60,
            key="male_minute",
        )

        st.markdown(
            "**Marcador**"
        )

        score_left, score_right = st.columns(
            2
        )

        with score_left:

            america_goals = st.number_input(
                "América",
                min_value=0,
                max_value=10,
                value=0,
                step=1,
                key="male_goals",
            )

        with score_right:

            opponent_goals = st.number_input(
                opponent,
                min_value=0,
                max_value=10,
                value=0,
                step=1,
                key="male_opponent_goals",
            )

        visible_state = st.selectbox(
            "Estado táctico actual",
            list(
                STATE_DISPLAY.values()
            ),
            key="male_state",
        )

        current_state = STATE_REVERSE[
            visible_state
        ]

        st.html(
            """
            <div class="sidebar-status">
                MODELO VARONIL ACTIVO
            </div>
            """
        )


# ============================================================
# SIDEBAR FEMENIL
# ============================================================

elif st.session_state["section"] == "Femenil":

    opponents = get_historical_opponents(
        female_match_history
    )

    with st.sidebar:

        st.html(
            """
            <div class="sidebar-section-title">
                ESCENARIO FEMENIL
            </div>
            """
        )

        venue_display = st.selectbox(
            "Condición",
            [
                "Local",
                "Visitante",
            ],
            key="female_venue",
        )

        venue = (
            "HOME"
            if venue_display == "Local"
            else
            "AWAY"
        )

        opponent = st.selectbox(
            "Rival",
            opponents,
            key="female_opponent",
        )

        minute = st.slider(
            "Minuto",
            1,
            90,
            60,
            key="female_minute",
        )

        st.markdown(
            "**Marcador**"
        )

        score_left, score_right = st.columns(
            2
        )

        with score_left:

            america_goals = st.number_input(
                "América W",
                min_value=0,
                max_value=10,
                value=0,
                step=1,
                key="female_goals",
            )

        with score_right:

            opponent_goals = st.number_input(
                opponent,
                min_value=0,
                max_value=10,
                value=0,
                step=1,
                key="female_opponent_goals",
            )

        visible_state = st.selectbox(
            "Estado táctico actual",
            list(
                STATE_DISPLAY.values()
            ),
            key="female_state",
        )

        current_state = STATE_REVERSE[
            visible_state
        ]

        st.html(
            """
            <div class="sidebar-status">
                MODELO FEMENIL ACTIVO
            </div>
            """
        )


# ============================================================
# HERO
# ============================================================

st.html(
    """
    <div class="america-header">

        <div class="header-kicker">
            INTELIGENCIA TÁCTICA · CLUB AMÉRICA
        </div>

        <div class="header-title">
            CoachLens
        </div>

        <div class="header-subtitle">
            Anticipa cambios tácticos ·
            Entiende el contexto ·
            Apoya decisiones
        </div>

    </div>
    """
)


# ============================================================
# NAVEGACIÓN
# ============================================================

nav_options = [
    "Varonil",
    "Femenil",
    "Revisión histórica",
    "Modelo",
]


selected_section = st.radio(
    "Navegación principal",
    nav_options,
    index=nav_options.index(
        st.session_state[
            "section"
        ]
    ),
    horizontal=True,
    label_visibility="collapsed",
    key="top_navigation",
)


if (
    selected_section
    !=
    st.session_state[
        "section"
    ]
):

    st.session_state[
        "section"
    ] = selected_section

    st.rerun()


# ============================================================
# VARONIL
# ============================================================

if st.session_state["section"] == "Varonil":

    st.html(
        """
        <div class="page-kicker">
            MODELO VARONIL
        </div>

        <div class="page-title">
            Pronóstico táctico a 5 minutos
        </div>

        <div class="page-description">
            El modelo estima cuál será el estado táctico
            más probable de América durante los próximos
            cinco minutos.
        </div>
        """
    )

    if (
        male_model is None
        or
        male_simulator.empty
        or
        not male_feature_names
    ):

        st.error(
            "No están disponibles todos los archivos del modelo Varonil."
        )

    else:

        score_state = (
            int(
                america_goals
            )
            -
            int(
                opponent_goals
            )
        )

        historical_matches = count_matches(
            male_match_history,
            opponent,
        )

        try:

            (
                feature_row,
                neighbors,
                similarity,
            ) = build_context_from_history(
                male_simulator,
                opponent,
                venue,
                minute,
                score_state,
                current_state,
                male_feature_names,
            )

            feature_row = feature_row[
                male_feature_names
            ].copy()

            dmatrix = xgb.DMatrix(
                feature_row.to_numpy(
                    dtype=float
                ),
                feature_names=
                    male_feature_names,
            )

            probabilities = np.asarray(
                male_model.predict(
                    dmatrix
                )
            )

            if probabilities.ndim == 1:

                probabilities = (
                    probabilities.reshape(
                        1,
                        -1,
                    )
                )

            probabilities = probabilities[
                0
            ]

            ranking = np.argsort(
                probabilities
            )[::-1]

            first_index = int(
                ranking[
                    0
                ]
            )

            second_index = int(
                ranking[
                    1
                ]
            )

            predicted_state = STATE_LEVELS[
                first_index
            ]

            second_state = STATE_LEVELS[
                second_index
            ]

            confidence = float(
                probabilities[
                    first_index
                ]
            )

            second_probability = float(
                probabilities[
                    second_index
                ]
            )

            margin = (
                confidence
                -
                second_probability
            )

            action = get_coach_action(
                male_coach,
                predicted_state,
            )


            c1, c2, c3 = st.columns(
                [
                    1,
                    1,
                    1.25,
                ],
                gap="medium",
            )

            with c1:

                st.html(
                    f"""
                    <div class="result-card">

                        <div class="result-card-label">
                            Estado más probable
                        </div>

                        <div class="result-card-value prediction-state">
                            {state_display(predicted_state)}
                        </div>

                        <div class="result-card-text">
                            Probabilidad:
                            {safe_percent(confidence)}
                        </div>

                    </div>
                    """
                )

            with c2:

                st.html(
                    f"""
                    <div class="result-card">

                        <div class="result-card-label">
                            Segunda alternativa
                        </div>

                        <div class="result-card-value prediction-state">
                            {state_display(second_state)}
                        </div>

                        <div class="result-card-text">
                            Probabilidad:
                            {safe_percent(second_probability)}
                        </div>

                    </div>
                    """
                )

            with c3:

                st.html(
                    f"""
                    <div class="result-card action-card">

                        <div class="result-card-label">
                            Acción sugerida
                        </div>

                        <div class="action-title">
                            {action["title"]}
                        </div>

                        <div class="action-text">
                            {action["text"]}
                        </div>

                    </div>
                    """
                )


            st.write("")


            left, right = st.columns(
                2,
                gap="large",
            )

            with left:

                st.html(
                    f"""
                    <div class="confidence-card">

                        <div class="confidence-label">
                            Confianza del pronóstico
                        </div>

                        <div class="confidence-number">
                            {safe_percent(confidence)}
                        </div>

                        <div class="confidence-details">

                            Segunda opción:
                            <strong>
                                {state_display(second_state)}
                            </strong>

                            ·
                            {safe_percent(second_probability)}

                            <br>

                            Margen:
                            <strong>
                                {safe_percent(margin)}
                            </strong>

                        </div>

                    </div>
                    """
                )

            with right:

                st.html(
                    f"""
                    <div class="similarity-card">

                        <div class="similarity-label">
                            Similitud del contexto
                        </div>

                        <div class="similarity-number">
                            {similarity:.1f}%
                        </div>

                        <div class="similarity-bar">

                            <div
                                class="similarity-fill"
                                style="width:{similarity:.1f}%;">
                            </div>

                        </div>

                        <div class="similarity-status">
                            Percentil histórico
                        </div>

                        <div class="similarity-description">
                            Posición relativa del escenario
                            frente a los contextos históricos.
                        </div>

                        <div class="similarity-history">
                            Partidos jugados frente a
                            {opponent}: {historical_matches}
                        </div>

                    </div>
                    """
                )

        except Exception as error:

            st.error(
                f"No se pudo generar el análisis Varonil: {error}"
            )


# ============================================================
# FEMENIL
# ============================================================

elif st.session_state["section"] == "Femenil":

    st.html(
        """
        <div class="page-kicker">
            MODELO FEMENIL
        </div>

        <div class="page-title">
            Riesgo de cambio táctico
        </div>

        <div class="page-description">
            El modelo estima la probabilidad de que
            América Femenil cambie de estado táctico
            durante los próximos cinco minutos.
        </div>
        """
    )

    female_ready = (
        female_model is not None
        and
        not female_simulator.empty
        and
        female_feature_names
        and
        np.isfinite(
            female_threshold
        )
    )

    if not female_ready:

        st.error(
            "No están disponibles todos los archivos del modelo Femenil."
        )

    else:

        score_state = (
            int(
                america_goals
            )
            -
            int(
                opponent_goals
            )
        )

        historical_matches = count_matches(
            female_match_history,
            opponent,
        )

        try:

            (
                feature_row,
                neighbors,
                similarity,
            ) = build_context_from_history(
                female_simulator,
                opponent,
                venue,
                minute,
                score_state,
                current_state,
                female_feature_names,
                female_medians,
            )

            for feature in female_feature_names:

                if feature not in feature_row.columns:

                    feature_row[
                        feature
                    ] = female_medians.get(
                        feature,
                        0.0,
                    )

            feature_row = feature_row[
                female_feature_names
            ].copy()

            dmatrix = xgb.DMatrix(
                feature_row.to_numpy(
                    dtype=float
                ),
                feature_names=
                    female_feature_names,
            )

            probability_change = float(
                np.asarray(
                    female_model.predict(
                        dmatrix
                    )
                ).reshape(
                    -1
                )[0]
            )

            predicted_change = (
                probability_change
                >=
                female_threshold
            )

            difference_threshold = (
                probability_change
                -
                female_threshold
            )


            if predicted_change:

                signal_title = (
                    "CAMBIO PROBABLE"
                )

                signal_text = (
                    "La señal supera el umbral de decisión "
                    "seleccionado durante VALIDATION."
                )

            else:

                signal_title = (
                    "ESTABILIDAD PROBABLE"
                )

                signal_text = (
                    "La señal permanece por debajo del "
                    "umbral de decisión seleccionado "
                    "durante VALIDATION."
                )


            historical_transition = (
                calculate_historical_transition(
                    neighbors,
                    current_state,
                )
            )

            transition_state = (
                historical_transition[
                    "state"
                ]
            )

            transition_share = (
                historical_transition[
                    "share"
                ]
            )

            transition_n = (
                historical_transition[
                    "observations"
                ]
            )


            if (
                transition_state is not None
                and
                np.isfinite(
                    transition_share
                )
            ):

                transition_title = (
                    state_display(
                        transition_state
                    )
                )

                transition_percent = (
                    safe_percent(
                        transition_share
                    )
                )

                female_action = (
                    get_coach_action(
                        male_coach,
                        transition_state,
                    )
                )

            else:

                transition_title = (
                    "SIN PATRÓN DOMINANTE"
                )

                transition_percent = (
                    "N/D"
                )

                female_action = {
                    "title":
                        "Sin transición dominante",

                    "text":
                        "No existe suficiente evidencia "
                        "histórica para asociar una acción "
                        "a un estado posterior dominante.",
                }


            f1, f2, f3 = st.columns(
                [
                    1,
                    1,
                    1.25,
                ],
                gap="medium",
            )


            with f1:

                st.html(
                    f"""
                    <div class="result-card">

                        <div class="result-card-label">
                            Probabilidad de cambio
                        </div>

                        <div class="result-card-value">
                            {safe_percent(probability_change)}
                        </div>

                        <div class="result-card-text">
                            Horizonte:
                            próximos cinco minutos.
                        </div>

                    </div>
                    """
                )


            with f2:

                st.html(
                    f"""
                    <div class="result-card">

                        <div class="result-card-label">
                            Escenario histórico más frecuente
                        </div>

                        <div class="result-card-value prediction-state">
                            {transition_title}
                        </div>

                        <div class="result-card-text">

                            Participación:
                            {transition_percent}

                            <br>

                            Casos similares:
                            {transition_n}

                        </div>

                    </div>
                    """
                )


            with f3:

                st.html(
                    f"""
                    <div class="result-card action-card">

                        <div class="result-card-label">
                            Acción sugerida
                        </div>

                        <div class="action-title">
                            {female_action["title"]}
                        </div>

                        <div class="action-text">
                            {female_action["text"]}
                        </div>

                    </div>
                    """
                )


            st.write("")


            female_left, female_right = (
                st.columns(
                    2,
                    gap="large",
                )
            )


            with female_left:

                st.html(
                    f"""
                    <div class="confidence-card">

                        <div class="confidence-label">
                            Señal del modelo
                        </div>

                        <div class="confidence-number">
                            {signal_title}
                        </div>

                        <div class="confidence-details">

                            Probabilidad de cambio:
                            <strong>
                                {safe_percent(probability_change)}
                            </strong>

                            <br>

                            Umbral de decisión:
                            <strong>
                                {safe_percent(female_threshold)}
                            </strong>

                            <br>

                            Diferencia:
                            <strong>
                                {percentage_points(difference_threshold)}
                            </strong>

                        </div>

                        <div class="confidence-description">
                            {signal_text}
                        </div>

                    </div>
                    """
                )


            with female_right:

                st.html(
                    f"""
                    <div class="similarity-card">

                        <div class="similarity-label">
                            Similitud del contexto
                        </div>

                        <div class="similarity-number">
                            {similarity:.1f}%
                        </div>

                        <div class="similarity-bar">

                            <div
                                class="similarity-fill"
                                style="width:{similarity:.1f}%;">
                            </div>

                        </div>

                        <div class="similarity-status">
                            Percentil histórico
                        </div>

                        <div class="similarity-description">
                            Posición relativa del escenario
                            frente a los contextos históricos
                            disponibles en TRAIN y VALIDATION.
                        </div>

                        <div class="similarity-history">
                            Partidos jugados frente a
                            {opponent}: {historical_matches}
                        </div>

                    </div>
                    """
                )

        except Exception as error:

            st.error(
                f"No se pudo generar el análisis Femenil: {error}"
            )


# ============================================================
# REVISIÓN HISTÓRICA
# ============================================================

elif st.session_state["section"] == "Revisión histórica":

    st.html(
        """
        <div class="page-kicker">
            VALIDACIÓN HISTÓRICA
        </div>

        <div class="page-title">
            Revisión de partidos reales
        </div>

        <div class="page-description">
            Compara las predicciones guardadas con lo
            realmente observado en partidos del conjunto TEST.
        </div>
        """
    )


    review_mode = st.radio(
        "Equipo",
        [
            "Varonil",
            "Femenil",
        ],
        horizontal=True,
        key="review_mode",
    )


    # ========================================================
    # HISTÓRICO VARONIL
    # ========================================================

    if review_mode == "Varonil":

        if male_coach.empty:

            st.warning(
                "No hay historial Varonil disponible."
            )

        else:

            history = male_coach.copy()

            history[
                "match_date"
            ] = pd.to_datetime(
                history[
                    "match_date"
                ],
                errors="coerce",
            )

            history[
                "_match_id"
            ] = history[
                "match_id"
            ].map(
                normalize_match_id
            )

            played_ids = get_played_match_ids(
                male_match_history
            )

            if played_ids:

                history = history[
                    history[
                        "_match_id"
                    ].isin(
                        played_ids
                    )
                ].copy()

            matches = (
                history[
                    [
                        "_match_id",
                        "match_date",
                        "opponent",
                    ]
                ]
                .drop_duplicates()
                .sort_values(
                    "match_date",
                    ascending=False,
                )
                .copy()
            )

            matches[
                "label"
            ] = (
                matches[
                    "match_date"
                ]
                .dt.strftime(
                    "%Y-%m-%d"
                )
                +
                " · América vs "
                +
                matches[
                    "opponent"
                ]
                .astype(str)
            )

            selected_match = st.selectbox(
                "Partido",
                matches[
                    "label"
                ].tolist(),
                key="male_review_match",
            )

            match_id = (
                matches.loc[
                    matches[
                        "label"
                    ]
                    ==
                    selected_match,
                    "_match_id",
                ]
                .iloc[0]
            )

            match_data = history[
                history[
                    "_match_id"
                ]
                ==
                match_id
            ].copy()

            minutes = sorted(
                pd.to_numeric(
                    match_data[
                        "minute"
                    ],
                    errors="coerce",
                )
                .dropna()
                .astype(int)
                .unique()
                .tolist()
            )

            selected_minute = st.select_slider(
                "Minuto",
                options=
                    minutes,
                value=
                    minutes[
                        len(minutes)
                        //
                        2
                    ],
                key="male_review_minute",
            )

            row = match_data[
                pd.to_numeric(
                    match_data[
                        "minute"
                    ],
                    errors="coerce",
                )
                ==
                selected_minute
            ].iloc[0]

            current = str(
                row[
                    "current_state"
                ]
            ).upper()

            predicted = str(
                row[
                    "predicted_state"
                ]
            ).upper()

            second = str(
                row[
                    "second_state"
                ]
            ).upper()

            actual = str(
                row[
                    "actual_future_state"
                ]
            ).upper()

            if actual == predicted:

                result = (
                    "ACIERTO EXACTO"
                )

            elif actual == second:

                result = (
                    "ACIERTO TOP-2"
                )

            else:

                result = (
                    "NO ACIERTO"
                )


            r1, r2, r3, r4 = st.columns(
                4,
                gap="medium",
            )


            cards = [
                (
                    r1,
                    "Estado actual",
                    state_display(
                        current
                    ),
                ),
                (
                    r2,
                    "Primera predicción",
                    state_display(
                        predicted
                    ),
                ),
                (
                    r3,
                    "Estado real +5 min",
                    state_display(
                        actual
                    ),
                ),
                (
                    r4,
                    "Resultado",
                    result,
                ),
            ]


            for (
                column,
                label,
                value,
            ) in cards:

                with column:

                    st.html(
                        f"""
                        <div class="metric-card">

                            <div class="metric-label">
                                {label}
                            </div>

                            <div class="history-state-value">
                                {value}
                            </div>

                        </div>
                        """
                    )


            st.write("")


            st.html(
                f"""
                <div class="history-explanation">

                    <div class="history-explanation-title">
                        ¿Cómo interpretar el resultado?
                    </div>

                    <div class="history-explanation-grid">

                        <div class="history-explanation-item">

                            <div class="history-explanation-badge exact">
                                ACIERTO EXACTO · TOP-1
                            </div>

                            <div class="history-explanation-text">
                                El estado observado coincide
                                exactamente con la primera opción
                                del modelo.
                            </div>

                        </div>


                        <div class="history-explanation-item">

                            <div class="history-explanation-badge top2">
                                ACIERTO TOP-2
                            </div>

                            <div class="history-explanation-text">
                                El estado real no fue la primera
                                opción, pero sí apareció como la
                                segunda alternativa más probable.
                            </div>

                        </div>


                        <div class="history-explanation-item">

                            <div class="history-explanation-badge miss">
                                NO ACIERTO
                            </div>

                            <div class="history-explanation-text">
                                El estado observado no apareció
                                entre las dos principales
                                alternativas del modelo.
                            </div>

                        </div>

                    </div>


                    <div
                        style="
                            margin-top: 20px;
                            color: #40536C;
                            font-size: 15px;
                            line-height: 1.5;
                        "
                    >

                        Rendimiento global en TEST:

                        <strong style="color:#061E39;">
                            Top-1:
                            {safe_percent(male_evidence["top1"])}
                        </strong>

                        ·

                        <strong style="color:#061E39;">
                            Top-2:
                            {safe_percent(male_evidence["top2"])}
                        </strong>

                    </div>

                </div>
                """
            )


    # ========================================================
    # HISTÓRICO FEMENIL
    # ========================================================

    else:

        if female_change_history.empty:

            st.warning(
                "No hay historial Femenil disponible."
            )

        else:

            history = (
                female_change_history
                .copy()
            )

            history[
                "match_date"
            ] = pd.to_datetime(
                history[
                    "match_date"
                ],
                errors="coerce",
            )

            matches = (
                history[
                    [
                        "match_id",
                        "match_date",
                        "opponent",
                    ]
                ]
                .drop_duplicates()
                .sort_values(
                    "match_date",
                    ascending=False,
                )
                .copy()
            )

            matches[
                "label"
            ] = (
                matches[
                    "match_date"
                ]
                .dt.strftime(
                    "%Y-%m-%d"
                )
                +
                " · América W vs "
                +
                matches[
                    "opponent"
                ]
                .astype(str)
            )

            selected_match = st.selectbox(
                "Partido",
                matches[
                    "label"
                ].tolist(),
                key="female_review_match",
            )

            match_id = (
                matches.loc[
                    matches[
                        "label"
                    ]
                    ==
                    selected_match,
                    "match_id",
                ]
                .iloc[0]
            )

            match_data = history[
                history[
                    "match_id"
                ]
                ==
                match_id
            ].copy()

            minutes = sorted(
                pd.to_numeric(
                    match_data[
                        "minute"
                    ],
                    errors="coerce",
                )
                .dropna()
                .astype(int)
                .unique()
                .tolist()
            )

            selected_minute = st.select_slider(
                "Minuto",
                options=
                    minutes,
                value=
                    minutes[
                        len(minutes)
                        //
                        2
                    ],
                key="female_review_minute",
            )

            row = match_data[
                pd.to_numeric(
                    match_data[
                        "minute"
                    ],
                    errors="coerce",
                )
                ==
                selected_minute
            ].iloc[0]

            current = str(
                row[
                    "current_state"
                ]
            ).upper()

            probability = float(
                row[
                    "probability_change"
                ]
            )

            predicted_change_raw = str(
                row[
                    "predicted_change"
                ]
            ).upper()

            actual_change_raw = str(
                row[
                    "actual_change"
                ]
            ).upper()

            actual_change = (
                actual_change_raw
                .replace(
                    "_",
                    " "
                )
            )

            result = (
                "ACIERTO"
                if predicted_change_raw
                ==
                actual_change_raw
                else
                "NO ACIERTO"
            )


            fr1, fr2, fr3, fr4 = st.columns(
                4,
                gap="medium",
            )


            female_cards = [
                (
                    fr1,
                    "Estado actual",
                    state_display(
                        current
                    ),
                ),
                (
                    fr2,
                    "Probabilidad de cambio",
                    safe_percent(
                        probability
                    ),
                ),
                (
                    fr3,
                    "Cambio real +5 min",
                    actual_change,
                ),
                (
                    fr4,
                    "Resultado",
                    result,
                ),
            ]


            for (
                column,
                label,
                value,
            ) in female_cards:

                with column:

                    st.html(
                        f"""
                        <div class="metric-card">

                            <div class="metric-label">
                                {label}
                            </div>

                            <div class="history-state-value">
                                {value}
                            </div>

                        </div>
                        """
                    )


            st.write("")


            st.html(
                f"""
                <div class="history-explanation">

                    <div class="history-explanation-title">
                        ¿Cómo interpretar el resultado?
                    </div>

                    <div class="history-explanation-grid">

                        <div class="history-explanation-item">

                            <div class="history-explanation-badge exact">
                                ACIERTO
                            </div>

                            <div class="history-explanation-text">
                                El modelo clasificó correctamente
                                si el estado táctico cambiaría
                                o permanecería estable durante
                                los siguientes cinco minutos.
                            </div>

                        </div>

                        <div class="history-explanation-item">

                            <div class="history-explanation-badge miss">
                                NO ACIERTO
                            </div>

                            <div class="history-explanation-text">
                                La clasificación del modelo no
                                coincidió con el comportamiento
                                táctico realmente observado.
                            </div>

                        </div>

                        <div class="history-explanation-item">

                            <div class="history-explanation-badge top2">
                                PROBABILIDAD
                            </div>

                            <div class="history-explanation-text">

                                El modelo asignó una probabilidad
                                de cambio de

                                <strong>
                                    {safe_percent(probability)}
                                </strong>.

                                El umbral aprendido determina
                                si la señal se clasifica como
                                cambio o estabilidad.

                            </div>

                        </div>

                    </div>

                </div>
                """
            )


# ============================================================
# MODELO
# ============================================================

elif st.session_state["section"] == "Modelo":

    st.html(
        """
        <div class="page-kicker">
            EVIDENCIA Y METODOLOGÍA
        </div>

        <div class="page-title">
            Rendimiento del modelo
        </div>

        <div class="page-description">
            Evaluación temporal sobre conjuntos TEST
            separados del entrenamiento.
        </div>
        """
    )


    model_mode = st.radio(
        "Equipo",
        [
            "Varonil",
            "Femenil",
        ],
        horizontal=True,
        key="model_mode",
    )


    # ========================================================
    # MODELO VARONIL
    # ========================================================

    if model_mode == "Varonil":

        mc1, mc2, mc3, mc4 = (
            st.columns(
                4,
                gap="medium",
            )
        )

        male_cards = [
            (
                mc1,
                "Exactitud Top-1",
                safe_percent(
                    male_evidence[
                        "top1"
                    ]
                ),
            ),
            (
                mc2,
                "Exactitud Top-2",
                safe_percent(
                    male_evidence[
                        "top2"
                    ]
                ),
            ),
            (
                mc3,
                "Baseline",
                safe_percent(
                    male_evidence[
                        "baseline"
                    ]
                ),
            ),
            (
                mc4,
                "Exactitud balanceada",
                safe_percent(
                    male_evidence[
                        "balanced"
                    ]
                ),
            ),
        ]


        for (
            column,
            label,
            value,
        ) in male_cards:

            with column:

                st.html(
                    f"""
                    <div class="metric-card">

                        <div class="metric-label">
                            {label}
                        </div>

                        <div class="metric-value">
                            {value}
                        </div>

                    </div>
                    """
                )


        st.caption(
            f"Evaluación sobre "
            f"{male_evidence['n']:,} "
            f"observaciones TEST."
        )


        st.write("")


        st.html(
            """
            <div class="history-explanation">

                <div class="history-explanation-title">
                    Interpretación
                </div>

                <div class="history-explanation-text">

                    <strong>Top-1</strong>
                    representa el acierto exacto.

                    <br><br>

                    <strong>Top-2</strong>
                    indica si el estado real apareció
                    entre las dos alternativas más probables.

                    <br><br>

                    <strong>Baseline</strong>
                    utiliza el estado táctico actual
                    como predicción futura.

                    <br><br>

                    <strong>Exactitud balanceada</strong>
                    promedia el desempeño entre
                    las cinco clases tácticas.

                </div>

            </div>
            """
        )


        st.write("")


        st.markdown(
            "### Influencia global · Modelo Varonil"
        )


        if (
            not male_shap.empty
            and
            all(
                column in
                male_shap.columns
                for column in [
                    "feature",
                    "mean_abs_shap",
                ]
            )
        ):

            importance = (
                male_shap
                .sort_values(
                    "mean_abs_shap",
                    ascending=False,
                )
                .head(
                    8
                )
                .copy()
            )

            importance[
                "label"
            ] = importance[
                "feature"
            ].map(
                friendly_feature_name
            )

            importance = (
                importance
                .sort_values(
                    "mean_abs_shap"
                )
            )

            figure = px.bar(
                importance,
                x="mean_abs_shap",
                y="label",
                orientation="h",
            )

            figure.update_traces(
                marker_color=
                    AMERICA_YELLOW
            )

            figure.update_layout(
                height=350,
                paper_bgcolor=
                    "rgba(0,0,0,0)",
                plot_bgcolor=
                    "rgba(0,0,0,0)",
                margin=dict(
                    l=10,
                    r=10,
                    t=10,
                    b=40,
                ),
                xaxis_title=
                    "Influencia SHAP promedio",
                yaxis_title="",
                font=dict(
                    color=
                        TEXT_DARK
                ),
            )

            st.plotly_chart(
                figure,
                use_container_width=True,
                config={
                    "displayModeBar":
                        False
                },
            )


    # ========================================================
    # MODELO FEMENIL
    # ========================================================

    else:

        accuracy = metric_from_table(
            female_metrics,
            "accuracy",
        )

        balanced = metric_from_table(
            female_metrics,
            "balanced_accuracy",
        )

        auc = metric_from_table(
            female_metrics,
            "auc",
        )

        f1_change = metric_from_table(
            female_metrics,
            "f1_change",
        )

        baseline_balanced = (
            metric_from_table(
                female_metrics,
                "majority_baseline_balanced",
            )
        )

        test_observations = (
            metric_from_table(
                female_metrics,
                "test_observations",
            )
        )


        fc1, fc2, fc3, fc4 = (
            st.columns(
                4,
                gap="medium",
            )
        )


        female_cards = [
            (
                fc1,
                "Accuracy",
                safe_percent(
                    accuracy
                ),
            ),
            (
                fc2,
                "Exactitud balanceada",
                safe_percent(
                    balanced
                ),
            ),
            (
                fc3,
                "AUC",
                safe_percent(
                    auc
                ),
            ),
            (
                fc4,
                "F1 · cambio",
                safe_percent(
                    f1_change
                ),
            ),
        ]


        for (
            column,
            label,
            value,
        ) in female_cards:

            with column:

                st.html(
                    f"""
                    <div class="metric-card">

                        <div class="metric-label">
                            {label}
                        </div>

                        <div class="metric-value">
                            {value}
                        </div>

                    </div>
                    """
                )


        if np.isfinite(
            test_observations
        ):

            test_text = (
                f"{int(test_observations):,}"
            )

        else:

            test_text = "N/D"


        st.caption(
            f"Evaluación sobre {test_text} "
            f"observaciones TEST · "
            f"Baseline balanceado: "
            f"{safe_percent(baseline_balanced)}."
        )


        st.write("")


        st.html(
            """
            <div class="history-explanation">

                <div class="history-explanation-title">
                    Interpretación
                </div>

                <div class="history-explanation-text">

                    El modelo Femenil estima si habrá un
                    <strong>cambio táctico</strong>
                    durante los siguientes cinco minutos.

                    <br><br>

                    <strong>Accuracy</strong>
                    mide el porcentaje total de
                    clasificaciones correctas.

                    <br><br>

                    <strong>Exactitud balanceada</strong>
                    considera por igual CAMBIA
                    y NO CAMBIA.

                    <br><br>

                    <strong>AUC</strong>
                    mide la capacidad para ordenar escenarios
                    con mayor y menor riesgo de cambio.

                    <br><br>

                    <strong>F1</strong>
                    combina precisión y sensibilidad
                    para la clase CAMBIA.

                </div>

            </div>
            """
        )


        st.write("")


        st.markdown(
            "### Influencia global · Modelo Femenil"
        )


        female_importance = (
            get_xgb_importance(
                female_model,
                top_n=8,
            )
        )


        if female_importance.empty:

            st.info(
                "No hay importancia disponible."
            )

        else:

            figure = px.bar(
                female_importance,
                x="importance",
                y="label",
                orientation="h",
            )

            figure.update_traces(
                marker_color=
                    AMERICA_YELLOW
            )

            figure.update_layout(
                height=350,
                paper_bgcolor=
                    "rgba(0,0,0,0)",
                plot_bgcolor=
                    "rgba(0,0,0,0)",
                margin=dict(
                    l=10,
                    r=10,
                    t=10,
                    b=40,
                ),
                xaxis_title=
                    "Ganancia promedio en XGBoost",
                yaxis_title="",
                font=dict(
                    color=
                        TEXT_DARK
                ),
            )

            st.plotly_chart(
                figure,
                use_container_width=True,
                config={
                    "displayModeBar":
                        False
                },
            )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "COACHLENS · CLUB AMÉRICA · "
    "XGBoost · StatsBomb Events + 360"
)