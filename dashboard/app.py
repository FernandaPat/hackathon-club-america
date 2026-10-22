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
# ============================================================
# ANÁLISIS COMPLEMENTARIO
# ============================================================

COMPLEMENTARY_SUBSTITUTIONS_PATH = (
    DATA_DIR
    / "complementary_substitutions.csv"
)

COMPLEMENTARY_SET_PIECES_PATH = (
    DATA_DIR
    / "complementary_set_pieces.csv"
)

COMPLEMENTARY_SET_PIECE_SUMMARY_PATH = (
    DATA_DIR
    / "complementary_set_piece_summary.csv"
)


# VARONIL

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


# FEMENIL

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
# PLAYBOOK
# ============================================================

TACTICAL_PLAYBOOK = {

    "CONTROL": {
        "title": "Sostener el control",
        "text": (
            "Mantener la posesión con paciencia, "
            "utilizar la amplitud y acelerar únicamente "
            "cuando aparezca una ventaja clara."
        ),
    },

    "PRESSURE": {
        "title": "Subir la presión",
        "text": (
            "Cerrar líneas interiores y activar la presión "
            "cuando el rival juegue hacia atrás o tenga "
            "un control incómodo."
        ),
    },

    "TRANSITION": {
        "title": "Atacar la transición",
        "text": (
            "Priorizar un primer pase limpio tras recuperar "
            "y atacar rápidamente los espacios que deja "
            "el rival."
        ),
    },

    "PROTECTION": {
        "title": "Cerrar espacios",
        "text": (
            "Proteger el carril central, reducir distancias "
            "entre líneas y evitar pérdidas en zonas "
            "comprometidas."
        ),
    },

    "URGENCY": {
        "title": "Aumentar el ritmo",
        "text": (
            "Acelerar la circulación, ocupar zonas de remate "
            "y reaccionar rápidamente después de una pérdida."
        ),
    },
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

    value = (
        str(value)
        .strip()
        .lower()
    )

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

    value = (
        str(value)
        .upper()
    )

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

    return pd.to_numeric(
        result["value"],
        errors="coerce",
    ).iloc[0]


def interpret_change_value(value):

    text = (
        str(value)
        .strip()
        .upper()
    )

    positives = {
        "1",
        "1.0",
        "TRUE",
        "T",
        "YES",
        "CAMBIA",
        "CHANGE",
    }

    return (
        "CAMBIA"
        if text in positives
        else "NO CAMBIA"
    )


# ============================================================
# SCORE STEPPER
# ============================================================

def score_stepper(
    label,
    key,
    default=0,
    min_value=0,
    max_value=10,
):

    if key not in st.session_state:
        st.session_state[key] = default

    st.html(
        f"""
        <div class="score-team-label">
            {label}
        </div>
        """
    )

    minus_col, value_col, plus_col = (
        st.columns(
            [
                1,
                1.3,
                1,
            ],
            gap="small",
        )
    )

    with minus_col:

        if st.button(
            "−",
            key=f"{key}_minus",
            use_container_width=True,
        ):

            st.session_state[key] = max(
                min_value,
                int(
                    st.session_state[key]
                )
                -
                1,
            )

    with value_col:

        st.html(
            f"""
            <div class="score-value-box">
                {int(st.session_state[key])}
            </div>
            """
        )

    with plus_col:

        if st.button(
            "+",
            key=f"{key}_plus",
            use_container_width=True,
        ):

            st.session_state[key] = min(
                max_value,
                int(
                    st.session_state[key]
                )
                +
                1,
            )

    return int(
        st.session_state[key]
    )


# ============================================================
# FEATURE NAMES
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
        return mapping[feature]

    result = (
        str(feature)
        .replace(
            "america_",
            "",
        )
        .replace(
            "rival_",
            "rival ",
        )
        .replace(
            "_last5",
            "",
        )
        .replace(
            "_last3",
            " reciente",
        )
        .replace(
            "_diff",
            "",
        )
        .replace(
            "_trend",
            " tendencia",
        )
        .replace(
            "_",
            " ",
        )
        .strip()
    )

    return result.capitalize()


# ============================================================
# MATCH HELPERS
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
                <=
                today
            )
        ].copy()

    return result


def get_historical_opponents(matches):

    data = played_matches_only(
        matches
    )

    if (
        data.empty
        or
        "opponent" not in data.columns
    ):
        return []

    return sorted(
        data[
            "opponent"
        ]
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

    if (
        data.empty
        or
        "opponent" not in data.columns
    ):
        return 0

    mask = (
        data["opponent"]
        .astype(str)
        .map(
            normalize_text
        )
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
            result[
                "match_id"
            ]
            .nunique()
        )

    return int(
        len(result)
    )


def get_played_match_ids(matches):

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
        data[
            "match_id"
        ]
        .map(
            normalize_match_id
        )
        .tolist()
    )


# ============================================================
# ROBUST SCALE
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

        scale = (
            numeric.std()
        )

    if (
        not np.isfinite(scale)
        or
        scale == 0
    ):

        scale = 1.0

    return float(scale)


# ============================================================
# HISTORICAL CONTEXT
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
                str(
                    venue
                ).upper()
            )
            .astype(float)
        )

    if "opponent" in pool.columns:

        components["opponent"] = (
            pool["opponent"]
            .astype(str)
            .map(
                normalize_text
            )
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

    pool["_distance"] = (
        components.mean(
            axis=1
        )
    )

    distances = (
        pool[
            "_distance"
        ]
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

    selected = (
        pool[
            pool["_distance"]
            <=
            cutoff
        ]
        .copy()
    )

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

    selected[
        "_weight"
    ] = np.exp(
        -
        selected[
            "_distance"
        ]
        /
        distance_scale
    )

    if (
        not np.isfinite(
            selected[
                "_weight"
            ]
        ).all()
        or
        selected[
            "_weight"
        ].sum()
        <=
        0
    ):

        selected[
            "_weight"
        ] = 1.0

    selected[
        "_weight"
    ] = (
        selected[
            "_weight"
        ]
        /
        selected[
            "_weight"
        ].sum()
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
                    "",
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
        selected[
            "_distance"
        ].mean()
    )

    similarity = float(
        (
            distances
            >=
            mean_distance
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
# HISTORICAL TRANSITION
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

    current_state = (
        str(
            current_state
        )
        .upper()
    )

    transition_data = (
        data[
            data[
                target_column
            ].ne(
                current_state
            )
            &
            data[
                target_column
            ].isin(
                STATE_LEVELS
            )
        ]
        .copy()
    )

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
        <=
        0
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

    distribution[
        "share"
    ] = (
        distribution[
            "weight"
        ]
        /
        distribution[
            "weight"
        ].sum()
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

    top_row = (
        distribution.iloc[0]
    )

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
# SIMULATION
# ============================================================

def simulate_male_scenarios(
    probabilities,
    state_levels,
    n_simulations=1000,
    seed=42,
):

    probabilities = np.asarray(
        probabilities,
        dtype=float,
    )

    probabilities = np.clip(
        probabilities,
        0,
        None,
    )

    probabilities = (
        probabilities
        /
        probabilities.sum()
    )

    rng = np.random.default_rng(
        seed
    )

    simulated_states = rng.choice(
        state_levels,
        size=n_simulations,
        p=probabilities,
    )

    results = (
        pd.Series(
            simulated_states
        )
        .value_counts()
        .reindex(
            state_levels,
            fill_value=0,
        )
        .rename_axis(
            "state"
        )
        .reset_index(
            name="simulations"
        )
    )

    results[
        "share"
    ] = (
        results[
            "simulations"
        ]
        /
        n_simulations
    )

    results[
        "percentage"
    ] = (
        results[
            "share"
        ]
        *
        100
    )

    results[
        "label"
    ] = (
        results[
            "state"
        ]
        .map(
            STATE_DISPLAY
        )
    )

    results[
        "text_label"
    ] = (
        results[
            "percentage"
        ]
        .map(
            lambda value:
            f"{value:.1f}%"
        )
    )

    return results


def simulate_female_scenarios(
    probability_change,
    transition_distribution,
    n_simulations=1000,
    seed=42,
):

    rng = np.random.default_rng(
        seed
    )

    change_flags = (
        rng.random(
            n_simulations
        )
        <
        float(
            probability_change
        )
    )

    change_count = int(
        change_flags.sum()
    )

    stable_count = (
        n_simulations
        -
        change_count
    )

    summary = pd.DataFrame(
        {
            "outcome": [
                "CAMBIO",
                "ESTABILIDAD",
            ],
            "simulations": [
                change_count,
                stable_count,
            ],
        }
    )

    summary[
        "share"
    ] = (
        summary[
            "simulations"
        ]
        /
        n_simulations
    )

    summary[
        "percentage"
    ] = (
        summary[
            "share"
        ]
        *
        100
    )

    summary[
        "text_label"
    ] = (
        summary[
            "percentage"
        ]
        .map(
            lambda value:
            f"{value:.1f}%"
        )
    )

    transition_results = (
        pd.DataFrame()
    )

    if (
        change_count > 0
        and
        transition_distribution is not None
        and
        not transition_distribution.empty
    ):

        state_column = (
            "future_tactical_state_5min"
        )

        if (
            state_column
            in
            transition_distribution.columns
        ):

            states = (
                transition_distribution[
                    state_column
                ]
                .astype(str)
                .to_numpy()
            )

            probabilities = (
                pd.to_numeric(
                    transition_distribution[
                        "share"
                    ],
                    errors="coerce",
                )
                .fillna(0)
                .to_numpy(
                    dtype=float
                )
            )

            if probabilities.sum() > 0:

                probabilities = (
                    probabilities
                    /
                    probabilities.sum()
                )

                simulated_transitions = (
                    rng.choice(
                        states,
                        size=change_count,
                        p=probabilities,
                    )
                )

                transition_results = (
                    pd.Series(
                        simulated_transitions
                    )
                    .value_counts()
                    .rename_axis(
                        "state"
                    )
                    .reset_index(
                        name="simulations"
                    )
                )

                transition_results[
                    "share"
                ] = (
                    transition_results[
                        "simulations"
                    ]
                    /
                    change_count
                )

                transition_results[
                    "percentage"
                ] = (
                    transition_results[
                        "share"
                    ]
                    *
                    100
                )

                transition_results[
                    "label"
                ] = (
                    transition_results[
                        "state"
                    ]
                    .map(
                        STATE_DISPLAY
                    )
                )

                transition_results[
                    "text_label"
                ] = (
                    transition_results[
                        "percentage"
                    ]
                    .map(
                        lambda value:
                        f"{value:.1f}%"
                    )
                )

    return (
        summary,
        transition_results,
    )


# ============================================================
# MALE METRICS
# ============================================================

def calculate_male_metrics(data):

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

    evaluation = (
        data
        .dropna(
            subset=required
        )
        .copy()
    )

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

    result[
        "n"
    ] = len(
        evaluation
    )

    result[
        "top1"
    ] = float(
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

        result[
            "top2"
        ] = float(
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

        result[
            "baseline"
        ] = float(
            (
                actual
                ==
                current
            ).mean()
        )

    recalls = []

    for state in STATE_LEVELS:

        mask = (
            actual.eq(
                state
            )
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

        result[
            "balanced"
        ] = float(
            np.mean(
                recalls
            )
        )

    return result


# ============================================================
# ACTION
# ============================================================

def get_coach_action(state):

    state = (
        str(state)
        .upper()
    )

    return TACTICAL_PLAYBOOK.get(
        state,
        {
            "title":
                "Monitorear el escenario",

            "text":
                "Revisar la evolución del contexto antes "
                "de realizar un ajuste táctico.",
        },
    )


# ============================================================
# IMPORTANCE
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
    ] = (
        result[
            "feature"
        ]
        .map(
            friendly_feature_name
        )
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
# LOAD DATA
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
# DATOS COMPLEMENTARIOS
# ============================================================

complementary_substitutions = load_csv(
    COMPLEMENTARY_SUBSTITUTIONS_PATH
)

complementary_set_pieces = load_csv(
    COMPLEMENTARY_SET_PIECES_PATH
)

complementary_set_piece_summary = load_csv(
    COMPLEMENTARY_SET_PIECE_SUMMARY_PATH
)

# ============================================================
# MODELS
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
# FEMALE MEDIANS
# ============================================================

female_medians = {}

if (
    not female_features_table.empty
    and
    all(
        column in female_features_table.columns
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
# FEMALE THRESHOLD
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


male_evidence = (
    calculate_male_metrics(
        male_coach
    )
)


# ============================================================
# SESSION
# ============================================================

if "section" not in st.session_state:

    st.session_state[
        "section"
    ] = "Varonil"


# ============================================================
# SIDEBAR
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

if st.session_state[
    "section"
] == "Varonil":

    opponents = (
        get_historical_opponents(
            male_match_history
        )
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
            else "AWAY"
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

        score_left, score_right = (
            st.columns(
                2,
                gap="small",
            )
        )

        with score_left:

            america_goals = (
                score_stepper(
                    label="América",
                    key="male_goals",
                )
            )

        with score_right:

            opponent_goals = (
                score_stepper(
                    label=opponent,
                    key="male_opponent_goals",
                )
            )

        visible_state = st.selectbox(
            "Estado táctico actual",
            list(
                STATE_DISPLAY.values()
            ),
            key="male_state",
        )

        current_state = (
            STATE_REVERSE[
                visible_state
            ]
        )


# ============================================================
# SIDEBAR FEMENIL
# ============================================================

elif st.session_state[
    "section"
] == "Femenil":

    opponents = (
        get_historical_opponents(
            female_match_history
        )
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
            else "AWAY"
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

        score_left, score_right = (
            st.columns(
                2,
                gap="small",
            )
        )

        with score_left:

            america_goals = (
                score_stepper(
                    label="América W",
                    key="female_goals",
                )
            )

        with score_right:

            opponent_goals = (
                score_stepper(
                    label=opponent,
                    key="female_opponent_goals",
                )
            )

        visible_state = st.selectbox(
            "Estado táctico actual",
            list(
                STATE_DISPLAY.values()
            ),
            key="female_state",
        )

        current_state = (
            STATE_REVERSE[
                visible_state
            ]
        )


# ============================================================
# HERO - CORREGIDO
# ============================================================

st.html(
    """
    <div class="coachlens-hero">

        <div class="coachlens-hero-kicker">
            INTELIGENCIA TÁCTICA · CLUB AMÉRICA
        </div>

        <div class="coachlens-hero-title">
            CoachLens
        </div>

        <div class="coachlens-hero-subtitle">
            Anticipa cambios tácticos ·
            Entiende el contexto ·
            Apoya decisiones
        </div>

        <div class="coachlens-hero-accent"></div>

    </div>
    """
)


# ============================================================
# NAV
# ============================================================

nav_options = [
    "Varonil",
    "Femenil",
    "Revisión histórica",
    "Análisis complementario",
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

if st.session_state[
    "section"
] == "Varonil":

    st.html(
        """
        <div class="page-kicker">
            MODELO VARONIL
        </div>

        <div class="page-title">
            Pronóstico táctico a 5 minutos
        </div>

        <div class="page-description">
            Estima el estado táctico más probable de América
            durante los próximos cinco minutos a partir del
            contexto actual del partido.
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
            "No están disponibles todos los archivos "
            "del modelo Varonil."
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

        historical_matches = (
            count_matches(
                male_match_history,
                opponent,
            )
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


            probabilities = (
                probabilities[
                    0
                ]
            )


            ranking = np.argsort(
                probabilities
            )[::-1]


            first_index = int(
                ranking[0]
            )

            second_index = int(
                ranking[1]
            )


            predicted_state = (
                STATE_LEVELS[
                    first_index
                ]
            )

            second_state = (
                STATE_LEVELS[
                    second_index
                ]
            )


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


            action = (
                get_coach_action(
                    predicted_state
                )
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

                        <div class="result-card-value">
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

                        <div class="result-card-value">
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
                    <div class="result-card">

                        <div class="result-card-label">
                            Acción asociada
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
                            Probabilidad de la predicción principal
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

                            Diferencia entre opciones:

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
                            Comparación frente a escenarios
                            históricos disponibles.
                        </div>

                        <div class="similarity-history">
                            Partidos ante
                            {opponent}: {historical_matches}
                        </div>

                    </div>
                    """
                )


            # =================================================
            # SIMULACIÓN
            # =================================================

            st.write("")
            st.write("")

            st.markdown(
                "### Simulación táctica de escenarios"
            )

            st.caption(
                "Se repite 1,000 veces el mismo contexto "
                "para representar la incertidumbre sobre "
                "los próximos cinco minutos."
            )


            male_simulation = (
                simulate_male_scenarios(
                    probabilities,
                    STATE_LEVELS,
                    1000,
                    42,
                )
            )


            dominant_row = (
                male_simulation
                .sort_values(
                    "share",
                    ascending=False,
                )
                .iloc[0]
            )


            st.html(
                f"""
                <div class="simulation-intro-box">

                    <div class="simulation-intro-title">
                        ¿Cómo leer esta simulación?
                    </div>

                    <div class="simulation-intro-text">

                        De
                        <strong>1,000 escenarios</strong>,

                        <strong>
                            {int(dominant_row["simulations"])}
                        </strong>

                        terminaron en

                        <strong class="simulation-highlight">
                            {state_display(dominant_row["state"])}
                        </strong>.

                        Eso representa

                        <strong>
                            {safe_percent(dominant_row["share"])}
                        </strong>.

                    </div>

                    <div class="simulation-intro-note">
                        No son 1,000 partidos completos.
                        Son posibles desenlaces tácticos
                        para este mismo contexto.
                    </div>

                </div>
                """
            )


            sim_left, sim_right = st.columns(
                [
                    1.7,
                    1,
                ],
                gap="large",
            )


            with sim_left:

                simulation_plot = (
                    male_simulation
                    .sort_values(
                        "percentage",
                        ascending=True,
                    )
                )


                figure_sim = px.bar(
                    simulation_plot,
                    x="percentage",
                    y="label",
                    orientation="h",
                    text="text_label",
                )


                figure_sim.update_traces(
                    marker_color=
                        AMERICA_YELLOW,

                    textposition=
                        "outside",

                    textfont=dict(
                        color="#061E39",
                        size=14,
                    ),

                    cliponaxis=False,
                )


                maximum = float(
                    simulation_plot[
                        "percentage"
                    ].max()
                )


                figure_sim.update_layout(
                    height=330,

                    paper_bgcolor=
                        "rgba(0,0,0,0)",

                    plot_bgcolor=
                        "rgba(0,0,0,0)",

                    margin=dict(
                        l=20,
                        r=90,
                        t=5,
                        b=45,
                    ),

                    xaxis_title=
                        "Porcentaje de escenarios",

                    yaxis_title="",

                    xaxis=dict(
                        range=[
                            0,
                            max(
                                40,
                                maximum + 8,
                            ),
                        ],

                        ticksuffix="%",

                        showgrid=True,

                        gridcolor=
                            "rgba(6,30,57,0.12)",

                        zeroline=False,

                        tickfont=dict(
                            color="#40536C",
                            size=13,
                        ),

                        title_font=dict(
                            color="#40536C",
                            size=14,
                        ),
                    ),

                    yaxis=dict(
                        showgrid=False,

                        tickfont=dict(
                            color="#061E39",
                            size=14,
                        ),
                    ),

                    font=dict(
                        color="#061E39",
                    ),
                )


                st.plotly_chart(
                    figure_sim,
                    use_container_width=True,
                    config={
                        "displayModeBar":
                            False
                    },
                )


            with sim_right:

                st.html(
                    f"""
                    <div class="simulation-summary-card">

                        <div class="simulation-summary-label">
                            ESCENARIO MÁS FRECUENTE
                        </div>

                        <div class="simulation-summary-value">
                            {state_display(dominant_row["state"])}
                        </div>

                        <div class="simulation-summary-share">
                            {safe_percent(dominant_row["share"])}
                        </div>

                        <div class="simulation-summary-text">
                            Estado que apareció con mayor
                            frecuencia entre los escenarios
                            simulados.
                        </div>

                    </div>
                    """
                )


        except Exception as error:

            st.error(
                f"No se pudo generar el análisis: {error}"
            )

# ============================================================
# ANÁLISIS COMPLEMENTARIO
# ============================================================

elif st.session_state[
    "section"
] == "Análisis complementario":

    st.html(
        """
        <div class="page-kicker">
            COMPONENTES COMPLEMENTARIOS
        </div>

        <div class="page-title">
            Jugadores y balón parado
        </div>

        <div class="page-description">
            Analiza sustituciones y jugadas a balón parado
            a partir de eventos reales de StatsBomb para
            complementar la lectura táctica de CoachLens.
        </div>
        """
    )


    # ========================================================
    # SELECTOR EQUIPO
    # ========================================================

    complementary_team = st.radio(
        "Equipo",
        [
            "Varonil",
            "Femenil",
        ],
        horizontal=True,
        key="complementary_team",
    )


    st.write("")


    # ========================================================
    # SUBSECCIONES
    # ========================================================

    complementary_section = st.radio(
        "Análisis",
        [
            "Jugadores y sustituciones",
            "Balón parado ofensivo",
            "Balón parado defensivo",
        ],
        horizontal=True,
        key="complementary_section",
    )


    # ========================================================
    # JUGADORES Y SUSTITUCIONES
    # ========================================================

    if complementary_section == "Jugadores y sustituciones":

        st.markdown(
            "### Uso de jugadores y sustituciones"
        )

        st.caption(
            "Se compara el comportamiento colectivo de los "
            "10 minutos anteriores y posteriores a cada "
            "sustitución registrada."
        )


        if complementary_substitutions.empty:

            st.warning(
                "No se encontraron datos de sustituciones."
            )

        else:

            subs = (
                complementary_substitutions[
                    complementary_substitutions[
                        "squad"
                    ]
                    ==
                    complementary_team
                ]
                .copy()
            )


            if subs.empty:

                st.warning(
                    "No existen sustituciones para el equipo seleccionado."
                )

            else:

                subs[
                    "minute"
                ] = pd.to_numeric(
                    subs[
                        "minute"
                    ],
                    errors="coerce",
                )


                subs = (
                    subs
                    .dropna(
                        subset=[
                            "minute",
                        ]
                    )
                    .sort_values(
                        [
                            "match_id",
                            "minute",
                        ],
                        ascending=[
                            False,
                            True,
                        ],
                    )
                )


                subs[
                    "selector"
                ] = (
                    "Partido "
                    +
                    subs[
                        "match_id"
                    ]
                    .astype(str)
                    +
                    " · Min "
                    +
                    subs[
                        "minute"
                    ]
                    .astype(int)
                    .astype(str)
                    +
                    " · "
                    +
                    subs[
                        "player_out"
                    ]
                    .fillna(
                        "N/D"
                    )
                    .astype(str)
                    +
                    " → "
                    +
                    subs[
                        "player_in"
                    ]
                    .fillna(
                        "N/D"
                    )
                    .astype(str)
                )


                selected_substitution = st.selectbox(
                    "Sustitución",
                    subs[
                        "selector"
                    ]
                    .tolist(),
                    key="selected_substitution",
                )


                selected_row = (
                    subs[
                        subs[
                            "selector"
                        ]
                        ==
                        selected_substitution
                    ]
                    .iloc[0]
                )


                substitution_minute = int(
                    selected_row[
                        "minute"
                    ]
                )


                player_out = str(
                    selected_row[
                        "player_out"
                    ]
                )


                player_in = str(
                    selected_row[
                        "player_in"
                    ]
                )


                # =================================================
                # CAMBIO
                # =================================================

                st.html(
                    f"""
                    <div class="substitution-hero">

                        <div class="substitution-minute">
                            MIN {substitution_minute}
                        </div>

                        <div class="substitution-player-grid">

                            <div class="substitution-player">

                                <div class="substitution-label">
                                    SALE
                                </div>

                                <div class="substitution-name">
                                    {player_out}
                                </div>

                            </div>


                            <div class="substitution-arrow">
                                →
                            </div>


                            <div class="substitution-player">

                                <div class="substitution-label">
                                    ENTRA
                                </div>

                                <div class="substitution-name">
                                    {player_in}
                                </div>

                            </div>

                        </div>

                    </div>
                    """
                )


                st.write("")


                # =================================================
                # MÉTRICAS ANTES / DESPUÉS
                # =================================================

                metric_definitions = [
                    (
                        "Tiros",
                        "shots_before",
                        "shots_after",
                        "shots_change",
                    ),
                    (
                        "Presiones",
                        "pressures_before",
                        "pressures_after",
                        "pressures_change",
                    ),
                    (
                        "Recuperaciones",
                        "recoveries_before",
                        "recoveries_after",
                        "recoveries_change",
                    ),
                    (
                        "Pases",
                        "passes_before",
                        "passes_after",
                        "passes_change",
                    ),
                ]


                metric_columns = st.columns(
                    4,
                    gap="medium",
                )


                changes_for_reading = {}


                for (
                    column,
                    (
                        label,
                        before_column,
                        after_column,
                        change_column,
                    ),
                ) in zip(
                    metric_columns,
                    metric_definitions,
                ):

                    before = int(
                        pd.to_numeric(
                            selected_row[
                                before_column
                            ],
                            errors="coerce",
                        )
                        if pd.notna(
                            selected_row[
                                before_column
                            ]
                        )
                        else 0
                    )


                    after = int(
                        pd.to_numeric(
                            selected_row[
                                after_column
                            ],
                            errors="coerce",
                        )
                        if pd.notna(
                            selected_row[
                                after_column
                            ]
                        )
                        else 0
                    )


                    change = after - before


                    changes_for_reading[
                        label
                    ] = change


                    if change > 0:

                        change_text = (
                            f"+{change}"
                        )

                        change_class = (
                            "positive"
                        )

                    elif change < 0:

                        change_text = str(
                            change
                        )

                        change_class = (
                            "negative"
                        )

                    else:

                        change_text = "0"

                        change_class = (
                            "neutral"
                        )


                    with column:

                        st.html(
                            f"""
                            <div class="before-after-card">

                                <div class="before-after-title">
                                    {label}
                                </div>

                                <div class="before-after-values">

                                    <div>

                                        <span>
                                            ANTES
                                        </span>

                                        <strong>
                                            {before}
                                        </strong>

                                    </div>

                                    <div class="before-after-arrow">
                                        →
                                    </div>

                                    <div>

                                        <span>
                                            DESPUÉS
                                        </span>

                                        <strong>
                                            {after}
                                        </strong>

                                    </div>

                                </div>

                                <div class="before-after-change {change_class}">
                                    Cambio: {change_text}
                                </div>

                            </div>
                            """
                        )


                # =================================================
                # LECTURA AUTOMÁTICA
                # =================================================

                ordered_changes = sorted(
                    changes_for_reading.items(),
                    key=lambda item:
                        abs(
                            item[1]
                        ),
                    reverse=True,
                )


                main_metric = (
                    ordered_changes[
                        0
                    ][0]
                )

                main_change = (
                    ordered_changes[
                        0
                    ][1]
                )


                if main_change > 0:

                    direction_text = (
                        "aumentó"
                    )

                elif main_change < 0:

                    direction_text = (
                        "disminuyó"
                    )

                else:

                    direction_text = (
                        "se mantuvo estable"
                    )


                st.html(
                    f"""
                    <div class="complementary-reading">

                        <div class="complementary-reading-title">
                            Lectura del cambio
                        </div>

                        <div class="complementary-reading-text">

                            El mayor cambio observado alrededor
                            de esta sustitución aparece en

                            <strong>
                                {main_metric.lower()}
                            </strong>,

                            donde la actividad

                            <strong>
                                {direction_text}
                            </strong>

                            durante los diez minutos posteriores.

                        </div>

                        <div class="complementary-reading-note">
                            La comparación es descriptiva.
                            No demuestra que la sustitución
                            haya causado el cambio observado.
                        </div>

                    </div>
                    """
                )


                # =================================================
                # GRÁFICA
                # =================================================

                comparison_data = pd.DataFrame(
                    {
                        "Métrica": [
                            "Tiros",
                            "Presiones",
                            "Recuperaciones",
                            "Pases",
                        ],

                        "Antes": [
                            selected_row[
                                "shots_before"
                            ],
                            selected_row[
                                "pressures_before"
                            ],
                            selected_row[
                                "recoveries_before"
                            ],
                            selected_row[
                                "passes_before"
                            ],
                        ],

                        "Después": [
                            selected_row[
                                "shots_after"
                            ],
                            selected_row[
                                "pressures_after"
                            ],
                            selected_row[
                                "recoveries_after"
                            ],
                            selected_row[
                                "passes_after"
                            ],
                        ],
                    }
                )


                comparison_long = (
                    comparison_data
                    .melt(
                        id_vars="Métrica",
                        var_name="Ventana",
                        value_name="Eventos",
                    )
                )


                comparison_long[
                    "Eventos"
                ] = pd.to_numeric(
                    comparison_long[
                        "Eventos"
                    ],
                    errors="coerce",
                ).fillna(
                    0
                )


                substitution_figure = px.bar(
                    comparison_long,
                    x="Métrica",
                    y="Eventos",
                    color="Ventana",
                    barmode="group",
                    text_auto=True,
                )


                substitution_figure.update_layout(
                    height=350,

                    paper_bgcolor=
                        "rgba(0,0,0,0)",

                    plot_bgcolor=
                        "rgba(0,0,0,0)",

                    margin=dict(
                        l=20,
                        r=20,
                        t=20,
                        b=30,
                    ),

                    legend_title_text="",

                    xaxis_title="",

                    yaxis_title=
                        "Número de eventos",

                    font=dict(
                        color="#061E39",
                    ),
                )


                st.plotly_chart(
                    substitution_figure,
                    use_container_width=True,
                    config={
                        "displayModeBar":
                            False
                    },
                )


    # ========================================================
    # BALÓN PARADO
    # ========================================================

    else:

        selected_phase = (
            "OFENSIVO"
            if complementary_section
            ==
            "Balón parado ofensivo"
            else
            "DEFENSIVO"
        )


        if selected_phase == "OFENSIVO":

            st.markdown(
                "### Balón parado ofensivo"
            )

            st.caption(
                "Analiza qué ocurre después de córners "
                "y tiros libres ejecutados por América."
            )

        else:

            st.markdown(
                "### Balón parado defensivo"
            )

            st.caption(
                "Analiza los córners y tiros libres del rival "
                "y las oportunidades concedidas por América."
            )


        if complementary_set_piece_summary.empty:

            st.warning(
                "No existen datos de balón parado."
            )

        else:

            summary = (
                complementary_set_piece_summary[
                    (
                        complementary_set_piece_summary[
                            "squad"
                        ]
                        ==
                        complementary_team
                    )
                    &
                    (
                        complementary_set_piece_summary[
                            "phase"
                        ]
                        ==
                        selected_phase
                    )
                ]
                .copy()
            )


            if summary.empty:

                st.warning(
                    "No existen registros para esta selección."
                )

            else:

                for numeric_column in [
                    "executions",
                    "shots",
                    "goals",
                    "shot_rate",
                    "goal_rate",
                ]:

                    if (
                        numeric_column
                        in
                        summary.columns
                    ):

                        summary[
                            numeric_column
                        ] = pd.to_numeric(
                            summary[
                                numeric_column
                            ],
                            errors="coerce",
                        ).fillna(
                            0
                        )


                total_executions = int(
                    summary[
                        "executions"
                    ].sum()
                )


                total_shots = int(
                    summary[
                        "shots"
                    ].sum()
                )


                total_goals = int(
                    summary[
                        "goals"
                    ].sum()
                )


                overall_shot_rate = (
                    total_shots
                    /
                    total_executions
                    if
                    total_executions > 0
                    else
                    0
                )


                overall_goal_rate = (
                    total_goals
                    /
                    total_executions
                    if
                    total_executions > 0
                    else
                    0
                )


                # =================================================
                # KPI CARDS
                # =================================================

                sp1, sp2, sp3, sp4 = (
                    st.columns(
                        4,
                        gap="medium",
                    )
                )


                set_piece_cards = [
                    (
                        sp1,
                        "Ejecuciones",
                        f"{total_executions:,}",
                    ),
                    (
                        sp2,
                        (
                            "Tiros generados"
                            if
                            selected_phase
                            ==
                            "OFENSIVO"
                            else
                            "Tiros concedidos"
                        ),
                        f"{total_shots:,}",
                    ),
                    (
                        sp3,
                        (
                            "Goles generados"
                            if
                            selected_phase
                            ==
                            "OFENSIVO"
                            else
                            "Goles concedidos"
                        ),
                        f"{total_goals:,}",
                    ),
                    (
                        sp4,
                        (
                            "% termina en tiro"
                        ),
                        safe_percent(
                            overall_shot_rate
                        ),
                    ),
                ]


                for (
                    column,
                    label,
                    value,
                ) in set_piece_cards:

                    with column:

                        st.html(
                            f"""
                            <div class="metric-card complementary-metric-card">

                                <div class="metric-label">
                                    {label}
                                </div>

                                <div class="metric-value">
                                    {value}
                                </div>

                            </div>
                            """
                        )


                st.write("")


                # =================================================
                # DESGLOSE CORNER / TIRO LIBRE
                # =================================================

                st.markdown(
                    "#### Desglose por tipo de jugada"
                )


                plot_data = summary.copy()


                plot_data[
                    "Tipo"
                ] = (
                    plot_data[
                        "set_piece"
                    ]
                    .replace(
                        {
                            "CÓRNER":
                                "Córner",

                            "TIRO LIBRE":
                                "Tiro libre",
                        }
                    )
                )


                plot_data[
                    "Tasa de tiro"
                ] = (
                    plot_data[
                        "shot_rate"
                    ]
                    *
                    100
                )


                plot_data[
                    "Tasa de gol"
                ] = (
                    plot_data[
                        "goal_rate"
                    ]
                    *
                    100
                )


                # =================================================
                # TABLA VISUAL
                # =================================================

                breakdown_columns = st.columns(
                    len(
                        plot_data
                    ),
                    gap="medium",
                )


                for (
                    column,
                    (_, row),
                ) in zip(
                    breakdown_columns,
                    plot_data.iterrows(),
                ):

                    with column:

                        st.html(
                            f"""
                            <div class="set-piece-type-card">

                                <div class="set-piece-type">
                                    {row["Tipo"]}
                                </div>

                                <div class="set-piece-stat-grid">

                                    <div>
                                        <span>EJECUCIONES</span>
                                        <strong>
                                            {int(row["executions"])}
                                        </strong>
                                    </div>

                                    <div>
                                        <span>
                                            {
                                                "TIROS"
                                                if selected_phase == "OFENSIVO"
                                                else "TIROS RIVAL"
                                            }
                                        </span>

                                        <strong>
                                            {int(row["shots"])}
                                        </strong>
                                    </div>

                                    <div>
                                        <span>
                                            {
                                                "GOLES"
                                                if selected_phase == "OFENSIVO"
                                                else "GOLES RIVAL"
                                            }
                                        </span>

                                        <strong>
                                            {int(row["goals"])}
                                        </strong>
                                    </div>

                                    <div>
                                        <span>% A TIRO</span>

                                        <strong>
                                            {safe_percent(row["shot_rate"])}
                                        </strong>
                                    </div>

                                </div>

                            </div>
                            """
                        )


                st.write("")


                # =================================================
                # GRÁFICA
                # =================================================

                rate_figure = px.bar(
                    plot_data,
                    x="Tipo",
                    y="Tasa de tiro",
                    text="Tasa de tiro",
                )


                rate_figure.update_traces(
                    marker_color=
                        AMERICA_YELLOW,

                    texttemplate=
                        "%{text:.1f}%",

                    textposition=
                        "outside",

                    cliponaxis=False,

                    textfont=dict(
                        color="#061E39",
                        size=14,
                    ),
                )


                max_rate = float(
                    plot_data[
                        "Tasa de tiro"
                    ].max()
                )


                rate_figure.update_layout(
                    height=330,

                    paper_bgcolor=
                        "rgba(0,0,0,0)",

                    plot_bgcolor=
                        "rgba(0,0,0,0)",

                    margin=dict(
                        l=20,
                        r=20,
                        t=20,
                        b=30,
                    ),

                    xaxis_title="",

                    yaxis_title=
                        "Jugadas que terminan en tiro",

                    yaxis=dict(
                        ticksuffix="%",

                        range=[
                            0,
                            max(
                                100,
                                max_rate + 10,
                            ),
                        ],

                        gridcolor=
                            "rgba(6,30,57,0.10)",

                        tickfont=dict(
                            color="#40536C",
                        ),
                    ),

                    xaxis=dict(
                        tickfont=dict(
                            color="#061E39",
                            size=14,
                        ),
                    ),

                    font=dict(
                        color="#061E39",
                    ),
                )


                st.plotly_chart(
                    rate_figure,
                    use_container_width=True,
                    config={
                        "displayModeBar":
                            False
                    },
                )


                # =================================================
                # INTERPRETACIÓN
                # =================================================

                best_row = (
                    plot_data
                    .sort_values(
                        "shot_rate",
                        ascending=False,
                    )
                    .iloc[0]
                )


                best_type = (
                    best_row[
                        "Tipo"
                    ]
                )


                best_rate = (
                    best_row[
                        "shot_rate"
                    ]
                )


                if selected_phase == "OFENSIVO":

                    interpretation = (
                        f"{best_type} presenta la mayor tasa "
                        f"de generación de tiro: "
                        f"{safe_percent(best_rate)} de las "
                        f"ejecuciones identificadas terminaron "
                        f"en un tiro de América."
                    )

                else:

                    interpretation = (
                        f"{best_type} representa la situación "
                        f"de mayor vulnerabilidad relativa: "
                        f"{safe_percent(best_rate)} de las "
                        f"ejecuciones rivales identificadas "
                        f"terminaron en tiro."
                    )


                st.html(
                    f"""
                    <div class="complementary-reading">

                        <div class="complementary-reading-title">
                            Lectura táctica
                        </div>

                        <div class="complementary-reading-text">
                            {interpretation}
                        </div>

                        <div class="complementary-reading-note">
                            Las tasas describen lo ocurrido
                            en la muestra histórica analizada.
                            No representan una probabilidad
                            causal futura.
                        </div>

                    </div>
                    """
                )
# ============================================================
# FEMENIL
# ============================================================

elif st.session_state[
    "section"
] == "Femenil":

    st.html(
        """
        <div class="page-kicker">
            MODELO FEMENIL
        </div>

        <div class="page-title">
            Riesgo de cambio táctico
        </div>

        <div class="page-description">
            Estima la probabilidad de que América Femenil
            cambie de estado táctico durante los próximos
            cinco minutos.
        </div>
        """
    )


    if (
        female_model is None
        or
        female_simulator.empty
        or
        not female_feature_names
        or
        not np.isfinite(
            female_threshold
        )
    ):

        st.error(
            "No están disponibles todos los archivos "
            "del modelo Femenil."
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


        historical_matches = (
            count_matches(
                female_match_history,
                opponent,
            )
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
            ]


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


            signal_title = (
                "CAMBIO PROBABLE"
                if predicted_change
                else "ESTABILIDAD PROBABLE"
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


            if transition_state:

                transition_title = (
                    state_display(
                        transition_state
                    )
                )

                action = (
                    get_coach_action(
                        transition_state
                    )
                )

            else:

                transition_title = (
                    "SIN PATRÓN"
                )

                action = {
                    "title":
                        "Monitorear el escenario",

                    "text":
                        "No existe una transición histórica "
                        "dominante en contextos similares.",
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
                            Próximos cinco minutos
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

                        <div class="result-card-value">
                            {transition_title}
                        </div>

                        <div class="result-card-text">

                            Participación:
                            {safe_percent(transition_share)}

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
                    <div class="result-card">

                        <div class="result-card-label">
                            Acción asociada
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
                            Señal del modelo
                        </div>

                        <div class="confidence-number signal-number">
                            {signal_title}
                        </div>

                        <div class="confidence-details">

                            Probabilidad:
                            <strong>
                                {safe_percent(probability_change)}
                            </strong>

                            <br>

                            Umbral:
                            <strong>
                                {safe_percent(female_threshold)}
                            </strong>

                            <br>

                            Diferencia:
                            <strong>
                                {percentage_points(
                                    probability_change
                                    -
                                    female_threshold
                                )}
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
                            Comparación frente a escenarios
                            históricos similares.
                        </div>

                        <div class="similarity-history">
                            Partidos ante
                            {opponent}: {historical_matches}
                        </div>

                    </div>
                    """
                )


            st.write("")
            st.write("")


            st.markdown(
                "### Simulación táctica de escenarios"
            )


            (
                female_simulation,
                female_transition_simulation,
            ) = simulate_female_scenarios(
                probability_change,
                historical_transition[
                    "distribution"
                ],
                1000,
                42,
            )


            dominant = (
                female_simulation
                .sort_values(
                    "share",
                    ascending=False,
                )
                .iloc[0]
            )


            st.html(
                f"""
                <div class="simulation-intro-box">

                    <div class="simulation-intro-title">
                        ¿Cómo leer esta simulación?
                    </div>

                    <div class="simulation-intro-text">

                        De
                        <strong>1,000 escenarios</strong>,

                        <strong>
                            {int(dominant["simulations"])}
                        </strong>

                        terminaron en

                        <strong class="simulation-highlight">
                            {dominant["outcome"]}
                        </strong>.

                        Esto representa

                        <strong>
                            {safe_percent(dominant["share"])}
                        </strong>.

                    </div>

                </div>
                """
            )


            sim_left, sim_right = st.columns(
                [
                    1.7,
                    1,
                ],
                gap="large",
            )


            with sim_left:

                female_plot = (
                    female_simulation
                    .sort_values(
                        "percentage",
                        ascending=True,
                    )
                )


                figure = px.bar(
                    female_plot,
                    x="percentage",
                    y="outcome",
                    orientation="h",
                    text="text_label",
                )


                figure.update_traces(
                    marker_color=
                        AMERICA_YELLOW,

                    textposition=
                        "outside",

                    textfont=dict(
                        color="#061E39",
                        size=14,
                    ),

                    cliponaxis=False,
                )


                figure.update_layout(
                    height=260,

                    paper_bgcolor=
                        "rgba(0,0,0,0)",

                    plot_bgcolor=
                        "rgba(0,0,0,0)",

                    margin=dict(
                        l=20,
                        r=90,
                        t=10,
                        b=45,
                    ),

                    xaxis_title=
                        "Porcentaje de escenarios",

                    yaxis_title="",

                    xaxis=dict(
                        range=[
                            0,
                            100,
                        ],

                        ticksuffix="%",

                        gridcolor=
                            "rgba(6,30,57,0.12)",

                        tickfont=dict(
                            color="#40536C",
                        ),

                        title_font=dict(
                            color="#40536C",
                        ),
                    ),

                    yaxis=dict(
                        tickfont=dict(
                            color="#061E39",
                            size=14,
                        ),
                    ),

                    font=dict(
                        color="#061E39",
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


            with sim_right:

                st.html(
                    f"""
                    <div class="simulation-summary-card">

                        <div class="simulation-summary-label">
                            RESULTADO MÁS FRECUENTE
                        </div>

                        <div class="simulation-summary-value">
                            {dominant["outcome"]}
                        </div>

                        <div class="simulation-summary-share">
                            {safe_percent(dominant["share"])}
                        </div>

                    </div>
                    """
                )


            if not female_transition_simulation.empty:

                st.markdown(
                    "#### Si ocurre un cambio, ¿hacia qué estado?"
                )


                transition_plot = (
                    female_transition_simulation
                    .sort_values(
                        "percentage",
                        ascending=True,
                    )
                )


                transition_figure = px.bar(
                    transition_plot,
                    x="percentage",
                    y="label",
                    orientation="h",
                    text="text_label",
                )


                transition_figure.update_traces(
                    marker_color=
                        AMERICA_YELLOW,

                    textposition=
                        "outside",

                    textfont=dict(
                        color="#061E39",
                        size=14,
                    ),

                    cliponaxis=False,
                )


                maximum = float(
                    transition_plot[
                        "percentage"
                    ].max()
                )


                transition_figure.update_layout(
                    height=290,

                    paper_bgcolor=
                        "rgba(0,0,0,0)",

                    plot_bgcolor=
                        "rgba(0,0,0,0)",

                    margin=dict(
                        l=20,
                        r=90,
                        t=10,
                        b=45,
                    ),

                    xaxis_title=
                        "Porcentaje de cambios simulados",

                    yaxis_title="",

                    xaxis=dict(
                        range=[
                            0,
                            max(
                                40,
                                maximum + 7,
                            ),
                        ],

                        ticksuffix="%",

                        gridcolor=
                            "rgba(6,30,57,0.12)",

                        tickfont=dict(
                            color="#40536C",
                        ),

                        title_font=dict(
                            color="#40536C",
                        ),
                    ),

                    yaxis=dict(
                        tickfont=dict(
                            color="#061E39",
                            size=14,
                        ),
                    ),

                    font=dict(
                        color="#061E39",
                    ),
                )


                st.plotly_chart(
                    transition_figure,
                    use_container_width=True,
                    config={
                        "displayModeBar":
                            False
                    },
                )


        except Exception as error:

            st.error(
                f"No se pudo generar el análisis: {error}"
            )


# ============================================================
# REVIEW
# ============================================================

elif st.session_state[
    "section"
] == "Revisión histórica":

    st.html(
        """
        <div class="page-kicker">
            VALIDACIÓN HISTÓRICA
        </div>

        <div class="page-title">
            Revisión de partidos TEST
        </div>

        <div class="page-description">
            Explora predicciones realizadas sobre partidos
            reservados para evaluación y no utilizados
            durante el entrenamiento.
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


    if review_mode == "Varonil":

        history = (
            male_coach.copy()
        )

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
        ] = (
            history[
                "match_id"
            ]
            .map(
                normalize_match_id
            )
        )


        played_ids = (
            get_played_match_ids(
                male_match_history
            )
        )


        if played_ids:

            history = history[
                history[
                    "_match_id"
                ].isin(
                    played_ids
                )
            ]


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
        ]


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
        )


        selected_minute = st.select_slider(
            "Minuto",
            options=minutes,
            value=minutes[
                len(minutes)
                //
                2
            ],
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


        cols = st.columns(
            4,
            gap="medium",
        )


        values = [
            (
                "Estado actual",
                state_display(
                    current
                ),
            ),

            (
                "Primera predicción",
                state_display(
                    predicted
                ),
            ),

            (
                "Estado real +5 min",
                state_display(
                    actual
                ),
            ),

            (
                "Resultado",
                result,
            ),
        ]


        for col, (
            label,
            value,
        ) in zip(
            cols,
            values,
        ):

            with col:

                st.html(
                    f"""
                    <div class="metric-card history-metric-card">

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
                            Coincide con la primera opción
                            del modelo.
                        </div>

                    </div>

                    <div class="history-explanation-item">

                        <div class="history-explanation-badge top2">
                            ACIERTO TOP-2
                        </div>

                        <div class="history-explanation-text">
                            Coincide con la segunda
                            alternativa del modelo.
                        </div>

                    </div>

                    <div class="history-explanation-item">

                        <div class="history-explanation-badge miss">
                            NO ACIERTO
                        </div>

                        <div class="history-explanation-text">
                            El estado real no apareció entre
                            las dos opciones principales.
                        </div>

                    </div>

                </div>

                <div class="history-test-summary">

                    Rendimiento TEST:

                    <strong>
                        Top-1:
                        {safe_percent(male_evidence["top1"])}
                    </strong>

                    ·

                    <strong>
                        Top-2:
                        {safe_percent(male_evidence["top2"])}
                    </strong>

                </div>

            </div>
            """
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
        ]


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
        )


        selected_minute = st.select_slider(
            "Minuto",
            options=minutes,
            value=minutes[
                len(minutes)
                //
                2
            ],
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


        predicted_change = (
            interpret_change_value(
                row[
                    "predicted_change"
                ]
            )
        )


        actual_change = (
            interpret_change_value(
                row[
                    "actual_change"
                ]
            )
        )


        result = (
            "ACIERTO"
            if
            predicted_change
            ==
            actual_change
            else
            "NO ACIERTO"
        )


        cols = st.columns(
            4,
            gap="medium",
        )


        values = [
            (
                "Estado actual",
                state_display(
                    current
                ),
            ),

            (
                "Probabilidad de cambio",
                safe_percent(
                    probability
                ),
            ),

            (
                "Cambio real +5 min",
                actual_change,
            ),

            (
                "Resultado",
                result,
            ),
        ]


        for col, (
            label,
            value,
        ) in zip(
            cols,
            values,
        ):

            with col:

                st.html(
                    f"""
                    <div class="metric-card history-metric-card">

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
                            cambio o estabilidad.
                        </div>

                    </div>

                    <div class="history-explanation-item">

                        <div class="history-explanation-badge miss">
                            NO ACIERTO
                        </div>

                        <div class="history-explanation-text">
                            La clasificación no coincidió
                            con lo observado.
                        </div>

                    </div>

                    <div class="history-explanation-item">

                        <div class="history-explanation-badge top2">
                            PROBABILIDAD
                        </div>

                        <div class="history-explanation-text">

                            Probabilidad estimada:

                            <strong>
                                {safe_percent(probability)}
                            </strong>.

                        </div>

                    </div>

                </div>

            </div>
            """
        )


# ============================================================
# MODELO
# ============================================================

elif st.session_state[
    "section"
] == "Modelo":

    st.html(
        """
        <div class="page-kicker">
            EVIDENCIA Y METODOLOGÍA
        </div>

        <div class="page-title">
            Rendimiento y marco analítico
        </div>

        <div class="page-description">
            Presenta el desempeño global, las variables
            relevantes y el framework de CoachLens.
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


    if model_mode == "Varonil":

        cols = st.columns(
            4,
            gap="medium",
        )


        metrics = [
            (
                "Exactitud Top-1",
                male_evidence[
                    "top1"
                ],
            ),

            (
                "Exactitud Top-2",
                male_evidence[
                    "top2"
                ],
            ),

            (
                "Baseline",
                male_evidence[
                    "baseline"
                ],
            ),

            (
                "Exactitud balanceada",
                male_evidence[
                    "balanced"
                ],
            ),
        ]


        for col, (
            label,
            value,
        ) in zip(
            cols,
            metrics,
        ):

            with col:

                st.html(
                    f"""
                    <div class="metric-card model-metric-card">

                        <div class="metric-label">
                            {label}
                        </div>

                        <div class="metric-value">
                            {safe_percent(value)}
                        </div>

                    </div>
                    """
                )


        st.caption(
            f"Evaluación sobre "
            f"{male_evidence['n']:,} "
            f"observaciones TEST."
        )


        st.html(
            """
            <div class="interpretation-box">

                <div class="interpretation-title">
                    ¿Qué significan estas métricas?
                </div>

                <div class="interpretation-columns">

                    <div>
                        <strong>Top-1</strong><br>
                        Coincidencia con la primera
                        predicción.
                    </div>

                    <div>
                        <strong>Top-2</strong><br>
                        El estado real está entre las
                        dos principales opciones.
                    </div>

                    <div>
                        <strong>Baseline</strong><br>
                        Compara contra mantener
                        el estado actual.
                    </div>

                    <div>
                        <strong>Balanceada</strong><br>
                        Promedia desempeño entre
                        las cinco clases.
                    </div>

                </div>

            </div>
            """
        )


        st.markdown(
            "### Influencia global · Modelo Varonil"
        )


        if not male_shap.empty:

            importance = (
                male_shap
                .sort_values(
                    "mean_abs_shap",
                    ascending=False,
                )
                .head(8)
                .copy()
            )


            importance[
                "label"
            ] = (
                importance[
                    "feature"
                ]
                .map(
                    friendly_feature_name
                )
            )


            importance = (
                importance
                .sort_values(
                    "mean_abs_shap"
                )
            )


            fig = px.bar(
                importance,
                x="mean_abs_shap",
                y="label",
                orientation="h",
            )


            fig.update_traces(
                marker_color=
                    AMERICA_YELLOW
            )


            fig.update_layout(
                height=350,

                paper_bgcolor=
                    "rgba(0,0,0,0)",

                plot_bgcolor=
                    "rgba(0,0,0,0)",

                xaxis_title=
                    "Influencia SHAP promedio",

                yaxis_title="",

                margin=dict(
                    l=20,
                    r=20,
                    t=10,
                    b=45,
                ),

                xaxis=dict(
                    tickfont=dict(
                        color="#40536C",
                        size=12,
                    ),

                    title_font=dict(
                        color="#40536C",
                        size=14,
                    ),
                ),

                yaxis=dict(
                    tickfont=dict(
                        color="#061E39",
                        size=13,
                    ),
                ),

                font=dict(
                    color="#061E39",
                ),
            )


            st.plotly_chart(
                fig,
                use_container_width=True,
                config={
                    "displayModeBar":
                        False
                },
            )


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

        f1 = metric_from_table(
            female_metrics,
            "f1_change",
        )


        cols = st.columns(
            4,
            gap="medium",
        )


        metrics = [
            (
                "Accuracy",
                accuracy,
            ),

            (
                "Exactitud balanceada",
                balanced,
            ),

            (
                "AUC",
                auc,
            ),

            (
                "F1 · cambio",
                f1,
            ),
        ]


        for col, (
            label,
            value,
        ) in zip(
            cols,
            metrics,
        ):

            with col:

                st.html(
                    f"""
                    <div class="metric-card model-metric-card">

                        <div class="metric-label">
                            {label}
                        </div>

                        <div class="metric-value">
                            {safe_percent(value)}
                        </div>

                    </div>
                    """
                )


        st.html(
            """
            <div class="interpretation-box">

                <div class="interpretation-title">
                    ¿Qué significan estas métricas?
                </div>

                <div class="interpretation-columns">

                    <div>
                        <strong>Accuracy</strong><br>
                        Porcentaje total correcto.
                    </div>

                    <div>
                        <strong>Balanceada</strong><br>
                        Evalúa por igual ambas clases.
                    </div>

                    <div>
                        <strong>AUC</strong><br>
                        Capacidad de separar escenarios.
                    </div>

                    <div>
                        <strong>F1</strong><br>
                        Combina precisión y sensibilidad.
                    </div>

                </div>

            </div>
            """
        )


        st.markdown(
            "### Influencia global · Modelo Femenil"
        )


        importance = (
            get_xgb_importance(
                female_model,
                8,
            )
        )


        if not importance.empty:

            fig = px.bar(
                importance,
                x="importance",
                y="label",
                orientation="h",
            )


            fig.update_traces(
                marker_color=
                    AMERICA_YELLOW
            )


            fig.update_layout(
                height=350,

                paper_bgcolor=
                    "rgba(0,0,0,0)",

                plot_bgcolor=
                    "rgba(0,0,0,0)",

                xaxis_title=
                    "Ganancia promedio en XGBoost",

                yaxis_title="",

                margin=dict(
                    l=20,
                    r=20,
                    t=10,
                    b=45,
                ),

                xaxis=dict(
                    tickfont=dict(
                        color="#40536C",
                        size=12,
                    ),

                    title_font=dict(
                        color="#40536C",
                        size=14,
                    ),
                ),

                yaxis=dict(
                    tickfont=dict(
                        color="#061E39",
                        size=13,
                    ),
                ),

                font=dict(
                    color="#061E39",
                ),
            )


            st.plotly_chart(
                fig,
                use_container_width=True,
                config={
                    "displayModeBar":
                        False
                },
            )


    # ========================================================
    # FRAMEWORK
    # ========================================================

    st.write("")
    st.write("")

    st.markdown(
        "### Framework analítico de CoachLens"
    )


    st.html(
        """
        <div class="framework-section-box">

            <div class="framework-section-title">
                1. Definición de objetos
            </div>

            <div class="object-grid">

                <div class="object-item">
                    <strong>Estado táctico</strong>
                    <span>
                        Resume el comportamiento colectivo
                        del equipo.
                    </span>
                </div>

                <div class="object-item">
                    <strong>Transición</strong>
                    <span>
                        Cambio entre el estado actual y el
                        observado cinco minutos después.
                    </span>
                </div>

                <div class="object-item">
                    <strong>Contexto similar</strong>
                    <span>
                        Situación histórica comparable en
                        minuto, marcador, rival y estado.
                    </span>
                </div>

                <div class="object-item">
                    <strong>Horizonte</strong>
                    <span>
                        Próximos cinco minutos.
                    </span>
                </div>

                <div class="object-item">
                    <strong>Cambio táctico</strong>
                    <span>
                        Modificación del estado colectivo.
                    </span>
                </div>

                <div class="object-item">
                    <strong>Simulación</strong>
                    <span>
                        Repetición probabilística del mismo
                        contexto.
                    </span>
                </div>

            </div>

        </div>
        """
    )


    st.html(
        """
        <div class="framework-section-box">

            <div class="framework-section-title">
                2. Supuestos explícitos
            </div>

            <div class="assumption-list">

                <div>
                    <strong>01</strong>
                    <span>
                        El comportamiento próximo puede
                        aproximarse con patrones históricos.
                    </span>
                </div>

                <div>
                    <strong>02</strong>
                    <span>
                        Los estados representan al colectivo,
                        no al jugador individual.
                    </span>
                </div>

                <div>
                    <strong>03</strong>
                    <span>
                        Similitud no implica causalidad.
                    </span>
                </div>

                <div>
                    <strong>04</strong>
                    <span>
                        Las acciones son apoyo a decisión,
                        no efectos causales garantizados.
                    </span>
                </div>

                <div>
                    <strong>05</strong>
                    <span>
                        Femenil predice cambio/estabilidad;
                        el estado posterior es histórico.
                    </span>
                </div>

                <div>
                    <strong>06</strong>
                    <span>
                        Las simulaciones no representan
                        partidos completos.
                    </span>
                </div>

            </div>

        </div>
        """
    )


    st.html(
    """
    <div class="framework-card">

        <div class="framework-title">
            Uso de jugadores
        </div>

        <div class="framework-text">
            Identifica jugadores sustituidos y compara
            el comportamiento colectivo en ventanas
            de diez minutos antes y después del cambio.
        </div>

    </div>


    <div class="framework-card">

        <div class="framework-title">
            Balón parado ofensivo
        </div>

        <div class="framework-text">
            Analiza córners y tiros libres,
            tiros generados, goles y tasa
            de llegada a remate.
        </div>

    </div>


    <div class="framework-card">

        <div class="framework-title">
            Balón parado defensivo
        </div>

        <div class="framework-text">
            Mide tiros y goles concedidos
            después de córners y tiros libres
            ejecutados por los rivales.
        </div>

    </div>
    """
)


    st.html(
        """
        <div class="framework-section-box">

            <div class="framework-section-title">
                4. Coherencia interna
            </div>

            <div class="coherence-flow">

                <div class="coherence-step">
                    <strong>Datos</strong>
                    <span>Events + 360</span>
                </div>

                <div class="coherence-arrow">→</div>

                <div class="coherence-step">
                    <strong>Variables</strong>
                    <span>Contexto + juego</span>
                </div>

                <div class="coherence-arrow">→</div>

                <div class="coherence-step">
                    <strong>Modelo</strong>
                    <span>XGBoost</span>
                </div>

                <div class="coherence-arrow">→</div>

                <div class="coherence-step">
                    <strong>Validación</strong>
                    <span>TEST</span>
                </div>

                <div class="coherence-arrow">→</div>

                <div class="coherence-step">
                    <strong>Interpretación</strong>
                    <span>Apoyo táctico</span>
                </div>

            </div>

        </div>
        """
    )


    st.html(
        """
        <div class="framework-section-box">

            <div class="framework-section-title">
                5. Alcance y límites
            </div>

            <div class="scope-box">

                <div class="scope-column">

                    <div class="scope-heading">
                        Incluido
                    </div>

                    <ul>
                        <li>Estados tácticos a +5 min.</li>
                        <li>Riesgo de cambio.</li>
                        <li>Presión y recuperación.</li>
                        <li>Progresión.</li>
                        <li>Contexto de partido.</li>
                        <li>Validación TEST.</li>
                        <li>Simulación táctica.</li>
                    </ul>

                </div>

                <div class="scope-column">

                    <div class="scope-heading">
                        Fuera del alcance actual
                    </div>

                    <ul>
                        <li>Marcador final.</li>
                        <li>Apuestas.</li>
                        <<li>Evaluación causal del impacto individual.</li>
                        <li>Efecto causal de alineaciones o sustituciones.</li>
                        <li>Modelado específico de marcaje individual.</li>
                        <li>Simulación de 90 minutos.</li>
                    </ul>

                </div>

            </div>

        </div>
        """
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "COACHLENS · CLUB AMÉRICA · "
    "XGBoost · StatsBomb Events + 360"
)