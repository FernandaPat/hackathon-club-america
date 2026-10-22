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

CSS_PATH = ROOT / "dashboard" / "styles" / "america.css"

SIMULATOR_DATA_PATH = (
    ROOT
    / "dashboard"
    / "data"
    / "simulator_context.csv"
)

MATCH_HISTORY_PATH = (
    ROOT
    / "dashboard"
    / "data"
    / "match_history.csv"
)

MODEL_PATH = (
    ROOT
    / "outputs"
    / "models"
    / "blend_xgboost_model.json"
)

COACH_ALL_PATH = (
    ROOT
    / "outputs"
    / "coach_engine"
    / "coach_engine_all_test.csv"
)

SHAP_GLOBAL_PATH = (
    ROOT
    / "outputs"
    / "models"
    / "interpretability"
    / "shap_global_importance.csv"
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
    visible: internal
    for internal, visible
    in STATE_DISPLAY.items()
}


STATE_FEATURES = [
    "current_state_control",
    "current_state_pressure",
    "current_state_transition",
    "current_state_protection",
    "current_state_urgency",
]


# ============================================================
# FEATURES REALES DEL XGBOOST
# ============================================================

MODEL_FEATURES = [
    "minute",
    "game_state_num",

    *STATE_FEATURES,

    "america_field_tilt_last5",
    "entries_final_third_diff_last5",
    "entries_box_diff_last5",
    "progressive_pass_diff_last5",
    "america_progressive_carries_last5",
    "america_progressive_passes_last5",
    "pressure_diff_last5",
    "recovery_height_diff_last5",

    "america_field_tilt_last3",
    "entries_final_third_diff_last3",
    "entries_box_diff_last3",
    "progressive_pass_diff_last3",
    "america_progressive_carries_last3",
    "america_progressive_passes_last3",
    "pressure_diff_last3",
    "recovery_height_diff_last3",

    "america_local_superiority_10_last5",
    "america_pressure_share_5_last5",
    "america_nearest_opponent_distance_last5",
    "america_support_share_10_last5",

    "america_local_superiority_10_last3",
    "america_pressure_share_5_last3",
    "america_nearest_opponent_distance_last3",
    "america_support_share_10_last3",
]


FEATURE_NAMES = {
    "minute": "Minuto del partido",
    "game_state_num": "Situación del marcador",

    "current_state_control": "Estado táctico actual",
    "current_state_pressure": "Estado táctico actual",
    "current_state_transition": "Estado táctico actual",
    "current_state_protection": "Estado táctico actual",
    "current_state_urgency": "Estado táctico actual",

    "america_field_tilt_last5":
        "Dominio territorial reciente",

    "entries_final_third_diff_last5":
        "Entradas al último tercio",

    "entries_box_diff_last5":
        "Entradas al área",

    "progressive_pass_diff_last5":
        "Pases progresivos",

    "america_progressive_carries_last5":
        "Conducciones progresivas",

    "america_progressive_passes_last5":
        "Pases progresivos de América",

    "pressure_diff_last5":
        "Balance de presión",

    "recovery_height_diff_last5":
        "Altura de recuperación",

    "america_field_tilt_last3":
        "Dominio territorial reciente",

    "entries_final_third_diff_last3":
        "Entradas recientes al último tercio",

    "entries_box_diff_last3":
        "Entradas recientes al área",

    "progressive_pass_diff_last3":
        "Progresión reciente",

    "america_progressive_carries_last3":
        "Conducciones progresivas recientes",

    "america_progressive_passes_last3":
        "Pases progresivos recientes",

    "pressure_diff_last3":
        "Presión reciente",

    "recovery_height_diff_last3":
        "Altura de recuperación reciente",

    "america_local_superiority_10_last5":
        "Superioridad numérica local",

    "america_pressure_share_5_last5":
        "Presión rival cercana",

    "america_nearest_opponent_distance_last5":
        "Distancia al rival más cercano",

    "america_support_share_10_last5":
        "Apoyo cercano",

    "america_local_superiority_10_last3":
        "Superioridad numérica reciente",

    "america_pressure_share_5_last3":
        "Presión rival reciente",

    "america_nearest_opponent_distance_last3":
        "Proximidad rival",

    "america_support_share_10_last3":
        "Apoyo reciente",
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

    model = xgb.Booster()

    model.load_model(
        str(path)
    )

    return model


# ============================================================
# HELPERS
# ============================================================

def normalize_text(value):

    value = str(
        value
    ).strip().lower()

    return "".join(
        character
        for character
        in unicodedata.normalize(
            "NFD",
            value,
        )
        if unicodedata.category(
            character
        )
        !=
        "Mn"
    )


def normalize_match_id(value):

    value = str(value)

    if value.endswith(".0"):
        value = value[:-2]

    return value


def state_display(state):

    return STATE_DISPLAY.get(
        str(state).upper(),
        str(state),
    )


def safe_percent(value):

    try:

        value = float(value)

        if not np.isfinite(value):
            return "N/D"

        if value <= 1:
            value *= 100

        return f"{value:.1f}%"

    except Exception:

        return "N/D"


def first_existing_column(
    data,
    candidates,
):

    for column in candidates:

        if column in data.columns:
            return column

    return None


# ============================================================
# PARTIDOS REALMENTE JUGADOS
# ============================================================

def played_matches_only(data):

    if data.empty:
        return data.copy()


    result = data.copy()


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


# ============================================================
# RIVALES
# ============================================================

def get_historical_opponents(
    matches
):

    data = played_matches_only(
        matches
    )


    if data.empty:
        return []


    if "opponent" in data.columns:

        return sorted(
            data["opponent"]
            .dropna()
            .astype(str)
            .str.strip()
            .unique()
            .tolist()
        )


    if (
        "home_team" in data.columns
        and
        "away_team" in data.columns
    ):

        teams = pd.concat(
            [
                data["home_team"],
                data["away_team"],
            ]
        )


        teams = (
            teams
            .dropna()
            .astype(str)
            .str.strip()
            .unique()
            .tolist()
        )


        america_names = {
            "america",
            "club america",
        }


        return sorted(
            [
                team
                for team in teams
                if normalize_text(team)
                not in america_names
            ]
        )


    return []


# ============================================================
# CONTAR PARTIDOS VS RIVAL
# ============================================================

def count_matches(
    matches,
    opponent,
):

    data = played_matches_only(
        matches
    )


    if data.empty:
        return 0


    target = normalize_text(
        opponent
    )


    if "opponent" in data.columns:

        mask = (
            data["opponent"]
            .astype(str)
            .map(normalize_text)
            .eq(target)
        )


    elif (
        "home_team" in data.columns
        and
        "away_team" in data.columns
    ):

        home = (
            data["home_team"]
            .astype(str)
            .map(normalize_text)
        )


        away = (
            data["away_team"]
            .astype(str)
            .map(normalize_text)
        )


        mask = (
            home.eq(target)
            |
            away.eq(target)
        )


    else:

        return 0


    filtered = data[
        mask
    ]


    if "match_id" in filtered.columns:

        return int(
            filtered["match_id"]
            .nunique()
        )


    return int(
        len(filtered)
    )


# ============================================================
# MATCH IDS JUGADOS
# ============================================================

def get_played_match_ids(
    matches
):

    data = played_matches_only(
        matches
    )


    if (
        data.empty
        or
        "match_id"
        not in data.columns
    ):

        return set()


    return set(
        data["match_id"]
        .map(normalize_match_id)
        .tolist()
    )


# ============================================================
# ESCALA ROBUSTA DATA-DRIVEN
# ============================================================

def robust_scale(series):

    numeric = pd.to_numeric(
        series,
        errors="coerce",
    )


    q1 = numeric.quantile(
        0.25
    )

    q3 = numeric.quantile(
        0.75
    )


    iqr = q3 - q1


    if (
        pd.isna(iqr)
        or
        iqr == 0
    ):

        std = numeric.std()


        if (
            pd.isna(std)
            or
            std == 0
        ):

            return 1.0


        return float(std)


    return float(iqr)


# ============================================================
# MOMENTOS HISTÓRICOS DIVERSOS
# ============================================================

def select_diverse_neighbors(
    pool,
    n_neighbors=20,
    max_per_match=2,
):

    pool = (
        pool
        .sort_values(
            "_distance"
        )
        .copy()
    )


    if "match_id" not in pool.columns:

        return (
            pool
            .head(
                n_neighbors
            )
            .copy()
        )


    selected = []

    counts = {}


    for index, row in pool.iterrows():

        match_id = normalize_match_id(
            row["match_id"]
        )


        current_count = counts.get(
            match_id,
            0,
        )


        if current_count >= max_per_match:
            continue


        selected.append(
            index
        )


        counts[
            match_id
        ] = current_count + 1


        if len(selected) >= n_neighbors:
            break


    return (
        pool.loc[
            selected
        ]
        .copy()
    )


# ============================================================
# ESCENARIO
# ============================================================

def complete_scenario_context(
    data,
    venue,
    opponent,
    minute,
    score_state,
    possession_choice,
    current_state,
    n_neighbors=20,
):

    pool = data.copy()


    pool["venue"] = (
        pool["venue"]
        .astype(str)
        .str.upper()
    )


    pool[
        "current_tactical_state"
    ] = (
        pool[
            "current_tactical_state"
        ]
        .astype(str)
        .str.upper()
    )


    pool["opponent"] = (
        pool["opponent"]
        .astype(str)
    )


    pool["minute"] = pd.to_numeric(
        pool["minute"],
        errors="coerce",
    )


    pool[
        "game_state_num"
    ] = pd.to_numeric(
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
            "No hay datos históricos suficientes."
        )


    # ========================================================
    # COMPONENTES DE DISTANCIA
    # TODOS NORMALIZADOS CON LA DISTRIBUCIÓN REAL
    # ========================================================

    minute_scale = robust_scale(
        pool["minute"]
    )


    score_scale = robust_scale(
        pool["game_state_num"]
    )


    minute_component = (
        (
            pool["minute"]
            -
            minute
        ).abs()
        /
        minute_scale
    )


    score_component = (
        (
            pool["game_state_num"]
            -
            score_state
        ).abs()
        /
        score_scale
    )


    venue_component = (
        pool["venue"]
        !=
        venue
    ).astype(float)


    state_component = (
        pool[
            "current_tactical_state"
        ]
        !=
        current_state
    ).astype(float)


    opponent_component = (
        pool["opponent"]
        .astype(str)
        .map(normalize_text)
        !=
        normalize_text(
            opponent
        )
    ).astype(float)


    components = pd.DataFrame(
        {
            "minute":
                minute_component,

            "score":
                score_component,

            "venue":
                venue_component,

            "state":
                state_component,

            "opponent":
                opponent_component,
        },
        index=
            pool.index,
    )


    # ========================================================
    # POSESIÓN SI EXISTE REALMENTE
    # ========================================================

    if (
        "america_possession_context"
        in pool.columns
    ):

        possession = pd.to_numeric(
            pool[
                "america_possession_context"
            ],
            errors="coerce",
        )


        if possession.notna().sum() > 0:

            possession_median = (
                possession.median()
            )


            possession_scale = robust_scale(
                possession
            )


            if possession_choice == "América":

                target_possession = (
                    possession[
                        possession
                        >=
                        possession_median
                    ]
                    .median()
                )


            elif possession_choice == "Rival":

                target_possession = (
                    possession[
                        possession
                        <
                        possession_median
                    ]
                    .median()
                )


            else:

                target_possession = (
                    possession_median
                )


            if pd.notna(
                target_possession
            ):

                components[
                    "possession"
                ] = (
                    (
                        possession
                        -
                        target_possession
                    ).abs()
                    /
                    possession_scale
                ).fillna(
                    components
                    .mean(
                        axis=1
                    )
                )


    # ========================================================
    # DISTANCIA FINAL
    #
    # NO HAY 2.2, 1.25, 1.50, ETC.
    # CADA COMPONENTE ENTRA CON LA MISMA ESCALA NORMALIZADA.
    # ========================================================

    pool[
        "_distance"
    ] = (
        components
        .mean(
            axis=1
        )
    )


    all_distances = (
        pool[
            "_distance"
        ]
        .dropna()
        .copy()
    )


    selected = select_diverse_neighbors(
        pool,
        n_neighbors=
            n_neighbors,
        max_per_match=
            2,
    )


    if selected.empty:

        raise ValueError(
            "No fue posible construir el escenario."
        )


    # ========================================================
    # PESOS DERIVADOS DE LAS DISTANCIAS OBSERVADAS
    # ========================================================

    selected_distances = (
        selected[
            "_distance"
        ]
        .to_numpy(
            dtype=float
        )
    )


    distance_scale = robust_scale(
        all_distances
    )


    weights = np.exp(
        -
        selected_distances
        /
        distance_scale
    )


    if (
        not np.isfinite(
            weights
        ).all()
        or
        weights.sum() == 0
    ):

        weights = np.ones(
            len(selected)
        )


    weights = (
        weights
        /
        weights.sum()
    )


    # ========================================================
    # FEATURES DEL XGBOOST
    # ========================================================

    feature_values = {}


    for feature in MODEL_FEATURES:

        if feature in STATE_FEATURES:
            continue


        if feature not in selected.columns:

            feature_values[
                feature
            ] = 0.0

            continue


        values = pd.to_numeric(
            selected[
                feature
            ],
            errors="coerce",
        ).to_numpy()


        valid = np.isfinite(
            values
        )


        if not valid.any():

            feature_values[
                feature
            ] = 0.0

            continue


        local_weights = (
            weights[
                valid
            ]
        )


        local_weights = (
            local_weights
            /
            local_weights.sum()
        )


        feature_values[
            feature
        ] = float(
            np.sum(
                values[
                    valid
                ]
                *
                local_weights
            )
        )


    # ========================================================
    # INPUTS DIRECTOS
    # ========================================================

    feature_values[
        "minute"
    ] = float(
        minute
    )


    feature_values[
        "game_state_num"
    ] = float(
        score_state
    )


    for state in STATE_LEVELS:

        variable = (
            "current_state_"
            +
            state.lower()
        )


        feature_values[
            variable
        ] = (
            1.0
            if state
            ==
            current_state
            else
            0.0
        )


    feature_row = pd.DataFrame(
        [
            {
                feature:
                    feature_values[
                        feature
                    ]
                for feature
                in MODEL_FEATURES
            }
        ]
    )


    # ========================================================
    # SIMILITUD RELATIVA DATA-DRIVEN
    #
    # MIDE QUÉ TAN CERCA ESTÁN LOS CONTEXTOS SELECCIONADOS
    # COMPARADOS CONTRA TODA LA DISTRIBUCIÓN CANDIDATA.
    # ========================================================

    scenario_distance = float(
        selected[
            "_distance"
        ]
        .mean()
    )


    similarity_percentile = float(
        (
            all_distances
            >=
            scenario_distance
        ).mean()
        *
        100
    )


    exact_opponent_rows = pool[
        pool["opponent"]
        .astype(str)
        .map(normalize_text)
        .eq(
            normalize_text(
                opponent
            )
        )
    ]


    exact_opponent_matches = (
        int(
            exact_opponent_rows[
                "match_id"
            ]
            .nunique()
        )
        if
        "match_id"
        in exact_opponent_rows.columns
        else
        0
    )


    diagnostic = {
        "similarity_percentile":
            similarity_percentile,

        "scenario_distance":
            scenario_distance,

        "exact_opponent_matches":
            exact_opponent_matches,

        "historical_moments":
            int(
                len(
                    selected
                )
            ),

        "distinct_matches":
            int(
                selected[
                    "match_id"
                ]
                .nunique()
            )
            if
            "match_id"
            in selected.columns
            else
            0,
    }


    return (
        feature_row,
        selected,
        similarity_percentile,
        diagnostic,
    )


# ============================================================
# MÉTRICAS REALES DEL TEST
# ============================================================

def calculate_model_evidence(
    data
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


    evaluation = (
        data
        .dropna(
            subset=
                required
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

        state_mask = (
            actual
            ==
            state
        )


        if state_mask.sum() == 0:
            continue


        recalls.append(
            float(
                (
                    predicted[
                        state_mask
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
# DISTRIBUCIÓN REAL DE CONFIANZA DEL MODELO
# ============================================================

def get_confidence_distribution(
    coach_data
):

    confidence_column = (
        first_existing_column(
            coach_data,
            [
                "confidence",
                "top1_probability",
                "predicted_probability",
                "prediction_probability",
                "probability",
            ],
        )
    )


    if confidence_column is None:

        return pd.Series(
            dtype=float
        )


    confidence = pd.to_numeric(
        coach_data[
            confidence_column
        ],
        errors="coerce",
    ).dropna()


    return confidence


def confidence_percentile(
    current_confidence,
    coach_data,
):

    distribution = (
        get_confidence_distribution(
            coach_data
        )
    )


    if distribution.empty:
        return np.nan


    return float(
        (
            distribution
            <=
            current_confidence
        ).mean()
        *
        100
    )


# ============================================================
# RECOMENDACIÓN DESDE LOS DATOS
# NO EXISTE FALLBACK INVENTADO
# ============================================================

def get_data_driven_action(
    coach_data,
    predicted_state,
):

    if coach_data.empty:

        return None


    action_column = (
        first_existing_column(
            coach_data,
            [
                "recommended_action",
                "coach_action",
                "action",
                "suggested_action",
                "recommendation",
                "coach_recommendation",
            ],
        )
    )


    if action_column is None:

        return None


    state_column = (
        first_existing_column(
            coach_data,
            [
                "predicted_state",
                "prediction",
                "future_state",
            ],
        )
    )


    data = coach_data.copy()


    if state_column is not None:

        filtered = data[
            data[
                state_column
            ]
            .astype(str)
            .str.upper()
            .eq(
                predicted_state
            )
        ].copy()


        if not filtered.empty:

            data = filtered


    actions = (
        data[
            action_column
        ]
        .dropna()
        .astype(str)
        .str.strip()
    )


    actions = actions[
        actions
        !=
        ""
    ]


    if actions.empty:

        return None


    counts = (
        actions
        .value_counts()
    )


    return str(
        counts.index[
            0
        ]
    )


# ============================================================
# DATOS
# ============================================================

simulator_data = load_csv(
    SIMULATOR_DATA_PATH
)


match_history = load_csv(
    MATCH_HISTORY_PATH
)


coach_all = load_csv(
    COACH_ALL_PATH
)


shap_global = load_csv(
    SHAP_GLOBAL_PATH
)


# ============================================================
# MODELO
# ============================================================

model = None


if MODEL_PATH.exists():

    try:

        model = load_model(
            MODEL_PATH
        )

    except Exception as error:

        st.error(
            f"No se pudo cargar el modelo: {error}"
        )


# ============================================================
# EVIDENCIA GLOBAL
# ============================================================

model_evidence = (
    calculate_model_evidence(
        coach_all
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
                        Inteligencia táctica
                    </div>

                </div>

            </div>

        </div>
        """
    )


    st.divider()


# ============================================================
# INPUTS VARONIL
# ============================================================

if (
    st.session_state[
        "section"
    ]
    ==
    "Varonil"
):

    opponents = (
        get_historical_opponents(
            match_history
        )
    )


    with st.sidebar:

        st.html(
            """
            <div class="sidebar-section-title">
                Escenario
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
            if venue_display
            ==
            "Local"
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
            min_value=1,
            max_value=90,
            value=60,
            key="male_minute",
        )


        st.markdown(
            "**Marcador**"
        )


        score_left, score_right = (
            st.columns(2)
        )


        with score_left:

            america_goals = (
                st.number_input(
                    "América",
                    min_value=0,
                    max_value=10,
                    value=0,
                    step=1,
                    key=
                        "male_america_goals",
                )
            )


        with score_right:

            opponent_goals = (
                st.number_input(
                    opponent,
                    min_value=0,
                    max_value=10,
                    value=0,
                    step=1,
                    key=
                        "male_opponent_goals",
                )
            )


        possession_choice = (
            st.selectbox(
                "Posesión",
                [
                    "América",
                    "Equilibrada",
                    "Rival",
                ],
                index=1,
                key=
                    "male_possession",
            )
        )


        visible_state = (
            st.selectbox(
                "Estado táctico actual",
                list(
                    STATE_DISPLAY.values()
                ),
                key=
                    "male_state",
            )
        )


        current_state = (
            STATE_REVERSE[
                visible_state
            ]
        )


        st.write("")


        if model is not None:

            st.html(
                """
                <div class="sidebar-status">
                    MODELO ACTIVO · XGBoost
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
            INTELIGENCIA TÁCTICA ·
            <strong>CLUB AMÉRICA</strong>
        </div>

        <div class="header-title">
            CoachLens
        </div>

        <div class="header-subtitle">
            Anticipa la siguiente fase táctica ·
            Entiende por qué ·
            Actúa antes de que cambie el partido
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
    index=
        nav_options.index(
            st.session_state[
                "section"
            ]
        ),
    horizontal=True,
    label_visibility=
        "collapsed",
    key=
        "top_navigation",
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

if (
    st.session_state[
        "section"
    ]
    ==
    "Varonil"
):

    st.html(
        """
        <div class="page-kicker">
            Simulador de decisiones
        </div>

        <div class="page-title">
            Lectura táctica
        </div>

        <div class="page-description">
            Pronóstico del modelo para los próximos 5 minutos.
        </div>
        """
    )


    if (
        model is None
        or
        simulator_data.empty
    ):

        st.error(
            "No están disponibles los archivos necesarios."
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


        total_matches = (
            count_matches(
                match_history,
                opponent,
            )
        )


        try:

            (
                scenario_features,
                neighbors,
                similarity_percentile,
                diagnostic,
            ) = complete_scenario_context(

                simulator_data,

                venue=
                    venue,

                opponent=
                    opponent,

                minute=
                    minute,

                score_state=
                    score_state,

                possession_choice=
                    possession_choice,

                current_state=
                    current_state,
            )


            # =================================================
            # PREDICCIÓN REAL
            # =================================================

            prediction_matrix = xgb.DMatrix(
                scenario_features[
                    MODEL_FEATURES
                ],
                feature_names=
                    MODEL_FEATURES,
            )


            probabilities = np.asarray(
                model.predict(
                    prediction_matrix
                )
            )


            if probabilities.ndim == 1:

                probabilities = (
                    probabilities
                    .reshape(
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


            top1_index = int(
                ranking[
                    0
                ]
            )


            top2_index = int(
                ranking[
                    1
                ]
            )


            predicted_state = (
                STATE_LEVELS[
                    top1_index
                ]
            )


            second_state = (
                STATE_LEVELS[
                    top2_index
                ]
            )


            confidence = float(
                probabilities[
                    top1_index
                ]
            )


            second_probability = float(
                probabilities[
                    top2_index
                ]
            )


            probability_gap = (
                confidence
                -
                second_probability
            )


            # =================================================
            # CONFIANZA RELATIVA CALCULADA DESDE TEST
            # =================================================

            confidence_rank = (
                confidence_percentile(
                    confidence,
                    coach_all,
                )
            )


            # =================================================
            # RECOMENDACIÓN SOLO SI EXISTE EN DATOS
            # =================================================

            historical_action = (
                get_data_driven_action(
                    coach_all,
                    predicted_state,
                )
            )


            # =================================================
            # TARJETAS
            # =================================================

            prediction_col, confidence_col, action_col = (
                st.columns(
                    [
                        1.0,
                        0.85,
                        1.15,
                    ],
                    gap="medium",
                )
            )


            with prediction_col:

                st.html(
                    f"""
                    <div class="result-card">

                        <div class="result-card-label">
                            Predicción a 5 minutos
                        </div>

                        <div class="result-card-value prediction-state">
                            {state_display(predicted_state)}
                        </div>

                        <div class="result-card-text">
                            Probabilidad:
                            {safe_percent(confidence)}
                            <br>
                            Segunda alternativa:
                            {state_display(second_state)}
                            ·
                            {safe_percent(second_probability)}
                        </div>

                    </div>
                    """
                )


            with confidence_col:

                percentile_text = (
                    f"{confidence_rank:.1f}%"
                    if
                    np.isfinite(
                        confidence_rank
                    )
                    else
                    "N/D"
                )


                st.html(
                    f"""
                    <div class="result-card">

                        <div class="result-card-label">
                            Confianza relativa
                        </div>

                        <div class="result-card-value">
                            {percentile_text}
                        </div>

                        <div class="result-card-text">
                            Percentil de esta confianza frente
                            a las predicciones del conjunto TEST.

                            <br><br>

                            Diferencia Top-1 / Top-2:
                            {safe_percent(probability_gap)}
                        </div>

                    </div>
                    """
                )


            with action_col:

                if historical_action is not None:

                    action_value = (
                        historical_action
                    )


                    action_description = (
                        "Recomendación recuperada del motor "
                        "histórico de CoachLens para este "
                        "estado pronosticado."
                    )


                else:

                    action_value = (
                        "NO DISPONIBLE"
                    )


                    action_description = (
                        "No existe una recomendación táctica "
                        "respaldada por los datos cargados. "
                        "CoachLens no genera una acción artificial."
                    )


                st.html(
                    f"""
                    <div class="result-card action-card">

                        <div class="result-card-label">
                            Acción registrada
                        </div>

                        <div class="action-title">
                            {action_value}
                        </div>

                        <div class="action-text">
                            {action_description}
                        </div>

                    </div>
                    """
                )


            # =================================================
            # INFORMACIÓN DEL CONTEXTO
            # =================================================

            st.write("")


            left_context, right_context = (
                st.columns(
                    2,
                    gap="large",
                )
            )


            with left_context:

                st.html(
                    f"""
                    <div class="confidence-card">

                        <div class="confidence-label">
                            Salida probabilística del modelo
                        </div>

                        <div class="confidence-number">
                            {safe_percent(confidence)}
                        </div>

                        <div class="confidence-details">

                            Primera opción:
                            <strong>
                                {state_display(predicted_state)}
                            </strong>

                            <br>

                            Segunda opción:
                            <strong>
                                {state_display(second_state)}
                            </strong>

                            ·
                            {safe_percent(second_probability)}

                            <br>

                            Separación:
                            <strong>
                                {safe_percent(probability_gap)}
                            </strong>

                        </div>

                        <div class="confidence-description">
                            Todos estos valores provienen
                            directamente de la salida probabilística
                            del XGBoost cargado.
                        </div>

                    </div>
                    """
                )


            with right_context:

                st.html(
                    f"""
                    <div class="similarity-card">

                        <div class="similarity-label">
                            Similitud relativa del contexto
                        </div>

                        <div class="similarity-number">
                            {similarity_percentile:.1f}%
                        </div>

                        <div class="similarity-bar">

                            <div
                                class="similarity-fill"
                                style="width:{similarity_percentile:.1f}%;">
                            </div>

                        </div>

                        <div class="similarity-status">
                            Percentil histórico
                        </div>

                        <div class="similarity-description">
                            Porcentaje de contextos candidatos
                            que están igual o más lejos que los
                            momentos seleccionados para construir
                            este escenario.
                        </div>

                        <div class="similarity-history">
                            Partidos jugados frente a
                            {opponent}: {total_matches}.
                        </div>

                    </div>
                    """
                )


        except Exception as error:

            st.error(
                f"No se pudo generar el análisis: {error}"
            )


# ============================================================
# FEMENIL
# ============================================================

elif (
    st.session_state[
        "section"
    ]
    ==
    "Femenil"
):

    st.html(
        """
        <div class="page-kicker">
            CoachLens Femenil
        </div>

        <div class="page-title">
            Inteligencia táctica femenil
        </div>

        <div class="page-description">
            Esta sección se conectará al pipeline y
            modelo específico de América Femenil.
        </div>
        """
    )


# ============================================================
# REVISIÓN HISTÓRICA
# ============================================================

elif (
    st.session_state[
        "section"
    ]
    ==
    "Revisión histórica"
):

    st.html(
        """
        <div class="page-kicker">
            Validación histórica
        </div>

        <div class="page-title">
            Revisión de partidos reales
        </div>

        <div class="page-description">
            Compara las predicciones guardadas del conjunto TEST
            contra lo que realmente ocurrió cinco minutos después.
        </div>
        """
    )


    if coach_all.empty:

        st.warning(
            "No hay datos de TEST disponibles."
        )


    else:

        history = (
            coach_all
            .copy()
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


        history[
            "match_date"
        ] = pd.to_datetime(
            history[
                "match_date"
            ],
            errors="coerce",
        )


        today = (
            pd.Timestamp
            .today()
            .normalize()
        )


        history = history[
            history[
                "match_date"
            ].notna()
            &
            (
                history[
                    "match_date"
                ]
                <=
                today
            )
        ].copy()


        played_ids = (
            get_played_match_ids(
                match_history
            )
        )


        if played_ids:

            history = history[
                history[
                    "_match_id"
                ]
                .isin(
                    played_ids
                )
            ].copy()


        if history.empty:

            st.warning(
                "No hay partidos históricos válidos."
            )


        else:

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


            selected_match = (
                st.selectbox(
                    "Partido",
                    matches[
                        "label"
                    ].tolist(),
                )
            )


            selected_id = (
                matches.loc[
                    matches[
                        "label"
                    ]
                    ==
                    selected_match,
                    "_match_id",
                ]
                .iloc[
                    0
                ]
            )


            match_data = history[
                history[
                    "_match_id"
                ]
                ==
                selected_id
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


            if minutes:

                selected_minute = (
                    st.select_slider(
                        "Minuto",
                        options=
                            minutes,
                        value=
                            minutes[
                                len(
                                    minutes
                                )
                                //
                                2
                            ],
                    )
                )


                case = match_data[
                    pd.to_numeric(
                        match_data[
                            "minute"
                        ],
                        errors="coerce",
                    )
                    ==
                    selected_minute
                ].iloc[
                    0
                ]


                current_raw = str(
                    case.get(
                        "current_state",
                        "",
                    )
                ).upper()


                predicted_raw = str(
                    case.get(
                        "predicted_state",
                        "",
                    )
                ).upper()


                second_raw = str(
                    case.get(
                        "second_state",
                        "",
                    )
                ).upper()


                actual_raw = str(
                    case.get(
                        "actual_future_state",
                        "",
                    )
                ).upper()


                current = state_display(
                    current_raw
                )


                predicted = state_display(
                    predicted_raw
                )


                second = state_display(
                    second_raw
                )


                actual = state_display(
                    actual_raw
                )


                # =================================================
                # RESULTADO CALCULADO
                # =================================================

                if (
                    actual_raw
                    ==
                    predicted_raw
                ):

                    result_text = (
                        "ACIERTO EXACTO"
                    )


                    result_description = (
                        "El estado observado coincidió con "
                        "la primera predicción."
                    )


                    result_class = (
                        "history-result-exact"
                    )


                elif (
                    second_raw
                    not in
                    [
                        "",
                        "NAN",
                        "NONE",
                    ]
                    and
                    actual_raw
                    ==
                    second_raw
                ):

                    result_text = (
                        "ACIERTO TOP-2"
                    )


                    result_description = (
                        "El estado observado coincidió con "
                        "la segunda alternativa."
                    )


                    result_class = (
                        "history-result-top2"
                    )


                else:

                    result_text = (
                        "NO ACIERTO"
                    )


                    result_description = (
                        "El estado real no estaba entre "
                        "las dos alternativas principales."
                    )


                    result_class = (
                        "history-result-miss"
                    )


                h1, h2, h3, h4 = (
                    st.columns(
                        [
                            1,
                            1,
                            1,
                            0.95,
                        ],
                        gap="medium",
                    )
                )


                with h1:

                    st.html(
                        f"""
                        <div class="metric-card">

                            <div class="metric-label">
                                Estado actual
                            </div>

                            <div class="history-state-value">
                                {current}
                            </div>

                            <div class="metric-subtext">
                                Estado registrado en el
                                minuto seleccionado.
                            </div>

                        </div>
                        """
                    )


                with h2:

                    st.html(
                        f"""
                        <div class="metric-card">

                            <div class="metric-label">
                                Primera predicción
                            </div>

                            <div class="history-state-value">
                                {predicted}
                            </div>

                            <div class="metric-subtext">
                                Primera alternativa guardada
                                por el modelo.
                            </div>

                        </div>
                        """
                    )


                with h3:

                    st.html(
                        f"""
                        <div class="metric-card">

                            <div class="metric-label">
                                Estado real +5 min
                            </div>

                            <div class="history-state-value">
                                {actual}
                            </div>

                            <div class="metric-subtext">
                                Estado observado realmente
                                cinco minutos después.
                            </div>

                        </div>
                        """
                    )


                with h4:

                    st.html(
                        f"""
                        <div class="metric-card">

                            <div class="metric-label">
                                Resultado
                            </div>

                            <div class="history-result-value {result_class}">
                                {result_text}
                            </div>

                            <div class="metric-subtext">
                                {result_description}
                            </div>

                        </div>
                        """
                    )


                if (
                    second_raw
                    not in
                    [
                        "",
                        "NAN",
                        "NONE",
                    ]
                ):

                    st.caption(
                        f"Segunda alternativa registrada: {second}"
                    )


                # =================================================
                # EXPLICACIÓN CON MÉTRICAS CALCULADAS
                # =================================================

                historical_top1 = (
                    safe_percent(
                        model_evidence[
                            "top1"
                        ]
                    )
                )


                historical_top2 = (
                    safe_percent(
                        model_evidence[
                            "top2"
                        ]
                    )
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
                                    ACIERTO EXACTO
                                </div>

                                <div class="history-explanation-text">
                                    El estado observado coincide
                                    con la alternativa de mayor
                                    probabilidad del modelo.
                                </div>

                            </div>


                            <div class="history-explanation-item">

                                <div class="history-explanation-badge top2">
                                    ACIERTO TOP-2
                                </div>

                                <div class="history-explanation-text">
                                    El estado real corresponde
                                    a la segunda alternativa
                                    registrada por el modelo.
                                </div>

                            </div>


                            <div class="history-explanation-item">

                                <div class="history-explanation-badge miss">
                                    NO ACIERTO
                                </div>

                                <div class="history-explanation-text">
                                    El estado real no apareció
                                    dentro de las dos principales
                                    alternativas.
                                </div>

                            </div>

                        </div>


                        <div class="history-explanation-note">

                            Rendimiento calculado actualmente
                            sobre el conjunto TEST:

                            <strong>
                                Top-1: {historical_top1}
                            </strong>

                            ·

                            <strong>
                                Top-2: {historical_top2}
                            </strong>.

                        </div>

                    </div>
                    """
                )


# ============================================================
# MODELO
# ============================================================

elif (
    st.session_state[
        "section"
    ]
    ==
    "Modelo"
):

    st.html(
        """
        <div class="page-kicker">
            Evidencia y metodología
        </div>

        <div class="page-title">
            Rendimiento del modelo
        </div>

        <div class="page-description">
            Métricas calculadas a partir de las predicciones
            del conjunto TEST.
        </div>
        """
    )


    m1, m2, m3, m4 = (
        st.columns(
            4,
            gap="medium",
        )
    )


    with m1:

        st.html(
            f"""
            <div class="metric-card">

                <div class="metric-label">
                    Exactitud Top-1
                </div>

                <div class="metric-value">
                    {safe_percent(model_evidence["top1"])}
                </div>

                <div class="metric-subtext">
                    Coincidencia exacta entre la primera
                    predicción y el estado observado.
                </div>

            </div>
            """
        )


    with m2:

        st.html(
            f"""
            <div class="metric-card">

                <div class="metric-label">
                    Exactitud Top-2
                </div>

                <div class="metric-value">
                    {safe_percent(model_evidence["top2"])}
                </div>

                <div class="metric-subtext">
                    El estado observado aparece dentro
                    de las dos primeras alternativas.
                </div>

            </div>
            """
        )


    with m3:

        st.html(
            f"""
            <div class="metric-card">

                <div class="metric-label">
                    Baseline
                </div>

                <div class="metric-value">
                    {safe_percent(model_evidence["baseline"])}
                </div>

                <div class="metric-subtext">
                    Persistencia del estado actual como
                    referencia predictiva.
                </div>

            </div>
            """
        )


    with m4:

        st.html(
            f"""
            <div class="metric-card">

                <div class="metric-label">
                    Exactitud balanceada
                </div>

                <div class="metric-value">
                    {safe_percent(model_evidence["balanced"])}
                </div>

                <div class="metric-subtext">
                    Promedio del recall calculado entre
                    las clases tácticas presentes.
                </div>

            </div>
            """
        )


    st.caption(
        f"Evaluación calculada sobre "
        f"{model_evidence['n']:,} observaciones del conjunto TEST."
    )


    st.divider()


    st.markdown(
        "### Cómo funciona CoachLens"
    )


    c1, c2, c3 = st.columns(
        3
    )


    with c1:

        st.markdown(
            """
**Predicción**

El archivo XGBoost cargado calcula una distribución de
probabilidad sobre los cinco estados tácticos.
"""
        )


    with c2:

        st.markdown(
            """
**Escenario**

Las variables no seleccionadas directamente se estiman
a partir de contextos históricos similares utilizando
escalas calculadas desde los propios datos.
"""
        )


    with c3:

        st.markdown(
            """
**Evaluación**

Las métricas mostradas se vuelven a calcular desde las
predicciones almacenadas del conjunto TEST.
"""
        )


    # ========================================================
    # SHAP
    # ========================================================

    if (
        not shap_global.empty
        and
        "mean_abs_shap"
        in shap_global.columns
        and
        "feature"
        in shap_global.columns
    ):

        st.divider()


        st.markdown(
            "### Variables con mayor influencia"
        )


        shap_plot = (
            shap_global
            .sort_values(
                "mean_abs_shap",
                ascending=False,
            )
            .head(
                12
            )
            .copy()
        )


        shap_plot[
            "label"
        ] = (
            shap_plot[
                "feature"
            ]
            .map(
                FEATURE_NAMES
            )
            .fillna(
                shap_plot[
                    "feature"
                ]
            )
        )


        shap_plot = (
            shap_plot
            .sort_values(
                "mean_abs_shap",
                ascending=True,
            )
        )


        shap_figure = px.bar(
            shap_plot,
            x=
                "mean_abs_shap",
            y=
                "label",
            orientation=
                "h",
        )


        shap_figure.update_traces(
            marker_color=
                AMERICA_YELLOW
        )


        shap_figure.update_layout(
            paper_bgcolor=
                "rgba(0,0,0,0)",

            plot_bgcolor=
                "rgba(0,0,0,0)",

            font=dict(
                color=
                    TEXT_DARK
            ),

            xaxis_title=
                "Importancia SHAP promedio",

            yaxis_title=
                "",

            height=
                450,
        )


        st.plotly_chart(
            shap_figure,
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