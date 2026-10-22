from pathlib import Path
import unicodedata

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
import xgboost as xgb

from config.charts import (
    apply_coachlens_chart_style,
    BEFORE_COLOR,
    AFTER_COLOR,
    COACH_YELLOW,
    COACH_NAVY,
)


# ============================================================
# STREAMLIT
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

CSS_PATH = DASHBOARD_DIR / "styles" / "america.css"


# ============================================================
# VARONIL
# ============================================================

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


# ============================================================
# FEMENIL
# ============================================================

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
# COMPLEMENTARIO
# ============================================================

COMPLEMENTARY_SUBSTITUTIONS_PATH = (
    DATA_DIR
    / "complementary_substitutions.csv"
)

COMPLEMENTARY_SET_PIECES_PATH = (
    DATA_DIR
    / "complementary_set_pieces.csv"
)


# ============================================================
# CSS
# ============================================================

def load_css(path):

    if not path.exists():
        return

    css = path.read_text(
        encoding="utf-8"
    )

    st.html(
        f"""
        <style>
        {css}
        </style>
        """
    )


load_css(CSS_PATH)


# ============================================================
# ESTADOS TÁCTICOS
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


# ============================================================
# PLAYBOOK
# ============================================================

TACTICAL_PLAYBOOK = {

    "CONTROL": {
        "title":
            "Sostener el control",

        "text":
            (
                "Mantener la posesión con paciencia, "
                "utilizar la amplitud y acelerar cuando "
                "aparezca una ventaja clara."
            ),
    },

    "PRESSURE": {
        "title":
            "Subir la presión",

        "text":
            (
                "Cerrar líneas interiores y activar la "
                "presión cuando el rival juegue hacia atrás "
                "o tenga un control incómodo."
            ),
    },

    "TRANSITION": {
        "title":
            "Atacar la transición",

        "text":
            (
                "Priorizar un primer pase limpio tras "
                "recuperar y atacar rápidamente los "
                "espacios disponibles."
            ),
    },

    "PROTECTION": {
        "title":
            "Cerrar espacios",

        "text":
            (
                "Proteger el carril central, reducir "
                "distancias entre líneas y evitar pérdidas "
                "en zonas comprometidas."
            ),
    },

    "URGENCY": {
        "title":
            "Aumentar el ritmo",

        "text":
            (
                "Acelerar la circulación, ocupar zonas "
                "de remate y reaccionar rápidamente "
                "después de una pérdida."
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

    return pd.read_csv(
        path
    )


@st.cache_resource
def load_model(path):

    model = xgb.Booster()

    model.load_model(
        str(path)
    )

    return model


# ============================================================
# HELPERS GENERALES
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
        )
        !=
        "Mn"
    )


def normalize_match_id(value):

    value = str(
        value
    )

    if value.endswith(
        ".0"
    ):
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

        value = float(
            value
        )

        if not np.isfinite(
            value
        ):
            return "N/D"

        if abs(value) <= 1:
            value *= 100

        return f"{value:.1f}%"

    except Exception:

        return "N/D"


def percentage_points(value):

    try:

        value = float(
            value
        )

        if not np.isfinite(
            value
        ):
            return "N/D"

        return (
            f"{value * 100:+.1f} pp"
        )

    except Exception:

        return "N/D"


def metric_from_table(
    data,
    metric_name,
):

    if data.empty:
        return np.nan

    if not {
        "metric",
        "value",
    }.issubset(
        data.columns
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


def interpret_change_value(
    value
):

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

    if text in positives:
        return "CAMBIA"

    return "NO CAMBIA"


# ============================================================
# METADATA DE PARTIDOS
# ============================================================

def build_match_metadata(
    match_history,
    team_name,
):

    if (
        match_history.empty
        or
        "match_id"
        not in match_history.columns
    ):

        return pd.DataFrame(
            columns=[
                "_match_id",
                "match_date",
                "opponent",
            ]
        )


    data = (
        match_history
        .copy()
    )


    data["_match_id"] = (
        data["match_id"]
        .map(
            normalize_match_id
        )
    )


    if (
        "match_date"
        in data.columns
    ):

        data["match_date"] = pd.to_datetime(
            data["match_date"],
            errors="coerce",
        )

    else:

        data["match_date"] = pd.NaT


    if (
        "opponent"
        in data.columns
    ):

        data["opponent"] = (
            data["opponent"]
            .astype(str)
        )

    else:

        home_candidates = [
            "home_team.name",
            "home_team_name",
            "home_team",
        ]

        away_candidates = [
            "away_team.name",
            "away_team_name",
            "away_team",
        ]


        home_column = next(
            (
                column
                for column
                in home_candidates
                if column
                in data.columns
            ),
            None,
        )


        away_column = next(
            (
                column
                for column
                in away_candidates
                if column
                in data.columns
            ),
            None,
        )


        if (
            home_column is not None
            and
            away_column is not None
        ):

            data["opponent"] = np.where(
                data[
                    home_column
                ]
                .astype(str)
                .map(
                    normalize_text
                )
                ==
                normalize_text(
                    team_name
                ),

                data[
                    away_column
                ],

                data[
                    home_column
                ],
            )

        else:

            data["opponent"] = (
                "Rival no disponible"
            )


    return (
        data[
            [
                "_match_id",
                "match_date",
                "opponent",
            ]
        ]
        .drop_duplicates(
            "_match_id"
        )
    )


def attach_match_metadata(
    data,
    metadata,
):

    if data.empty:
        return data.copy()


    result = (
        data.copy()
    )


    if (
        "match_id"
        not in result.columns
    ):

        return result


    result["_match_id"] = (
        result["match_id"]
        .map(
            normalize_match_id
        )
    )


    for column in [
        "match_date",
        "opponent",
    ]:

        if column in result.columns:

            result = result.drop(
                columns=[
                    column
                ]
            )


    if metadata.empty:

        result["match_date"] = pd.NaT

        result["opponent"] = (
            "Rival no disponible"
        )

        return result


    result = result.merge(
        metadata,
        on="_match_id",
        how="left",
    )


    result["opponent"] = (
        result["opponent"]
        .fillna(
            "Rival no disponible"
        )
        .astype(str)
    )


    return result


def make_match_label(
    match_date,
    opponent,
    team_name,
):

    parsed_date = pd.to_datetime(
        match_date,
        errors="coerce",
    )


    if pd.isna(
        parsed_date
    ):

        date_text = (
            "Fecha no disponible"
        )

    else:

        date_text = (
            parsed_date.strftime(
                "%Y-%m-%d"
            )
        )


    return (
        f"{date_text} · "
        f"{team_name} vs "
        f"{opponent}"
    )


# ============================================================
# PARTIDOS JUGADOS
# ============================================================

def played_matches_only(
    data
):

    if data.empty:

        return data.copy()


    result = (
        data.copy()
    )


    if {
        "home_score",
        "away_score",
    }.issubset(
        result.columns
    ):

        home_score = pd.to_numeric(
            result["home_score"],
            errors="coerce",
        )

        away_score = pd.to_numeric(
            result["away_score"],
            errors="coerce",
        )


        result = (
            result[
                home_score.notna()
                &
                away_score.notna()
            ]
            .copy()
        )


    if (
        "match_date"
        in result.columns
    ):

        result["match_date"] = pd.to_datetime(
            result["match_date"],
            errors="coerce",
        )


        today = (
            pd.Timestamp
            .today()
            .normalize()
        )


        result = (
            result[
                result[
                    "match_date"
                ].notna()
                &
                (
                    result[
                        "match_date"
                    ]
                    <=
                    today
                )
            ]
            .copy()
        )


    return result


def get_historical_opponents(
    matches
):

    data = played_matches_only(
        matches
    )


    if (
        data.empty
        or
        "opponent"
        not in data.columns
    ):

        return []


    return sorted(
        data["opponent"]
        .dropna()
        .astype(str)
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
        "opponent"
        not in data.columns
    ):

        return 0


    result = data[
        data["opponent"]
        .astype(str)
        .map(
            normalize_text
        )
        ==
        normalize_text(
            opponent
        )
    ]


    if (
        "match_id"
        in result.columns
    ):

        return int(
            result[
                "match_id"
            ]
            .nunique()
        )


    return int(
        len(
            result
        )
    )


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
        .map(
            normalize_match_id
        )
        .tolist()
    )


# ============================================================
# MARCADOR + / -
# ============================================================

def increase_score(
    key,
    max_value=10,
):

    current_value = int(
        st.session_state.get(
            key,
            0,
        )
    )

    st.session_state[
        key
    ] = min(
        max_value,
        current_value + 1,
    )


def decrease_score(
    key,
    min_value=0,
):

    current_value = int(
        st.session_state.get(
            key,
            0,
        )
    )

    st.session_state[
        key
    ] = max(
        min_value,
        current_value - 1,
    )


def score_stepper(
    label,
    key,
    default=0,
    min_value=0,
    max_value=10,
):

    if (
        key
        not in
        st.session_state
    ):

        st.session_state[
            key
        ] = int(
            default
        )


    st.html(
        f"""
        <div class="score-team-label">
            {label}
        </div>
        """
    )


    (
        minus_col,
        score_col,
        plus_col,
    ) = st.columns(
        [
            1,
            1.45,
            1,
        ],
        gap="small",
    )


    with minus_col:

        st.button(
            "−",
            key=
                f"{key}_minus",
            use_container_width=True,
            on_click=
                decrease_score,
            args=(
                key,
                min_value,
            ),
        )


    with score_col:

        st.html(
            f"""
            <div class="score-value-box">
                {int(st.session_state[key])}
            </div>
            """
        )


    with plus_col:

        st.button(
            "+",
            key=
                f"{key}_plus",
            use_container_width=True,
            on_click=
                increase_score,
            args=(
                key,
                max_value,
            ),
        )


    return int(
        st.session_state[
            key
        ]
    )


# ============================================================
# FEATURES AMIGABLES
# ============================================================

def friendly_feature_name(
    feature
):

    mapping = {

        "game_state_num":
            "Estado del marcador",

        "minute":
            "Minuto del partido",

        "america_support_share_10_last5":
            "Apoyos cercanos",

        "america_nearest_opponent_distance_last5":
            "Distancia al rival más cercano",

        "recovery_height_diff_last5":
            "Altura de recuperación",

        "america_pressure_share_5_last5":
            "Intensidad de presión",

        "current_state_urgency":
            "Estar actualmente en urgencia",

        "current_state_protection":
            "Estar actualmente en protección",

        "current_state_control":
            "Estar actualmente en control",

        "current_state_pressure":
            "Estar actualmente en presión",

        "current_state_transition":
            "Estar actualmente en transición",

        "progressive_pass_diff_last5":
            "Pases progresivos",

        "pressure_diff_last5":
            "Diferencia de presión",

        "entries_box_diff_last5":
            "Entradas al área",

        "america_field_tilt_last5":
            "Dominio territorial",
    }


    if feature in mapping:

        return mapping[
            feature
        ]


    return (
        str(feature)
        .replace(
            "america_",
            "",
        )
        .replace(
            "_last5",
            "",
        )
        .replace(
            "_",
            " ",
        )
        .strip()
        .capitalize()
    )


# ============================================================
# ESCALA ROBUSTA
# ============================================================

def robust_scale(
    series
):

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


    scale = (
        q3
        -
        q1
    )


    if (
        not np.isfinite(
            scale
        )
        or
        scale == 0
    ):

        scale = (
            numeric.std()
        )


    if (
        not np.isfinite(
            scale
        )
        or
        scale == 0
    ):

        scale = 1.0


    return float(
        scale
    )


# ============================================================
# CONTEXTO VARONIL
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

    pool = (
        data.copy()
    )


    if pool.empty:

        raise ValueError(
            "No existen contextos históricos."
        )


    medians = (
        medians
        or
        {}
    )


    pool["minute"] = pd.to_numeric(
        pool["minute"],
        errors="coerce",
    )


    pool["game_state_num"] = pd.to_numeric(
        pool["game_state_num"],
        errors="coerce",
    )


    pool = (
        pool
        .dropna(
            subset=[
                "minute",
                "game_state_num",
            ]
        )
        .copy()
    )


    components = pd.DataFrame(
        index=
            pool.index
    )


    components[
        "minute"
    ] = (
        (
            pool[
                "minute"
            ]
            -
            float(
                minute
            )
        ).abs()
        /
        robust_scale(
            pool[
                "minute"
            ]
        )
    )


    components[
        "score"
    ] = (
        (
            pool[
                "game_state_num"
            ]
            -
            float(
                score_state
            )
        ).abs()
        /
        robust_scale(
            pool[
                "game_state_num"
            ]
        )
    )


    if (
        "venue"
        in pool.columns
    ):

        components[
            "venue"
        ] = (
            pool[
                "venue"
            ]
            .astype(str)
            .str.upper()
            .ne(
                str(
                    venue
                ).upper()
            )
            .astype(float)
        )


    if (
        "opponent"
        in pool.columns
    ):

        components[
            "opponent"
        ] = (
            pool[
                "opponent"
            ]
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


    if (
        "current_tactical_state"
        in pool.columns
    ):

        components[
            "state"
        ] = (
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


    pool[
        "_distance"
    ] = (
        components
        .mean(
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


    cutoff = float(
        distances.quantile(
            0.25
        )
    )


    selected = (
        pool[
            pool[
                "_distance"
            ]
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


    distance_scale = (
        robust_scale(
            distances
        )
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

            expected = (
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
                expected
                ==
                str(
                    current_state
                ).upper()
            )

            continue


        if (
            feature
            not in
            selected.columns
        ):

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


        if (
            valid.sum()
            ==
            0
        ):

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
# CONTEXTO FEMENIL
# ============================================================

def build_female_context_from_history(
    data,
    opponent,
    venue,
    minute,
    score_state,
    current_state,
    feature_names,
    medians=None,
):

    pool = (
        data.copy()
    )


    if pool.empty:

        raise ValueError(
            "No existen contextos históricos para Femenil."
        )


    medians = (
        medians
        or
        {}
    )


    pool[
        "minute"
    ] = pd.to_numeric(
        pool[
            "minute"
        ],
        errors="coerce",
    )


    pool[
        "game_state_num"
    ] = pd.to_numeric(
        pool[
            "game_state_num"
        ],
        errors="coerce",
    )


    pool = (
        pool
        .dropna(
            subset=[
                "minute",
                "game_state_num",
            ]
        )
        .copy()
    )


    if pool.empty:

        raise ValueError(
            "No existen observaciones femeniles válidas."
        )


    candidates = (
        pool.copy()
    )


    if (
        "current_tactical_state"
        in candidates.columns
    ):

        filtered = (
            candidates[
                candidates[
                    "current_tactical_state"
                ]
                .astype(str)
                .str.upper()
                .eq(
                    str(
                        current_state
                    ).upper()
                )
            ]
            .copy()
        )


        if not filtered.empty:

            candidates = (
                filtered
            )


    if (
        "venue"
        in candidates.columns
    ):

        filtered = (
            candidates[
                candidates[
                    "venue"
                ]
                .astype(str)
                .str.upper()
                .eq(
                    str(
                        venue
                    ).upper()
                )
            ]
            .copy()
        )


        if not filtered.empty:

            candidates = (
                filtered
            )


    if (
        "opponent"
        in candidates.columns
    ):

        filtered = (
            candidates[
                candidates[
                    "opponent"
                ]
                .astype(str)
                .map(
                    normalize_text
                )
                .eq(
                    normalize_text(
                        opponent
                    )
                )
            ]
            .copy()
        )


        if not filtered.empty:

            candidates = (
                filtered
            )


    components = pd.DataFrame(
        index=
            candidates.index
    )


    components[
        "minute"
    ] = (
        (
            candidates[
                "minute"
            ]
            -
            float(
                minute
            )
        ).abs()
        /
        robust_scale(
            candidates[
                "minute"
            ]
        )
    )


    components[
        "score"
    ] = (
        (
            candidates[
                "game_state_num"
            ]
            -
            float(
                score_state
            )
        ).abs()
        /
        robust_scale(
            candidates[
                "game_state_num"
            ]
        )
    )


    if (
        "venue"
        in candidates.columns
    ):

        components[
            "venue"
        ] = (
            candidates[
                "venue"
            ]
            .astype(str)
            .str.upper()
            .ne(
                str(
                    venue
                ).upper()
            )
            .astype(float)
        )


    if (
        "opponent"
        in candidates.columns
    ):

        components[
            "opponent"
        ] = (
            candidates[
                "opponent"
            ]
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


    candidates[
        "_distance"
    ] = (
        components
        .mean(
            axis=1
        )
    )


    candidates = (
        candidates
        .replace(
            [
                np.inf,
                -np.inf,
            ],
            np.nan,
        )
        .dropna(
            subset=[
                "_distance",
            ]
        )
        .sort_values(
            "_distance"
        )
        .copy()
    )


    if candidates.empty:

        raise ValueError(
            "No se encontraron contextos femeniles comparables."
        )


    neighbor_count = int(
        np.ceil(
            np.sqrt(
                len(
                    candidates
                )
            )
        )
    )


    neighbor_count = max(
        1,
        neighbor_count,
    )


    selected = (
        candidates
        .head(
            neighbor_count
        )
        .copy()
    )


    distance_scale = (
        robust_scale(
            candidates[
                "_distance"
            ]
        )
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


        if (
            feature
            not in
            selected.columns
        ):

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


        if (
            valid.sum()
            ==
            0
        ):

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


    mean_selected_distance = float(
        selected[
            "_distance"
        ].mean()
    )


    similarity = float(
        (
            candidates[
                "_distance"
            ]
            >=
            mean_selected_distance
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

    target = (
        "future_tactical_state_5min"
    )


    if (
        neighbors.empty
        or
        target
        not in
        neighbors.columns
    ):

        return {
            "state": None,
            "share": np.nan,
            "observations": 0,
            "distribution": pd.DataFrame(),
        }


    data = (
        neighbors.copy()
    )


    data[
        target
    ] = (
        data[
            target
        ]
        .astype(str)
        .str.upper()
        .str.strip()
    )


    transition_data = (
        data[
            data[
                target
            ]
            .ne(
                str(
                    current_state
                ).upper()
            )
            &
            data[
                target
            ]
            .isin(
                STATE_LEVELS
            )
        ]
        .copy()
    )


    if transition_data.empty:

        return {
            "state": None,
            "share": np.nan,
            "observations": 0,
            "distribution": pd.DataFrame(),
        }


    if (
        "_weight"
        not in
        transition_data.columns
    ):

        transition_data[
            "_weight"
        ] = 1.0


    distribution = (
        transition_data
        .groupby(
            target,
            as_index=False,
        )
        .agg(
            weight=(
                "_weight",
                "sum",
            ),
            observations=(
                target,
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


    first = (
        distribution
        .iloc[0]
    )


    return {
        "state":
            str(
                first[
                    target
                ]
            ),

        "share":
            float(
                first[
                    "share"
                ]
            ),

        "observations":
            int(
                len(
                    transition_data
                )
            ),

        "distribution":
            distribution,
    }


# ============================================================
# ACCIONES
# ============================================================

def get_coach_action(
    state
):

    return TACTICAL_PLAYBOOK.get(
        str(
            state
        ).upper(),

        {
            "title":
                "Monitorear el escenario",

            "text":
                (
                    "Revisar la evolución del contexto "
                    "antes de realizar un ajuste."
                ),
        },
    )


# ============================================================
# MÉTRICAS VARONIL
# ============================================================

def calculate_male_metrics(
    data
):

    output = {
        "top1": np.nan,
        "top2": np.nan,
        "baseline": np.nan,
        "balanced": np.nan,
        "n": 0,
    }


    required = {
        "actual_future_state",
        "predicted_state",
    }


    if (
        data.empty
        or
        not required.issubset(
            data.columns
        )
    ):

        return output


    df = (
        data
        .dropna(
            subset=
                list(
                    required
                )
        )
        .copy()
    )


    if df.empty:

        return output


    actual = (
        df[
            "actual_future_state"
        ]
        .astype(str)
        .str.upper()
    )


    predicted = (
        df[
            "predicted_state"
        ]
        .astype(str)
        .str.upper()
    )


    output[
        "n"
    ] = len(
        df
    )


    output[
        "top1"
    ] = float(
        (
            actual
            ==
            predicted
        ).mean()
    )


    if (
        "second_state"
        in df.columns
    ):

        second = (
            df[
                "second_state"
            ]
            .astype(str)
            .str.upper()
        )


        output[
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


    if (
        "current_state"
        in df.columns
    ):

        current = (
            df[
                "current_state"
            ]
            .astype(str)
            .str.upper()
        )


        output[
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
            actual
            ==
            state
        )


        if (
            mask.sum()
            >
            0
        ):

            recalls.append(
                (
                    predicted[
                        mask
                    ]
                    ==
                    state
                ).mean()
            )


    if recalls:

        output[
            "balanced"
        ] = float(
            np.mean(
                recalls
            )
        )


    return output


# ============================================================
# IMPORTANCIA XGBOOST
# ============================================================

def get_xgb_importance(
    model,
    top_n=8,
):

    if model is None:

        return pd.DataFrame()


    score = (
        model.get_score(
            importance_type=
                "gain"
        )
    )


    if not score:

        return pd.DataFrame()


    output = pd.DataFrame(
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


    output[
        "label"
    ] = (
        output[
            "feature"
        ]
        .map(
            friendly_feature_name
        )
    )


    return (
        output
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
# SIMULACIÓN
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


    simulated = rng.choice(
        state_levels,
        size=n_simulations,
        p=probabilities,
    )


    result = (
        pd.Series(
            simulated
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
            name=
                "simulations"
        )
    )


    result[
        "share"
    ] = (
        result[
            "simulations"
        ]
        /
        n_simulations
    )


    result[
        "percentage"
    ] = (
        result[
            "share"
        ]
        *
        100
    )


    result[
        "label"
    ] = (
        result[
            "state"
        ]
        .map(
            STATE_DISPLAY
        )
    )


    return result


def simulate_female_scenarios(
    probability_change,
    n_simulations=1000,
    seed=42,
):

    rng = np.random.default_rng(
        seed
    )


    changes = (
        rng.random(
            n_simulations
        )
        <
        probability_change
    )


    change_count = int(
        changes.sum()
    )


    stable_count = (
        n_simulations
        -
        change_count
    )


    result = pd.DataFrame(
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


    result[
        "share"
    ] = (
        result[
            "simulations"
        ]
        /
        n_simulations
    )


    result[
        "percentage"
    ] = (
        result[
            "share"
        ]
        *
        100
    )


    return result


# ============================================================
# IDENTIDAD DE JUEGO
# ============================================================

def first_existing_column(
    data,
    candidates,
):

    for column in candidates:

        if column in data.columns:

            return column

    return None


def safe_numeric_mean(
    data,
    column,
):

    if (
        column is None
        or
        column
        not in
        data.columns
        or
        data.empty
    ):

        return np.nan


    values = pd.to_numeric(
        data[
            column
        ],
        errors="coerce",
    )


    values = values[
        np.isfinite(
            values
        )
    ]


    if values.empty:

        return np.nan


    return float(
        values.mean()
    )


def build_play_identity_metrics(
    data
):

    if data.empty:

        return []


    metric_definitions = [

        {
            "section":
                "Fase ofensiva",

            "name":
                "Construcción",

            "candidates": [
                "pass_diff_last5",
                "passes_diff_last5",
                "america_pass_share_last5",
                "america_possession_share_last5",
            ],
        },

        {
            "section":
                "Fase ofensiva",

            "name":
                "Progresión",

            "candidates": [
                "progressive_pass_diff_last5",
                "america_progressive_passes_last5",
                "progressive_passes_last5",
            ],
        },

        {
            "section":
                "Fase ofensiva",

            "name":
                "Generación de ocasiones",

            "candidates": [
                "entries_box_diff_last5",
                "shots_diff_last5",
                "america_shots_last5",
                "america_entries_box_last5",
            ],
        },

        {
            "section":
                "Fase defensiva",

            "name":
                "Presión",

            "candidates": [
                "pressure_diff_last5",
                "america_pressure_share_5_last5",
                "america_pressure_share_last5",
            ],
        },

        {
            "section":
                "Fase defensiva",

            "name":
                "Recuperaciones",

            "candidates": [
                "recovery_height_diff_last5",
                "recoveries_diff_last5",
                "america_recoveries_last5",
            ],
        },

        {
            "section":
                "Fase defensiva",

            "name":
                "Dominio territorial",

            "candidates": [
                "america_field_tilt_last5",
                "field_tilt_diff_last5",
                "field_tilt_last5",
            ],
        },
    ]


    output = []


    for metric in metric_definitions:

        column = first_existing_column(
            data,
            metric[
                "candidates"
            ],
        )


        if column is None:

            continue


        value = safe_numeric_mean(
            data,
            column,
        )


        if not np.isfinite(
            value
        ):

            continue


        output.append(
            {
                "section":
                    metric[
                        "section"
                    ],

                "name":
                    metric[
                        "name"
                    ],

                "column":
                    column,

                "value":
                    value,
            }
        )


    return output


def relative_index(
    selected_value,
    global_value,
):

    if (
        not np.isfinite(
            selected_value
        )
        or
        not np.isfinite(
            global_value
        )
        or
        global_value == 0
    ):

        return np.nan


    return (
        selected_value
        /
        global_value
        *
        100
    )


def identity_reading(
    value
):

    if not np.isfinite(
        value
    ):

        return (
            "Sin datos"
        )


    if value >= 110:

        return (
            "Por encima de su referencia histórica"
        )


    if value <= 90:

        return (
            "Por debajo de su referencia histórica"
        )


    return (
        "Cerca de su referencia histórica"
    )


# ============================================================
# USO DE JUGADORES
# ============================================================

def build_player_usage_summary(
    data
):

    if data.empty:

        return {}


    result = {}


    if (
        "player_out"
        in data.columns
    ):

        series = (
            data[
                "player_out"
            ]
            .dropna()
            .astype(str)
        )


        if not series.empty:

            counts = (
                series
                .value_counts()
            )

            result[
                "most_subbed_out"
            ] = (
                counts.index[0]
            )

            result[
                "most_subbed_out_count"
            ] = int(
                counts.iloc[0]
            )


    if (
        "player_in"
        in data.columns
    ):

        series = (
            data[
                "player_in"
            ]
            .dropna()
            .astype(str)
        )


        if not series.empty:

            counts = (
                series
                .value_counts()
            )

            result[
                "most_subbed_in"
            ] = (
                counts.index[0]
            )

            result[
                "most_subbed_in_count"
            ] = int(
                counts.iloc[0]
            )


    if (
        "minute"
        in data.columns
    ):

        minutes = (
            pd.to_numeric(
                data[
                    "minute"
                ],
                errors="coerce",
            )
            .dropna()
        )


        if not minutes.empty:

            result[
                "median_minute"
            ] = int(
                round(
                    minutes.median()
                )
            )


    if {
        "player_out",
        "player_in",
    }.issubset(
        data.columns
    ):

        pairs = (
            data[
                [
                    "player_out",
                    "player_in",
                ]
            ]
            .dropna()
            .astype(str)
            .copy()
        )


        if not pairs.empty:

            pairs[
                "pair"
            ] = (
                pairs[
                    "player_out"
                ]
                +
                " → "
                +
                pairs[
                    "player_in"
                ]
            )


            pair_counts = (
                pairs[
                    "pair"
                ]
                .value_counts()
            )


            result[
                "common_pair"
            ] = (
                pair_counts.index[0]
            )


            result[
                "common_pair_count"
            ] = int(
                pair_counts.iloc[0]
            )


    return result


# ============================================================
# COMPLEMENTARIOS
# ============================================================

def render_before_after(
    values,
    decimal_places=1,
):

    columns = st.columns(
        len(
            values
        ),
        gap="medium",
    )


    changes = {}


    for (
        column,
        (
            label,
            before,
            after,
        ),
    ) in zip(
        columns,
        values,
    ):


        before = float(
            before
        )

        after = float(
            after
        )


        difference = (
            after
            -
            before
        )


        changes[
            label
        ] = (
            difference
        )


        if difference > 0:

            css_class = (
                "positive"
            )

        elif difference < 0:

            css_class = (
                "negative"
            )

        else:

            css_class = (
                "neutral"
            )


        if (
            decimal_places
            ==
            0
        ):

            before_text = (
                f"{before:.0f}"
            )

            after_text = (
                f"{after:.0f}"
            )

            difference_text = (
                f"{difference:+.0f}"
            )

        else:

            before_text = (
                f"{before:.1f}"
            )

            after_text = (
                f"{after:.1f}"
            )

            difference_text = (
                f"{difference:+.1f}"
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
                                {before_text}
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
                                {after_text}
                            </strong>

                        </div>

                    </div>


                    <div class="before-after-change {css_class}">
                        Cambio: {difference_text}
                    </div>

                </div>
                """
            )


    return changes


def build_change_reading(
    changes
):

    increases = [
        (
            key,
            value,
        )
        for key, value
        in changes.items()
        if value > 0
    ]


    decreases = [
        (
            key,
            value,
        )
        for key, value
        in changes.items()
        if value < 0
    ]


    sentences = []


    if increases:

        strongest = max(
            increases,
            key=
                lambda item:
                    item[1],
        )


        sentences.append(
            f"El mayor aumento se observó en "
            f"{strongest[0].lower()} "
            f"({strongest[1]:+.1f})."
        )


    if decreases:

        strongest = min(
            decreases,
            key=
                lambda item:
                    item[1],
        )


        sentences.append(
            f"La mayor disminución se observó en "
            f"{strongest[0].lower()} "
            f"({strongest[1]:+.1f})."
        )


    if not sentences:

        return (
            "Las métricas analizadas se mantuvieron "
            "sin cambios."
        )


    return " ".join(
        sentences
    )


def aggregate_set_pieces(
    data
):

    if data.empty:

        return (
            pd.DataFrame()
        )


    data = (
        data.copy()
    )


    for column in [
        "executions",
        "shots",
        "goals",
    ]:

        if (
            column
            not in
            data.columns
        ):

            data[
                column
            ] = 0


        data[
            column
        ] = (
            pd.to_numeric(
                data[
                    column
                ],
                errors="coerce",
            )
            .fillna(0)
        )


    if (
        "set_piece"
        not in
        data.columns
    ):

        return (
            pd.DataFrame()
        )


    result = (
        data
        .groupby(
            "set_piece",
            as_index=False,
        )
        .agg(
            executions=(
                "executions",
                "sum",
            ),

            shots=(
                "shots",
                "sum",
            ),

            goals=(
                "goals",
                "sum",
            ),
        )
    )


    result[
        "shot_rate"
    ] = np.where(
        result[
            "executions"
        ]
        >
        0,

        result[
            "shots"
        ]
        /
        result[
            "executions"
        ],

        0,
    )


    return result


# ============================================================
# CARGA
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


complementary_substitutions = load_csv(
    COMPLEMENTARY_SUBSTITUTIONS_PATH
)

complementary_set_pieces = load_csv(
    COMPLEMENTARY_SET_PIECES_PATH
)


# ============================================================
# METADATA
# ============================================================

male_metadata = (
    build_match_metadata(
        male_match_history,
        "América",
    )
)


female_metadata = (
    build_match_metadata(
        female_match_history,
        "América W",
    )
)


# ============================================================
# MODELOS
# ============================================================

male_model = (
    load_model(
        MALE_MODEL_PATH
    )
    if
    MALE_MODEL_PATH.exists()
    else
    None
)


female_model = (
    load_model(
        FEMALE_MODEL_PATH
    )
    if
    FEMALE_MODEL_PATH.exists()
    else
    None
)


male_feature_names = (
    list(
        male_model.feature_names
    )
    if
    (
        male_model
        is not None
        and
        male_model.feature_names
    )
    else
    []
)


female_feature_names = (
    list(
        female_model.feature_names
    )
    if
    (
        female_model
        is not None
        and
        female_model.feature_names
    )
    else
    []
)


# ============================================================
# MEDIANAS FEMENIL
# ============================================================

female_medians = {}


if (
    not female_features_table.empty
    and
    {
        "feature",
        "train_median",
    }.issubset(
        female_features_table.columns
    )
):


    for _, row in (
        female_features_table
        .iterrows()
    ):


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
# UMBRAL FEMENIL
# ============================================================

female_threshold = np.nan


if (
    not female_selection.empty
    and
    "selected_threshold"
    in female_selection.columns
):

    female_threshold = (
        pd.to_numeric(
            female_selection[
                "selected_threshold"
            ],
            errors="coerce",
        )
        .iloc[0]
    )


if not np.isfinite(
    female_threshold
):

    female_threshold = (
        metric_from_table(
            female_metrics,
            "selected_threshold",
        )
    )


male_evidence = (
    calculate_male_metrics(
        male_coach
    )
)


# ============================================================
# SESSION STATE
# ============================================================

if (
    "section"
    not in
    st.session_state
):

    st.session_state[
        "section"
    ] = (
        "Varonil"
    )


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

if (
    st.session_state[
        "section"
    ]
    ==
    "Varonil"
):


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


        venue_display = (
            st.selectbox(
                "Condición",
                [
                    "Local",
                    "Visitante",
                ],
                key=
                    "male_venue",
            )
        )


        venue = (
            "HOME"
            if
            venue_display
            ==
            "Local"
            else
            "AWAY"
        )


        opponent = (
            st.selectbox(
                "Rival",
                opponents,
                key=
                    "male_opponent",
            )
        )


        minute = st.slider(
            "Minuto",
            1,
            90,
            60,
            key=
                "male_minute",
        )


        st.markdown(
            "**Marcador**"
        )


        (
            score_left,
            score_right,
        ) = st.columns(
            2,
            gap="small",
        )


        with score_left:

            america_goals = (
                score_stepper(
                    "América",
                    "male_goals",
                )
            )


        with score_right:

            opponent_goals = (
                score_stepper(
                    opponent,
                    "male_opponent_goals",
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


# ============================================================
# SIDEBAR FEMENIL
# ============================================================

elif (
    st.session_state[
        "section"
    ]
    ==
    "Femenil"
):


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


        venue_display = (
            st.selectbox(
                "Condición",
                [
                    "Local",
                    "Visitante",
                ],
                key=
                    "female_venue",
            )
        )


        venue = (
            "HOME"
            if
            venue_display
            ==
            "Local"
            else
            "AWAY"
        )


        opponent = (
            st.selectbox(
                "Rival",
                opponents,
                key=
                    "female_opponent",
            )
        )


        minute = st.slider(
            "Minuto",
            1,
            90,
            60,
            key=
                "female_minute",
        )


        st.markdown(
            "**Marcador**"
        )


        (
            score_left,
            score_right,
        ) = st.columns(
            2,
            gap="small",
        )


        with score_left:

            america_goals = (
                score_stepper(
                    "América W",
                    "female_goals",
                )
            )


        with score_right:

            opponent_goals = (
                score_stepper(
                    opponent,
                    "female_opponent_goals",
                )
            )


        visible_state = (
            st.selectbox(
                "Estado táctico actual",
                list(
                    STATE_DISPLAY.values()
                ),
                key=
                    "female_state",
            )
        )


        current_state = (
            STATE_REVERSE[
                visible_state
            ]
        )


# ============================================================
# HERO
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
# NAVEGACIÓN
# ============================================================

nav_options = [
    "Varonil",
    "Femenil",
    "Revisión histórica",
    "Análisis complementario",
    "Modelo",
]


selected_section = (
    st.radio(
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
    ] = (
        selected_section
    )

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
        male_model
        is None
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
            ) = (
                build_context_from_history(
                    male_simulator,
                    opponent,
                    venue,
                    minute,
                    score_state,
                    current_state,
                    male_feature_names,
                )
            )


            feature_row = (
                feature_row[
                    male_feature_names
                ]
            )


            matrix = xgb.DMatrix(
                feature_row.to_numpy(
                    dtype=float
                ),
                feature_names=
                    male_feature_names,
            )


            probabilities = np.asarray(
                male_model.predict(
                    matrix
                )
            )


            if (
                probabilities.ndim
                >
                1
            ):

                probabilities = (
                    probabilities[0]
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


            (
                c1,
                c2,
                c3,
            ) = st.columns(
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


            (
                left,
                right,
            ) = st.columns(
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
                            frente a contextos históricos.
                        </div>

                        <div class="similarity-history">
                            Partidos frente a
                            {opponent}: {historical_matches}
                        </div>

                    </div>
                    """
                )


            st.write("")


            st.markdown(
                "### Simulación táctica de escenarios"
            )


            simulated = (
                simulate_male_scenarios(
                    probabilities,
                    STATE_LEVELS,
                )
            )


            dominant = (
                simulated
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
                            {state_display(dominant["state"])}
                        </strong>.

                        Eso representa

                        <strong>
                            {safe_percent(dominant["share"])}
                        </strong>.

                    </div>

                    <div class="simulation-intro-note">
                        No son 1,000 partidos completos.
                        Son posibles desenlaces tácticos
                        para el mismo contexto.
                    </div>

                </div>
                """
            )


            (
                sim_left,
                sim_right,
            ) = st.columns(
                [
                    1.7,
                    1,
                ],
                gap="large",
            )


            with sim_left:


                plot_data = (
                    simulated
                    .sort_values(
                        "percentage",
                        ascending=True,
                    )
                )


                fig = px.bar(
                    plot_data,
                    x=
                        "percentage",
                    y=
                        "label",
                    orientation=
                        "h",
                    text=
                        "percentage",
                )


                fig.update_traces(
                    marker_color=
                        COACH_YELLOW,

                    texttemplate=
                        "%{text:.1f}%",

                    textposition=
                        "outside",

                    cliponaxis=
                        False,
                )


                fig = (
                    apply_coachlens_chart_style(
                        fig,
                        height=
                            320,
                        x_title=
                            "Porcentaje de escenarios",
                        y_title=
                            "",
                    )
                )


                maximum = float(
                    plot_data[
                        "percentage"
                    ].max()
                )


                fig.update_xaxes(
                    ticksuffix=
                        "%",
                    range=[
                        0,
                        max(
                            40,
                            maximum
                            +
                            8,
                        ),
                    ],
                )


                st.plotly_chart(
                    fig,
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
                            {state_display(dominant["state"])}
                        </div>

                        <div class="simulation-summary-share">
                            {safe_percent(dominant["share"])}
                        </div>

                        <div class="simulation-summary-text">
                            Estado que apareció con mayor
                            frecuencia entre los escenarios.
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
        female_model
        is None
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
            ) = (
                build_female_context_from_history(
                    female_simulator,
                    opponent,
                    venue,
                    minute,
                    score_state,
                    current_state,
                    female_feature_names,
                    female_medians,
                )
            )


            feature_row = (
                feature_row[
                    female_feature_names
                ]
            )


            matrix = xgb.DMatrix(
                feature_row.to_numpy(
                    dtype=float
                ),
                feature_names=
                    female_feature_names,
            )


            probability_change = float(
                np.asarray(
                    female_model.predict(
                        matrix
                    )
                )
                .reshape(-1)[0]
            )


            predicted_change = (
                probability_change
                >=
                female_threshold
            )


            signal = (
                "CAMBIO PROBABLE"
                if
                predicted_change
                else
                "ESTABILIDAD PROBABLE"
            )


            transition = (
                calculate_historical_transition(
                    neighbors,
                    current_state,
                )
            )


            transition_state = (
                transition[
                    "state"
                ]
            )


            transition_share = (
                transition[
                    "share"
                ]
            )


            transition_n = (
                transition[
                    "observations"
                ]
            )


            if transition_state:

                transition_title = (
                    state_display(
                        transition_state
                    )
                )

            else:

                transition_title = (
                    "SIN PATRÓN"
                )


            if predicted_change:


                if transition_state:


                    action = (
                        get_coach_action(
                            transition_state
                        )
                    )


                    action_explanation = (
                        f"El modelo detecta cambio probable. "
                        f"Entre los contextos históricos "
                        f"comparables que cambiaron, "
                        f"{state_display(transition_state)} "
                        f"fue el destino más frecuente."
                    )


                else:


                    action = {
                        "title":
                            "Monitorear el cambio",

                        "text":
                            (
                                "Existe riesgo de cambio, "
                                "pero los contextos históricos "
                                "comparables no muestran un "
                                "destino dominante."
                            ),
                    }


                    action_explanation = (
                        "No existe suficiente concentración "
                        "histórica en un único estado posterior."
                    )


            else:


                action = {
                    "title":
                        "Mantener la estructura",

                    "text":
                        (
                            f"La probabilidad de cambio "
                            f"permanece por debajo del umbral. "
                            f"Conviene mantener y monitorear "
                            f"el estado "
                            f"{state_display(current_state)}."
                        ),
                }


                if transition_state:

                    action_explanation = (
                        f"Si finalmente ocurriera un cambio, "
                        f"{state_display(transition_state)} "
                        f"sería el destino histórico más "
                        f"frecuente entre los contextos "
                        f"comparables."
                    )

                else:

                    action_explanation = (
                        "No existe un destino histórico "
                        "dominante entre los contextos "
                        "comparables."
                    )


            (
                f1,
                f2,
                f3,
            ) = st.columns(
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
                            Horizonte: próximos cinco minutos
                        </div>

                    </div>
                    """
                )


            with f2:

                st.html(
                    f"""
                    <div class="result-card">

                        <div class="result-card-label">
                            Si ocurre un cambio
                        </div>

                        <div class="result-card-value">
                            {transition_title}
                        </div>

                        <div class="result-card-text">

                            Participación histórica:
                            {safe_percent(transition_share)}

                            <br>

                            Contextos utilizados:
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

                        <div class="action-context">
                            {action_explanation}
                        </div>

                    </div>
                    """
                )


            st.write("")


            (
                left,
                right,
            ) = st.columns(
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
                            {signal}
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

                            Distancia:

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
                            Contexto histórico
                        </div>

                        <div class="similarity-description">
                            El modelo estima cambio o estabilidad.
                            El posible destino se obtiene de los
                            contextos históricos más comparables.
                        </div>

                        <div class="similarity-history">
                            Partidos frente a
                            {opponent}: {historical_matches}
                        </div>

                    </div>
                    """
                )


            st.write("")


            st.markdown(
                "### Simulación táctica de escenarios"
            )


            female_sim = (
                simulate_female_scenarios(
                    probability_change
                )
            )


            dominant = (
                female_sim
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

                        El modelo estima una probabilidad
                        de cambio de

                        <strong>
                            {safe_percent(probability_change)}
                        </strong>.

                        Al repetir el escenario
                        <strong>1,000 veces</strong>,

                        <strong>
                            {int(dominant["simulations"])}
                        </strong>

                        repeticiones terminaron en

                        <strong class="simulation-highlight">
                            {dominant["outcome"]}
                        </strong>.

                    </div>

                    <div class="simulation-intro-note">
                        La pequeña diferencia entre la
                        probabilidad del modelo y la proporción
                        simulada se debe a la variabilidad
                        aleatoria de las repeticiones.
                    </div>

                </div>
                """
            )


            (
                sim_left,
                sim_right,
            ) = st.columns(
                [
                    1.7,
                    1,
                ],
                gap="large",
            )


            with sim_left:


                plot_data = (
                    female_sim
                    .sort_values(
                        "percentage",
                        ascending=True,
                    )
                )


                fig = px.bar(
                    plot_data,
                    x=
                        "percentage",
                    y=
                        "outcome",
                    orientation=
                        "h",
                    text=
                        "percentage",
                )


                fig.update_traces(
                    marker_color=
                        COACH_YELLOW,

                    texttemplate=
                        "%{text:.1f}%",

                    textposition=
                        "outside",

                    cliponaxis=
                        False,
                )


                fig = (
                    apply_coachlens_chart_style(
                        fig,
                        height=
                            230,
                        x_title=
                            "Porcentaje de escenarios",
                        y_title=
                            "",
                    )
                )


                fig.update_xaxes(
                    ticksuffix=
                        "%",
                    range=[
                        0,
                        100,
                    ],
                )


                st.plotly_chart(
                    fig,
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


        except Exception as error:

            st.error(
                f"No se pudo generar el análisis: {error}"
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
            VALIDACIÓN HISTÓRICA
        </div>

        <div class="page-title">
            Revisión de partidos TEST
        </div>

        <div class="page-description">
            Explora predicciones realizadas sobre partidos
            reservados para evaluación y no utilizados durante
            el entrenamiento.
        </div>
        """
    )


    review_team = (
        st.radio(
            "Equipo",
            [
                "Varonil",
                "Femenil",
            ],
            horizontal=True,
            key=
                "review_team",
        )
    )


    # ========================================================
    # VARONIL TEST
    # ========================================================

    if (
        review_team
        ==
        "Varonil"
    ):


        history = (
            male_coach
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

            history = (
                history[
                    history[
                        "_match_id"
                    ]
                    .isin(
                        played_ids
                    )
                ]
                .copy()
            )


        matches = (
            history[
                [
                    "_match_id",
                    "match_date",
                    "opponent",
                ]
            ]
            .drop_duplicates()
            .dropna(
                subset=[
                    "opponent",
                ]
            )
            .sort_values(
                "match_date",
                ascending=False,
            )
            .copy()
        )


        if matches.empty:


            st.warning(
                "No existen partidos TEST disponibles."
            )


        else:


            rivals = sorted(
                matches[
                    "opponent"
                ]
                .dropna()
                .astype(str)
                .unique()
                .tolist()
            )


            (
                rival_col,
                match_col,
            ) = st.columns(
                [
                    0.8,
                    2.2,
                ],
                gap="medium",
            )


            with rival_col:


                selected_rival = (
                    st.selectbox(
                        "Rival",
                        rivals,
                        key=
                            "history_male_opponent",
                    )
                )


            filtered_matches = (
                matches[
                    matches[
                        "opponent"
                    ]
                    .astype(str)
                    ==
                    selected_rival
                ]
                .copy()
            )


            filtered_matches[
                "label"
            ] = (
                filtered_matches[
                    "match_date"
                ]
                .dt.strftime(
                    "%Y-%m-%d"
                )
                +
                " · América vs "
                +
                filtered_matches[
                    "opponent"
                ]
                .astype(str)
            )


            with match_col:


                selected_match = (
                    st.selectbox(
                        "Partido",
                        filtered_matches[
                            "label"
                        ].tolist(),
                        key=
                            "history_male_match",
                    )
                )


            match_id = (
                filtered_matches.loc[
                    filtered_matches[
                        "label"
                    ]
                    ==
                    selected_match,
                    "_match_id",
                ]
                .iloc[0]
            )


            match_data = (
                history[
                    history[
                        "_match_id"
                    ]
                    ==
                    match_id
                ]
                .copy()
            )


            match_data[
                "minute"
            ] = pd.to_numeric(
                match_data[
                    "minute"
                ],
                errors="coerce",
            )


            match_data = (
                match_data
                .dropna(
                    subset=[
                        "minute",
                    ]
                )
                .sort_values(
                    "minute"
                )
                .reset_index(
                    drop=True
                )
            )


            selected_minute = (
                st.slider(
                    "Minuto",
                    0,
                    90,
                    45,
                    1,
                    key=
                        "history_male_minute",
                )
            )


            nearest_index = (
                (
                    match_data[
                        "minute"
                    ]
                    -
                    selected_minute
                )
                .abs()
                .idxmin()
            )


            row = (
                match_data.loc[
                    nearest_index
                ]
            )


            real_minute = int(
                row[
                    "minute"
                ]
            )


            if (
                real_minute
                !=
                selected_minute
            ):

                st.caption(
                    f"No existe observación TEST exactamente "
                    f"en el minuto {selected_minute}; "
                    f"se muestra la más cercana: "
                    f"minuto {real_minute}."
                )


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


            columns = (
                st.columns(
                    4,
                    gap="medium",
                )
            )


            cards = [

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


            for (
                column,
                (
                    label,
                    value,
                ),
            ) in zip(
                columns,
                cards,
            ):


                with column:


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
                                El estado observado coincide con
                                la primera predicción.
                            </div>

                        </div>


                        <div class="history-explanation-item">

                            <div class="history-explanation-badge top2">
                                ACIERTO TOP-2
                            </div>

                            <div class="history-explanation-text">
                                El estado real aparece como
                                segunda alternativa.
                            </div>

                        </div>


                        <div class="history-explanation-item">

                            <div class="history-explanation-badge miss">
                                NO ACIERTO
                            </div>

                            <div class="history-explanation-text">
                                El estado real no apareció
                                entre las dos primeras opciones.
                            </div>

                        </div>

                    </div>


                    <div class="history-test-summary">

                        TEST · Top-1:

                        <strong>
                            {safe_percent(male_evidence["top1"])}
                        </strong>

                        · Top-2:

                        <strong>
                            {safe_percent(male_evidence["top2"])}
                        </strong>

                    </div>

                </div>
                """
            )


    # ========================================================
    # FEMENIL TEST
    # ========================================================

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


        matches = (
            history[
                [
                    "_match_id",
                    "match_date",
                    "opponent",
                ]
            ]
            .drop_duplicates()
            .dropna(
                subset=[
                    "opponent",
                ]
            )
            .sort_values(
                "match_date",
                ascending=False,
            )
            .copy()
        )


        if matches.empty:


            st.warning(
                "No existen partidos TEST disponibles."
            )


        else:


            rivals = sorted(
                matches[
                    "opponent"
                ]
                .dropna()
                .astype(str)
                .unique()
                .tolist()
            )


            (
                rival_col,
                match_col,
            ) = st.columns(
                [
                    0.8,
                    2.2,
                ],
                gap="medium",
            )


            with rival_col:


                selected_rival = (
                    st.selectbox(
                        "Rival",
                        rivals,
                        key=
                            "history_female_opponent",
                    )
                )


            filtered_matches = (
                matches[
                    matches[
                        "opponent"
                    ]
                    .astype(str)
                    ==
                    selected_rival
                ]
                .copy()
            )


            filtered_matches[
                "label"
            ] = (
                filtered_matches[
                    "match_date"
                ]
                .dt.strftime(
                    "%Y-%m-%d"
                )
                +
                " · América W vs "
                +
                filtered_matches[
                    "opponent"
                ]
                .astype(str)
            )


            with match_col:


                selected_match = (
                    st.selectbox(
                        "Partido",
                        filtered_matches[
                            "label"
                        ].tolist(),
                        key=
                            "history_female_match",
                    )
                )


            match_id = (
                filtered_matches.loc[
                    filtered_matches[
                        "label"
                    ]
                    ==
                    selected_match,
                    "_match_id",
                ]
                .iloc[0]
            )


            match_data = (
                history[
                    history[
                        "_match_id"
                    ]
                    ==
                    match_id
                ]
                .copy()
            )


            match_data[
                "minute"
            ] = pd.to_numeric(
                match_data[
                    "minute"
                ],
                errors="coerce",
            )


            match_data = (
                match_data
                .dropna(
                    subset=[
                        "minute",
                    ]
                )
                .sort_values(
                    "minute"
                )
                .reset_index(
                    drop=True
                )
            )


            selected_minute = (
                st.slider(
                    "Minuto",
                    0,
                    90,
                    45,
                    1,
                    key=
                        "history_female_minute",
                )
            )


            nearest_index = (
                (
                    match_data[
                        "minute"
                    ]
                    -
                    selected_minute
                )
                .abs()
                .idxmin()
            )


            row = (
                match_data.loc[
                    nearest_index
                ]
            )


            real_minute = int(
                row[
                    "minute"
                ]
            )


            if (
                real_minute
                !=
                selected_minute
            ):

                st.caption(
                    f"No existe observación TEST exactamente "
                    f"en el minuto {selected_minute}; "
                    f"se muestra la más cercana: "
                    f"minuto {real_minute}."
                )


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


            columns = (
                st.columns(
                    4,
                    gap="medium",
                )
            )


            cards = [

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


            for (
                column,
                (
                    label,
                    value,
                ),
            ) in zip(
                columns,
                cards,
            ):


                with column:


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
                                La clasificación coincide con
                                el resultado observado.
                            </div>

                        </div>


                        <div class="history-explanation-item">

                            <div class="history-explanation-badge miss">
                                NO ACIERTO
                            </div>

                            <div class="history-explanation-text">
                                La clasificación no coincide
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
                                </strong>

                            </div>

                        </div>

                    </div>


                    <div class="history-test-summary">

                        Predicción:

                        <strong>
                            {predicted_change}
                        </strong>

                        · Resultado real:

                        <strong>
                            {actual_change}
                        </strong>

                    </div>

                </div>
                """
            )


# ============================================================
# ANÁLISIS COMPLEMENTARIO
# ============================================================

elif (
    st.session_state[
        "section"
    ]
    ==
    "Análisis complementario"
):


    st.html(
        """
        <div class="page-kicker">
            COMPONENTES COMPLEMENTARIOS
        </div>

        <div class="page-title">
            Identidad, jugadores y balón parado
        </div>

        <div class="page-description">
            Explora identidad de juego, uso de jugadores,
            sustituciones y situaciones a balón parado.
        </div>
        """
    )


    (
        team_col,
        view_col,
    ) = st.columns(
        2,
        gap="medium",
    )


    with team_col:

        complementary_team = (
            st.radio(
                "Equipo",
                [
                    "Varonil",
                    "Femenil",
                ],
                horizontal=True,
                key=
                    "complementary_team",
            )
        )


    with view_col:

        complementary_view = (
            st.radio(
                "Vista",
                [
                    "Global",
                    "Por partido",
                ],
                horizontal=True,
                key=
                    "complementary_view",
            )
        )


    complementary_section = (
        st.radio(
            "Análisis",
            [
                "Identidad de juego",
                "Jugadores y sustituciones",
                "Balón parado ofensivo",
                "Balón parado defensivo",
            ],
            horizontal=True,
            key=
                "complementary_section",
        )
    )


    if (
        complementary_team
        ==
        "Varonil"
    ):

        team_name = (
            "América"
        )

        metadata = (
            male_metadata
        )

    else:

        team_name = (
            "América W"
        )

        metadata = (
            female_metadata
        )


    # ========================================================
    # IDENTIDAD DE JUEGO
    # ========================================================

    if (
        complementary_section
        ==
        "Identidad de juego"
    ):


        st.markdown(
            "### Identidad de juego"
        )


        st.caption(
            "Resume construcción, progresión, generación "
            "de ocasiones y comportamiento defensivo."
        )


        if (
            complementary_team
            ==
            "Varonil"
        ):

            identity_data = (
                male_simulator
                .copy()
            )

        else:

            identity_data = (
                female_simulator
                .copy()
            )


        if (
            "match_id"
            in identity_data.columns
        ):

            identity_data = (
                attach_match_metadata(
                    identity_data,
                    metadata,
                )
            )


        filtered_identity = (
            identity_data
            .copy()
        )


        if (
            complementary_view
            ==
            "Por partido"
        ):


            if (
                "opponent"
                in filtered_identity.columns
            ):


                rivals = sorted(
                    filtered_identity[
                        "opponent"
                    ]
                    .dropna()
                    .astype(str)
                    .unique()
                    .tolist()
                )


                (
                    rival_col,
                    match_col,
                ) = st.columns(
                    [
                        0.9,
                        2.1,
                    ],
                    gap="medium",
                )


                with rival_col:


                    identity_rival = (
                        st.selectbox(
                            "Rival",
                            rivals,
                            key=
                                "identity_rival",
                        )
                    )


                filtered_identity = (
                    filtered_identity[
                        filtered_identity[
                            "opponent"
                        ]
                        .astype(str)
                        ==
                        identity_rival
                    ]
                    .copy()
                )


                if (
                    "_match_id"
                    in
                    filtered_identity.columns
                ):


                    matches = (
                        filtered_identity[
                            [
                                "_match_id",
                                "match_date",
                                "opponent",
                            ]
                        ]
                        .drop_duplicates()
                        .copy()
                    )


                    matches[
                        "label"
                    ] = (
                        matches.apply(
                            lambda row:
                                make_match_label(
                                    row[
                                        "match_date"
                                    ],
                                    row[
                                        "opponent"
                                    ],
                                    team_name,
                                ),
                            axis=1,
                        )
                    )


                    with match_col:


                        identity_match = (
                            st.selectbox(
                                "Partido",
                                matches[
                                    "label"
                                ].tolist(),
                                key=
                                    "identity_match",
                            )
                        )


                    identity_match_id = (
                        matches.loc[
                            matches[
                                "label"
                            ]
                            ==
                            identity_match,
                            "_match_id",
                        ]
                        .iloc[0]
                    )


                    filtered_identity = (
                        filtered_identity[
                            filtered_identity[
                                "_match_id"
                            ]
                            ==
                            identity_match_id
                        ]
                        .copy()
                    )


        selected_metrics = (
            build_play_identity_metrics(
                filtered_identity
            )
        )


        global_metrics = (
            build_play_identity_metrics(
                identity_data
            )
        )


        global_lookup = {
            metric[
                "name"
            ]:
                metric[
                    "value"
                ]

            for metric
            in global_metrics
        }


        st.html(
            """
            <div class="chart-help-box">

                <div class="chart-help-title">
                    ¿Cómo leer la identidad de juego?
                </div>

                <div class="chart-help-text">
                    Los indicadores muestran el comportamiento
                    del equipo frente a su propia referencia
                    histórica.
                </div>

                <div class="chart-help-note">
                    100% representa la referencia histórica
                    del equipo. No es una probabilidad ni
                    una calificación.
                </div>

            </div>
            """
        )


        if not selected_metrics:


            st.warning(
                "No existen suficientes variables disponibles "
                "para construir la identidad de juego."
            )


        else:


            offensive = [
                metric
                for metric
                in selected_metrics
                if
                metric[
                    "section"
                ]
                ==
                "Fase ofensiva"
            ]


            defensive = [
                metric
                for metric
                in selected_metrics
                if
                metric[
                    "section"
                ]
                ==
                "Fase defensiva"
            ]


            st.markdown(
                "#### Fase ofensiva"
            )


            if offensive:


                columns = st.columns(
                    len(
                        offensive
                    ),
                    gap="medium",
                )


                for (
                    column,
                    metric,
                ) in zip(
                    columns,
                    offensive,
                ):


                    baseline = (
                        global_lookup.get(
                            metric[
                                "name"
                            ],
                            np.nan,
                        )
                    )


                    index_value = (
                        relative_index(
                            metric[
                                "value"
                            ],
                            baseline,
                        )
                    )


                    with column:


                        st.html(
                            f"""
                            <div class="metric-card complementary-metric-card">

                                <div class="metric-label">
                                    {metric["name"]}
                                </div>

                                <div class="metric-value">
                                    {
                                        f"{index_value:.0f}%"
                                        if np.isfinite(index_value)
                                        else "N/D"
                                    }
                                </div>

                            </div>
                            """
                        )


                        st.caption(
                            identity_reading(
                                index_value
                            )
                        )


            st.markdown(
                "#### Fase defensiva"
            )


            if defensive:


                columns = st.columns(
                    len(
                        defensive
                    ),
                    gap="medium",
                )


                for (
                    column,
                    metric,
                ) in zip(
                    columns,
                    defensive,
                ):


                    baseline = (
                        global_lookup.get(
                            metric[
                                "name"
                            ],
                            np.nan,
                        )
                    )


                    index_value = (
                        relative_index(
                            metric[
                                "value"
                            ],
                            baseline,
                        )
                    )


                    with column:


                        st.html(
                            f"""
                            <div class="metric-card complementary-metric-card">

                                <div class="metric-label">
                                    {metric["name"]}
                                </div>

                                <div class="metric-value">
                                    {
                                        f"{index_value:.0f}%"
                                        if np.isfinite(index_value)
                                        else "N/D"
                                    }
                                </div>

                            </div>
                            """
                        )


                        st.caption(
                            identity_reading(
                                index_value
                            )
                        )


            chart_rows = []


            for metric in selected_metrics:


                baseline = (
                    global_lookup.get(
                        metric[
                            "name"
                        ],
                        np.nan,
                    )
                )


                index_value = (
                    relative_index(
                        metric[
                            "value"
                        ],
                        baseline,
                    )
                )


                if np.isfinite(
                    index_value
                ):

                    chart_rows.append(
                        {
                            "Indicador":
                                metric[
                                    "name"
                                ],

                            "Índice":
                                index_value,
                        }
                    )


            chart_data = pd.DataFrame(
                chart_rows
            )


            if not chart_data.empty:


                chart_data = (
                    chart_data
                    .sort_values(
                        "Índice",
                        ascending=True,
                    )
                )


                figure = px.bar(
                    chart_data,
                    x=
                        "Índice",
                    y=
                        "Indicador",
                    orientation=
                        "h",
                    text=
                        "Índice",
                )


                figure.update_traces(
                    marker_color=
                        COACH_YELLOW,

                    texttemplate=
                        "%{text:.0f}%",

                    textposition=
                        "outside",

                    cliponaxis=
                        False,
                )


                maximum = float(
                    chart_data[
                        "Índice"
                    ].max()
                )


                figure = (
                    apply_coachlens_chart_style(
                        figure,
                        height=
                            max(
                                300,
                                110
                                +
                                (
                                    len(
                                        chart_data
                                    )
                                    *
                                    45
                                ),
                            ),
                        x_title=
                            "Índice relativo · 100% = referencia histórica",
                        y_title=
                            "",
                    )
                )


                figure.update_xaxes(
                    ticksuffix=
                        "%",
                    range=[
                        0,
                        max(
                            130,
                            maximum
                            +
                            12,
                        ),
                    ],
                )


                figure.add_vline(
                    x=
                        100,
                    line_dash=
                        "dash",
                    line_width=
                        1,
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
    # JUGADORES Y SUSTITUCIONES
    # ========================================================

    elif (
        complementary_section
        ==
        "Jugadores y sustituciones"
    ):


        st.markdown(
            "### Uso de jugadores y sustituciones"
        )


        substitutions = (
            complementary_substitutions[
                complementary_substitutions[
                    "squad"
                ]
                ==
                complementary_team
            ]
            .copy()
        )


        substitutions = (
            attach_match_metadata(
                substitutions,
                metadata,
            )
        )


        numeric_columns = [
            "minute",
            "shots_before",
            "shots_after",
            "pressures_before",
            "pressures_after",
            "recoveries_before",
            "recoveries_after",
            "passes_before",
            "passes_after",
        ]


        for column in numeric_columns:

            if (
                column
                in substitutions.columns
            ):

                substitutions[
                    column
                ] = pd.to_numeric(
                    substitutions[
                        column
                    ],
                    errors=
                        "coerce",
                )


        if substitutions.empty:


            st.warning(
                "No existen sustituciones disponibles."
            )


        # ====================================================
        # GLOBAL
        # ====================================================

        elif (
            complementary_view
            ==
            "Global"
        ):


            total_substitutions = int(
                len(
                    substitutions
                )
            )


            usage = (
                build_player_usage_summary(
                    substitutions
                )
            )


            st.html(
                f"""
                <div class="global-context-card">

                    <div class="global-context-label">
                        SUSTITUCIONES ANALIZADAS
                    </div>

                    <div class="global-context-value">
                        {total_substitutions:,}
                    </div>

                    <div class="global-context-text">
                        Registros disponibles para
                        {complementary_team}.
                    </div>

                </div>
                """
            )


            usage_cards = []


            if (
                "most_subbed_in"
                in usage
            ):

                usage_cards.append(
                    (
                        "Jugador que más entra",
                        usage[
                            "most_subbed_in"
                        ],
                    )
                )


            if (
                "most_subbed_out"
                in usage
            ):

                usage_cards.append(
                    (
                        "Jugador que más sale",
                        usage[
                            "most_subbed_out"
                        ],
                    )
                )


            if (
                "median_minute"
                in usage
            ):

                usage_cards.append(
                    (
                        "Minuto típico del cambio",
                        str(
                            usage[
                                "median_minute"
                            ]
                        ),
                    )
                )


            if usage_cards:


                columns = st.columns(
                    len(
                        usage_cards
                    ),
                    gap="medium",
                )


                for (
                    column,
                    (
                        label,
                        value,
                    ),
                ) in zip(
                    columns,
                    usage_cards,
                ):


                    with column:


                        st.html(
                            f"""
                            <div class="metric-card complementary-metric-card">

                                <div class="metric-label">
                                    {label}
                                </div>

                                <div class="history-state-value">
                                    {value}
                                </div>

                            </div>
                            """
                        )


            if (
                "common_pair"
                in usage
            ):


                st.html(
                    f"""
                    <div class="complementary-reading">

                        <div class="complementary-reading-title">
                            Combinación de cambio más frecuente
                        </div>

                        <div class="complementary-reading-text">
                            {usage["common_pair"]}
                        </div>

                    </div>
                    """
                )


            st.html(
                """
                <div class="chart-help-box">

                    <div class="chart-help-title">
                        ¿Cómo leer esta vista?
                    </div>

                    <div class="chart-help-text">
                        Compara los diez minutos anteriores
                        y posteriores a las sustituciones.
                    </div>

                    <div class="chart-help-note">
                        Es una comparación descriptiva.
                        No demuestra causalidad.
                    </div>

                </div>
                """
            )


            global_values = [

                (
                    "Tiros",
                    substitutions[
                        "shots_before"
                    ].mean(),
                    substitutions[
                        "shots_after"
                    ].mean(),
                ),

                (
                    "Presiones",
                    substitutions[
                        "pressures_before"
                    ].mean(),
                    substitutions[
                        "pressures_after"
                    ].mean(),
                ),

                (
                    "Recuperaciones",
                    substitutions[
                        "recoveries_before"
                    ].mean(),
                    substitutions[
                        "recoveries_after"
                    ].mean(),
                ),

                (
                    "Pases",
                    substitutions[
                        "passes_before"
                    ].mean(),
                    substitutions[
                        "passes_after"
                    ].mean(),
                ),
            ]


            changes = (
                render_before_after(
                    global_values,
                    decimal_places=
                        1,
                )
            )


            reading = (
                build_change_reading(
                    changes
                )
            )


            st.html(
                f"""
                <div class="complementary-reading">

                    <div class="complementary-reading-title">
                        Lectura global
                    </div>

                    <div class="complementary-reading-text">
                        {reading}
                    </div>

                </div>
                """
            )


            plot_data = pd.DataFrame(
                {
                    "Métrica": [
                        x[0]
                        for x
                        in global_values
                    ],

                    "Antes": [
                        x[1]
                        for x
                        in global_values
                    ],

                    "Después": [
                        x[2]
                        for x
                        in global_values
                    ],
                }
            )


            long_data = (
                plot_data
                .melt(
                    id_vars=
                        "Métrica",
                    var_name=
                        "Ventana",
                    value_name=
                        "Promedio",
                )
            )


            figure = px.bar(
                long_data,
                x=
                    "Métrica",
                y=
                    "Promedio",
                color=
                    "Ventana",
                barmode=
                    "group",
                text=
                    "Promedio",

                color_discrete_map={
                    "Antes":
                        BEFORE_COLOR,

                    "Después":
                        AFTER_COLOR,
                },
            )


            figure.update_traces(
                texttemplate=
                    "%{text:.1f}",

                textposition=
                    "outside",

                cliponaxis=
                    False,
            )


            figure = (
                apply_coachlens_chart_style(
                    figure,
                    height=
                        350,
                    legend=
                        True,
                    x_title=
                        "",
                    y_title=
                        "Promedio de eventos por sustitución",
                )
            )


            st.plotly_chart(
                figure,
                use_container_width=True,
                config={
                    "displayModeBar":
                        False
                },
            )


        # ====================================================
        # POR PARTIDO
        # ====================================================

        else:


            st.caption(
                "Selecciona rival, partido y después una "
                "sustitución para revisar el detalle."
            )


            rivals = sorted(
                substitutions[
                    "opponent"
                ]
                .dropna()
                .astype(str)
                .unique()
                .tolist()
            )


            (
                rival_col,
                match_col,
            ) = st.columns(
                [
                    0.9,
                    2.1,
                ],
                gap="medium",
            )


            with rival_col:


                selected_rival = (
                    st.selectbox(
                        "Rival",
                        rivals,
                        key=
                            "substitution_rival",
                    )
                )


            rival_data = (
                substitutions[
                    substitutions[
                        "opponent"
                    ]
                    ==
                    selected_rival
                ]
                .copy()
            )


            match_options = (
                rival_data[
                    [
                        "_match_id",
                        "match_date",
                        "opponent",
                    ]
                ]
                .drop_duplicates()
                .copy()
            )


            match_options[
                "label"
            ] = (
                match_options.apply(
                    lambda row:
                        make_match_label(
                            row[
                                "match_date"
                            ],
                            row[
                                "opponent"
                            ],
                            team_name,
                        ),
                    axis=1,
                )
            )


            with match_col:


                selected_match = (
                    st.selectbox(
                        "Partido",
                        match_options[
                            "label"
                        ].tolist(),
                        key=
                            "substitution_match",
                    )
                )


            match_id = (
                match_options.loc[
                    match_options[
                        "label"
                    ]
                    ==
                    selected_match,
                    "_match_id",
                ]
                .iloc[0]
            )


            match_substitutions = (
                rival_data[
                    rival_data[
                        "_match_id"
                    ]
                    ==
                    match_id
                ]
                .copy()
            )


            # =================================================
            # RESUMEN DEL PARTIDO
            # =================================================

            match_usage = (
                build_player_usage_summary(
                    match_substitutions
                )
            )


            total_match_substitutions = int(
                len(
                    match_substitutions
                )
            )


            st.markdown(
                "#### Resumen de sustituciones del partido"
            )


            summary_cards = [
                (
                    "Sustituciones",
                    str(
                        total_match_substitutions
                    ),
                ),
            ]


            if (
                "most_subbed_in"
                in match_usage
            ):

                summary_cards.append(
                    (
                        "Jugador que entra",
                        match_usage[
                            "most_subbed_in"
                        ],
                    )
                )


            if (
                "most_subbed_out"
                in match_usage
            ):

                summary_cards.append(
                    (
                        "Jugador que sale",
                        match_usage[
                            "most_subbed_out"
                        ],
                    )
                )


            if (
                "median_minute"
                in match_usage
            ):

                summary_cards.append(
                    (
                        "Minuto típico",
                        str(
                            match_usage[
                                "median_minute"
                            ]
                        ),
                    )
                )


            summary_columns = (
                st.columns(
                    len(
                        summary_cards
                    ),
                    gap="medium",
                )
            )


            for (
                column,
                (
                    label,
                    value,
                ),
            ) in zip(
                summary_columns,
                summary_cards,
            ):


                with column:


                    st.html(
                        f"""
                        <div class="metric-card complementary-metric-card">

                            <div class="metric-label">
                                {label}
                            </div>

                            <div class="history-state-value">
                                {value}
                            </div>

                        </div>
                        """
                    )


            if (
                "common_pair"
                in match_usage
            ):


                pair_count = (
                    match_usage.get(
                        "common_pair_count",
                        1,
                    )
                )


                if (
                    pair_count
                    >
                    1
                ):

                    pair_description = (
                        f"Esta combinación apareció "
                        f"{pair_count} veces en el partido."
                    )

                else:

                    pair_description = (
                        "En este partido no se repitió "
                        "ninguna combinación de cambio; "
                        "se muestra una de las sustituciones "
                        "registradas."
                    )


                st.html(
                    f"""
                    <div class="complementary-reading">

                        <div class="complementary-reading-title">
                            Combinación de cambio más frecuente
                        </div>

                        <div class="complementary-reading-text">
                            {match_usage["common_pair"]}
                        </div>

                        <div class="complementary-reading-note">
                            {pair_description}
                        </div>

                    </div>
                    """
                )


            # =================================================
            # SELECTOR DE SUSTITUCIÓN
            # =================================================

            st.markdown(
                "#### Detalle de una sustitución"
            )


            match_substitutions[
                "selector"
            ] = (
                "Min "
                +
                match_substitutions[
                    "minute"
                ]
                .fillna(0)
                .astype(int)
                .astype(str)
                +
                " · "
                +
                match_substitutions[
                    "player_out"
                ]
                .fillna(
                    "N/D"
                )
                .astype(str)
                +
                " → "
                +
                match_substitutions[
                    "player_in"
                ]
                .fillna(
                    "N/D"
                )
                .astype(str)
            )


            selected_substitution = (
                st.selectbox(
                    "Sustitución",
                    match_substitutions[
                        "selector"
                    ].tolist(),
                    key=
                        "substitution_selection",
                )
            )


            row = (
                match_substitutions[
                    match_substitutions[
                        "selector"
                    ]
                    ==
                    selected_substitution
                ]
                .iloc[0]
            )


            st.html(
                f"""
                <div class="substitution-hero">

                    <div class="substitution-minute">
                        MINUTO {int(row["minute"])}
                    </div>

                    <div class="substitution-player-grid">

                        <div class="substitution-player">

                            <div class="substitution-label">
                                SALE
                            </div>

                            <div class="substitution-name">
                                {row["player_out"]}
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
                                {row["player_in"]}
                            </div>

                        </div>

                    </div>

                </div>
                """
            )


            values = [

                (
                    "Tiros",
                    row[
                        "shots_before"
                    ],
                    row[
                        "shots_after"
                    ],
                ),

                (
                    "Presiones",
                    row[
                        "pressures_before"
                    ],
                    row[
                        "pressures_after"
                    ],
                ),

                (
                    "Recuperaciones",
                    row[
                        "recoveries_before"
                    ],
                    row[
                        "recoveries_after"
                    ],
                ),

                (
                    "Pases",
                    row[
                        "passes_before"
                    ],
                    row[
                        "passes_after"
                    ],
                ),
            ]


            changes = (
                render_before_after(
                    values,
                    decimal_places=
                        0,
                )
            )


            reading = (
                build_change_reading(
                    changes
                )
            )


            st.html(
                f"""
                <div class="complementary-reading">

                    <div class="complementary-reading-title">
                        Lectura de la sustitución
                    </div>

                    <div class="complementary-reading-text">
                        {reading}
                    </div>

                    <div class="complementary-reading-note">
                        Comparación descriptiva de los diez
                        minutos anteriores y posteriores.
                        No representa causalidad.
                    </div>

                </div>
                """
            )


            plot_data = pd.DataFrame(
                {
                    "Métrica": [
                        x[0]
                        for x
                        in values
                    ],

                    "Antes": [
                        x[1]
                        for x
                        in values
                    ],

                    "Después": [
                        x[2]
                        for x
                        in values
                    ],
                }
            )


            long_data = (
                plot_data
                .melt(
                    id_vars=
                        "Métrica",
                    var_name=
                        "Ventana",
                    value_name=
                        "Eventos",
                )
            )


            figure = px.bar(
                long_data,
                x=
                    "Métrica",
                y=
                    "Eventos",
                color=
                    "Ventana",
                barmode=
                    "group",
                text=
                    "Eventos",

                color_discrete_map={
                    "Antes":
                        BEFORE_COLOR,

                    "Después":
                        AFTER_COLOR,
                },
            )


            figure.update_traces(
                textposition=
                    "outside",

                cliponaxis=
                    False,
            )


            figure = (
                apply_coachlens_chart_style(
                    figure,
                    height=
                        350,
                    legend=
                        True,
                    x_title=
                        "",
                    y_title=
                        "Número de eventos",
                )
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
    # BALÓN PARADO
    # ========================================================

    else:


        selected_phase = (
            "OFENSIVO"
            if
            complementary_section
            ==
            "Balón parado ofensivo"
            else
            "DEFENSIVO"
        )


        if (
            selected_phase
            ==
            "OFENSIVO"
        ):

            st.markdown(
                "### Balón parado ofensivo"
            )

            st.caption(
                "Analiza los córners y tiros libres "
                "ejecutados por América."
            )

        else:

            st.markdown(
                "### Balón parado defensivo"
            )

            st.caption(
                "Analiza los córners y tiros libres "
                "ejecutados por los rivales."
            )


        set_piece_data = (
            complementary_set_pieces[
                (
                    complementary_set_pieces[
                        "squad"
                    ]
                    ==
                    complementary_team
                )
                &
                (
                    complementary_set_pieces[
                        "phase"
                    ]
                    ==
                    selected_phase
                )
            ]
            .copy()
        )


        set_piece_data = (
            attach_match_metadata(
                set_piece_data,
                metadata,
            )
        )


        if (
            complementary_view
            ==
            "Por partido"
        ):


            rivals = sorted(
                set_piece_data[
                    "opponent"
                ]
                .dropna()
                .astype(str)
                .unique()
                .tolist()
            )


            (
                rival_col,
                match_col,
                type_col,
            ) = st.columns(
                [
                    0.9,
                    1.5,
                    0.8,
                ],
                gap="medium",
            )


            with rival_col:


                selected_rival = (
                    st.selectbox(
                        "Rival",
                        rivals,
                        key=
                            f"set_piece_rival_{selected_phase}",
                    )
                )


            rival_data = (
                set_piece_data[
                    set_piece_data[
                        "opponent"
                    ]
                    ==
                    selected_rival
                ]
                .copy()
            )


            match_options = (
                rival_data[
                    [
                        "_match_id",
                        "match_date",
                        "opponent",
                    ]
                ]
                .drop_duplicates()
                .copy()
            )


            match_options[
                "label"
            ] = (
                match_options.apply(
                    lambda row:
                        make_match_label(
                            row[
                                "match_date"
                            ],
                            row[
                                "opponent"
                            ],
                            team_name,
                        ),
                    axis=1,
                )
            )


            with match_col:


                selected_match = (
                    st.selectbox(
                        "Partido",
                        match_options[
                            "label"
                        ].tolist(),
                        key=
                            f"set_piece_match_{selected_phase}",
                    )
                )


            match_id = (
                match_options.loc[
                    match_options[
                        "label"
                    ]
                    ==
                    selected_match,
                    "_match_id",
                ]
                .iloc[0]
            )


            set_piece_data = (
                rival_data[
                    rival_data[
                        "_match_id"
                    ]
                    ==
                    match_id
                ]
                .copy()
            )


            available_types = (
                set_piece_data[
                    "set_piece"
                ]
                .dropna()
                .astype(str)
                .unique()
                .tolist()
            )


            type_options = [
                "Todos"
            ]


            if (
                "CÓRNER"
                in available_types
            ):

                type_options.append(
                    "Córner"
                )


            if (
                "TIRO LIBRE"
                in available_types
            ):

                type_options.append(
                    "Tiro libre"
                )


            with type_col:


                selected_type = (
                    st.selectbox(
                        "Tipo de jugada",
                        type_options,
                        key=
                            f"set_piece_type_{selected_phase}",
                    )
                )


            if (
                selected_type
                ==
                "Córner"
            ):

                set_piece_data = (
                    set_piece_data[
                        set_piece_data[
                            "set_piece"
                        ]
                        ==
                        "CÓRNER"
                    ]
                )


            elif (
                selected_type
                ==
                "Tiro libre"
            ):

                set_piece_data = (
                    set_piece_data[
                        set_piece_data[
                            "set_piece"
                        ]
                        ==
                        "TIRO LIBRE"
                    ]
                )


        summary = (
            aggregate_set_pieces(
                set_piece_data
            )
        )


        if summary.empty:


            st.warning(
                "No existen eventos para esta selección."
            )


        else:


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


            shot_rate = (
                total_shots
                /
                total_executions
                if
                total_executions
                >
                0
                else
                0
            )


            equivalent_per_100 = int(
                round(
                    shot_rate
                    *
                    100
                )
            )


            if (
                selected_phase
                ==
                "OFENSIVO"
            ):

                explanation = (
                    "Mide qué tan frecuentemente las "
                    "acciones de balón parado de América "
                    "terminaron generando un tiro."
                )


                dynamic_reading = (
                    f"En la selección actual, "
                    f"{safe_percent(shot_rate)} de las "
                    f"ejecuciones terminaron en tiro. "
                    f"Equivale aproximadamente a "
                    f"{equivalent_per_100} de cada 100 "
                    f"ejecuciones."
                )


                shot_label = (
                    "Tiros generados"
                )


                goal_label = (
                    "Goles generados"
                )


            else:

                explanation = (
                    "Mide qué tan frecuentemente las "
                    "acciones de balón parado del rival "
                    "terminaron generando un tiro "
                    "contra América."
                )


                dynamic_reading = (
                    f"En la selección actual, "
                    f"{safe_percent(shot_rate)} de las "
                    f"ejecuciones rivales terminaron en tiro. "
                    f"Equivale aproximadamente a "
                    f"{equivalent_per_100} de cada 100 "
                    f"ejecuciones."
                )


                shot_label = (
                    "Tiros concedidos"
                )


                goal_label = (
                    "Goles concedidos"
                )


            st.html(
                f"""
                <div class="chart-help-box">

                    <div class="chart-help-title">
                        ¿Qué significa este análisis?
                    </div>

                    <div class="chart-help-text">
                        {explanation}
                    </div>

                    <div class="chart-help-note">
                        {dynamic_reading}
                    </div>

                </div>
                """
            )


            (
                c1,
                c2,
                c3,
                c4,
            ) = st.columns(
                4,
                gap="medium",
            )


            cards = [

                (
                    c1,
                    "Ejecuciones",
                    f"{total_executions:,}",
                ),

                (
                    c2,
                    shot_label,
                    f"{total_shots:,}",
                ),

                (
                    c3,
                    goal_label,
                    f"{total_goals:,}",
                ),

                (
                    c4,
                    "Generan tiro",
                    safe_percent(
                        shot_rate
                    ),
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


            summary[
                "Tipo"
            ] = (
                summary[
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


            st.markdown(
                "#### Desglose por tipo de jugada"
            )


            breakdown_columns = (
                st.columns(
                    len(
                        summary
                    ),
                    gap="medium",
                )
            )


            for (
                column,
                (_, row),
            ) in zip(
                breakdown_columns,
                summary.iterrows(),
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

                                    <span>
                                        EJECUCIONES
                                    </span>

                                    <strong>
                                        {int(row["executions"])}
                                    </strong>

                                </div>

                                <div>

                                    <span>
                                        TIROS
                                    </span>

                                    <strong>
                                        {int(row["shots"])}
                                    </strong>

                                </div>

                                <div>

                                    <span>
                                        GOLES
                                    </span>

                                    <strong>
                                        {int(row["goals"])}
                                    </strong>

                                </div>

                                <div>

                                    <span>
                                        GENERAN TIRO
                                    </span>

                                    <strong>
                                        {safe_percent(row["shot_rate"])}
                                    </strong>

                                </div>

                            </div>

                        </div>
                        """
                    )


            plot_data = (
                summary[
                    [
                        "Tipo",
                        "shot_rate",
                    ]
                ]
                .copy()
            )


            plot_data[
                "Porcentaje"
            ] = (
                plot_data[
                    "shot_rate"
                ]
                *
                100
            )


            plot_data = (
                plot_data
                .sort_values(
                    "Porcentaje",
                    ascending=True,
                )
            )


            figure = px.bar(
                plot_data,
                x=
                    "Porcentaje",
                y=
                    "Tipo",
                orientation=
                    "h",
                text=
                    "Porcentaje",
            )


            figure.update_traces(
                marker_color=
                    COACH_YELLOW,

                texttemplate=
                    "%{text:.1f}%",

                textposition=
                    "outside",

                cliponaxis=
                    False,
            )


            maximum = float(
                plot_data[
                    "Porcentaje"
                ].max()
            )


            figure = (
                apply_coachlens_chart_style(
                    figure,
                    height=
                        max(
                            230,
                            100
                            +
                            (
                                len(
                                    plot_data
                                )
                                *
                                70
                            ),
                        ),
                    x_title=
                        "Porcentaje de ejecuciones que generan tiro",
                    y_title=
                        "",
                )
            )


            figure.update_xaxes(
                ticksuffix=
                    "%",
                range=[
                    0,
                    max(
                        50,
                        maximum
                        +
                        8,
                    ),
                ],
            )


            st.plotly_chart(
                figure,
                use_container_width=True,
                config={
                    "displayModeBar":
                        False
                },
            )


            main_row = (
                summary
                .sort_values(
                    "shot_rate",
                    ascending=False,
                )
                .iloc[0]
            )


            if (
                selected_phase
                ==
                "OFENSIVO"
            ):

                reading_title = (
                    "Mayor generación de tiro"
                )


                conclusion = (
                    f"{main_row['Tipo']} presenta la mayor "
                    f"tasa de generación de tiro: "
                    f"{safe_percent(main_row['shot_rate'])}."
                )


            else:

                reading_title = (
                    "Mayor vulnerabilidad relativa"
                )


                conclusion = (
                    f"{main_row['Tipo']} presenta la mayor "
                    f"tasa de tiro rival: "
                    f"{safe_percent(main_row['shot_rate'])}."
                )


            st.html(
                f"""
                <div class="complementary-reading">

                    <div class="complementary-reading-title">
                        {reading_title}
                    </div>

                    <div class="complementary-reading-text">
                        {conclusion}
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
            EVIDENCIA Y METODOLOGÍA
        </div>

        <div class="page-title">
            Rendimiento y marco analítico
        </div>

        <div class="page-description">
            Presenta desempeño, variables relevantes,
            supuestos y límites del sistema CoachLens.
        </div>
        """
    )


    model_team = (
        st.radio(
            "Equipo",
            [
                "Varonil",
                "Femenil",
            ],
            horizontal=True,
            key=
                "model_team",
        )
    )


    # ========================================================
    # VARONIL
    # ========================================================

    if (
        model_team
        ==
        "Varonil"
    ):


        columns = (
            st.columns(
                4,
                gap="medium",
            )
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


        for (
            column,
            (
                label,
                value,
            ),
        ) in zip(
            columns,
            metrics,
        ):


            with column:


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
                        Coincide la primera predicción con
                        el estado observado.
                    </div>

                    <div>
                        <strong>Top-2</strong><br>
                        El estado real aparece entre las
                        dos principales alternativas.
                    </div>

                    <div>
                        <strong>Baseline</strong><br>
                        Compara contra mantener el estado actual.
                    </div>

                    <div>
                        <strong>Balanceada</strong><br>
                        Promedia el rendimiento entre estados.
                    </div>

                </div>

            </div>
            """
        )


        if not male_shap.empty:


            st.markdown(
                "### Influencia global · Modelo Varonil"
            )


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


            figure = px.bar(
                importance,
                x=
                    "mean_abs_shap",
                y=
                    "label",
                orientation=
                    "h",
            )


            figure.update_traces(
                marker_color=
                    COACH_YELLOW
            )


            figure = (
                apply_coachlens_chart_style(
                    figure,
                    height=
                        350,
                    x_title=
                        "Influencia SHAP promedio",
                    y_title=
                        "",
                )
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
    # FEMENIL
    # ========================================================

    else:


        metric_names = [

            (
                "Accuracy",
                "accuracy",
            ),

            (
                "Exactitud balanceada",
                "balanced_accuracy",
            ),

            (
                "AUC",
                "auc",
            ),

            (
                "F1 · cambio",
                "f1_change",
            ),
        ]


        columns = (
            st.columns(
                4,
                gap="medium",
            )
        )


        for (
            column,
            (
                label,
                metric_name,
            ),
        ) in zip(
            columns,
            metric_names,
        ):


            value = (
                metric_from_table(
                    female_metrics,
                    metric_name,
                )
            )


            with column:


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
                        Porcentaje total de clasificaciones
                        correctas.
                    </div>

                    <div>
                        <strong>Balanceada</strong><br>
                        Evalúa cambio y estabilidad por igual.
                    </div>

                    <div>
                        <strong>AUC</strong><br>
                        Capacidad de separar escenarios de
                        cambio y estabilidad.
                    </div>

                    <div>
                        <strong>F1</strong><br>
                        Balance entre precisión y sensibilidad.
                    </div>

                </div>

            </div>
            """
        )


        importance = (
            get_xgb_importance(
                female_model,
                8,
            )
        )


        if not importance.empty:


            st.markdown(
                "### Influencia global · Modelo Femenil"
            )


            figure = px.bar(
                importance,
                x=
                    "importance",
                y=
                    "label",
                orientation=
                    "h",
            )


            figure.update_traces(
                marker_color=
                    COACH_YELLOW
            )


            figure = (
                apply_coachlens_chart_style(
                    figure,
                    height=
                        350,
                    x_title=
                        "Ganancia promedio en XGBoost",
                    y_title=
                        "",
                )
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
    # FRAMEWORK
    # ========================================================

    st.markdown(
        "### Framework analítico de CoachLens"
    )


    st.html(
        """
        <div class="framework-section-box">

            <div class="framework-section-title">
                Definición de objetos
            </div>

            <div class="framework-grid">

                <div class="framework-card">

                    <div class="framework-title">
                        Estado táctico
                    </div>

                    <div class="framework-text">
                        Representación del comportamiento
                        colectivo del equipo en un momento.
                    </div>

                </div>


                <div class="framework-card">

                    <div class="framework-title">
                        Transición
                    </div>

                    <div class="framework-text">
                        Cambio entre un estado actual
                        y un estado posterior.
                    </div>

                </div>


                <div class="framework-card">

                    <div class="framework-title">
                        Contexto
                    </div>

                    <div class="framework-text">
                        Combinación de minuto, marcador,
                        condición y comportamiento de juego.
                    </div>

                </div>


                <div class="framework-card">

                    <div class="framework-title">
                        Contexto similar
                    </div>

                    <div class="framework-text">
                        Situación histórica comparable
                        con el escenario actual.
                    </div>

                </div>


                <div class="framework-card">

                    <div class="framework-title">
                        Cambio táctico
                    </div>

                    <div class="framework-text">
                        Modificación del estado colectivo
                        dentro del horizonte de análisis.
                    </div>

                </div>


                <div class="framework-card">

                    <div class="framework-title">
                        Simulación
                    </div>

                    <div class="framework-text">
                        Repetición probabilística del mismo
                        contexto para mostrar incertidumbre.
                    </div>

                </div>

            </div>

        </div>
        """
    )


    st.html(
        """
        <div class="framework-section-box">

            <div class="framework-section-title">
                Supuestos explícitos
            </div>

            <div class="framework-grid">

                <div class="framework-card">

                    <div class="framework-title">
                        Horizonte
                    </div>

                    <div class="framework-text">
                        CoachLens utiliza un horizonte
                        futuro de cinco minutos.
                    </div>

                </div>


                <div class="framework-card">

                    <div class="framework-title">
                        Historial
                    </div>

                    <div class="framework-text">
                        Los escenarios históricos sirven
                        como referencia contextual.
                    </div>

                </div>


                <div class="framework-card">

                    <div class="framework-title">
                        Sustituciones
                    </div>

                    <div class="framework-text">
                        El análisis antes/después es
                        descriptivo y no causal.
                    </div>

                </div>


                <div class="framework-card">

                    <div class="framework-title">
                        Varonil
                    </div>

                    <div class="framework-text">
                        Estima un estado táctico futuro.
                    </div>

                </div>


                <div class="framework-card">

                    <div class="framework-title">
                        Femenil
                    </div>

                    <div class="framework-text">
                        Estima riesgo de cambio o estabilidad.
                    </div>

                </div>


                <div class="framework-card">

                    <div class="framework-title">
                        Simulación
                    </div>

                    <div class="framework-text">
                        Representa incertidumbre; no predice
                        el marcador final.
                    </div>

                </div>

            </div>

        </div>
        """
    )


    st.html(
        """
        <div class="framework-section-box">

            <div class="framework-section-title">
                Coherencia interna
            </div>

            <div class="framework-grid">

                <div class="framework-card">

                    <div class="framework-title">
                        Modelo Varonil
                    </div>

                    <div class="framework-text">
                        Predice el estado táctico probable
                        a cinco minutos.
                    </div>

                </div>


                <div class="framework-card">

                    <div class="framework-title">
                        Modelo Femenil
                    </div>

                    <div class="framework-text">
                        Predice riesgo de cambio táctico
                        a cinco minutos.
                    </div>

                </div>


                <div class="framework-card">

                    <div class="framework-title">
                        Revisión TEST
                    </div>

                    <div class="framework-text">
                        Contrasta estimaciones con
                        observaciones reservadas.
                    </div>

                </div>


                <div class="framework-card">

                    <div class="framework-title">
                        Identidad
                    </div>

                    <div class="framework-text">
                        Resume construcción, progresión,
                        ocasiones y comportamiento defensivo.
                    </div>

                </div>


                <div class="framework-card">

                    <div class="framework-title">
                        Sustituciones
                    </div>

                    <div class="framework-text">
                        Complementan el modelo mediante
                        evidencia descriptiva observada.
                    </div>

                </div>


                <div class="framework-card">

                    <div class="framework-title">
                        Balón parado
                    </div>

                    <div class="framework-text">
                        Evalúa producción y vulnerabilidad
                        mediante eventos registrados.
                    </div>

                </div>

            </div>

        </div>
        """
    )


    st.html(
        """
        <div class="framework-section-box">

            <div class="framework-section-title">
                Alcance y límites
            </div>

            <div class="scope-box">

                <div class="scope-column">

                    <div class="scope-heading">
                        Incluido
                    </div>

                    <ul>
                        <li>Estados tácticos.</li>
                        <li>Riesgo de cambio táctico.</li>
                        <li>Identidad de juego.</li>
                        <li>Construcción y progresión.</li>
                        <li>Generación de ocasiones.</li>
                        <li>Presión y recuperaciones.</li>
                        <li>Uso de jugadores.</li>
                        <li>Sustituciones.</li>
                        <li>Balón parado ofensivo.</li>
                        <li>Balón parado defensivo.</li>
                        <li>StatsBomb Events + 360.</li>
                        <li>Validación temporal TEST.</li>
                        <li>Simulación probabilística.</li>
                    </ul>

                </div>


                <div class="scope-column">

                    <div class="scope-heading">
                        Fuera del alcance actual
                    </div>

                    <ul>
                        <li>Predicción del marcador final.</li>
                        <li>Apuestas.</li>
                        <li>Inferencia causal individual.</li>
                        <li>Efecto causal de sustituciones.</li>
                        <li>Modelado completo de marcajes.</li>
                        <li>Simulación completa de 90 minutos.</li>
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