from pathlib import Path
import unicodedata

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
import xgboost as xgb


# ============================================================
# 1. CONFIGURACIÓN
# ============================================================

st.set_page_config(
    page_title="CoachLens | Club América",
    page_icon="🦅",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# 2. RUTAS
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

CSS_PATH = (
    ROOT
    / "dashboard"
    / "styles"
    / "america.css"
)

SIMULATOR_DATA_PATH = (
    ROOT
    / "dashboard"
    / "data"
    / "simulator_context.csv"
)

MATCH_HISTORY_DEPLOY_PATH = (
    ROOT
    / "dashboard"
    / "data"
    / "match_history.csv"
)

MATCH_HISTORY_LOCAL_PATH = (
    ROOT
    / "data"
    / "processed"
    / "america_male"
    / "america_matches.csv"
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

ALERT_PERFORMANCE_PATH = (
    ROOT
    / "outputs"
    / "coach_engine"
    / "coach_engine_alert_performance.csv"
)

SHAP_GLOBAL_PATH = (
    ROOT
    / "outputs"
    / "models"
    / "interpretability"
    / "shap_global_importance.csv"
)

GOAL_SUMMARY_PATH = (
    ROOT
    / "outputs"
    / "models"
    / "event_study"
    / "goal_event_summary.csv"
)

GOAL_EFFECTS_PATH = (
    ROOT
    / "outputs"
    / "models"
    / "event_study"
    / "goal_feature_effects.csv"
)


# ============================================================
# 3. CSS
# ============================================================

def load_css(path):

    if not path.exists():
        st.warning(
            f"No se encontró el CSS: {path}"
        )
        return

    with open(
        path,
        "r",
        encoding="utf-8",
    ) as css_file:

        st.html(
            f"""
            <style>
            {css_file.read()}
            </style>
            """
        )


load_css(CSS_PATH)


# ============================================================
# 4. COLORES
# ============================================================

AMERICA_NAVY = "#001F5B"
AMERICA_BLUE = "#003B7A"
AMERICA_YELLOW = "#F9D616"
AMERICA_RED = "#E21D2E"


# ============================================================
# 5. ESTADOS TÁCTICOS
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

STATE_FEATURES = [
    "current_state_control",
    "current_state_pressure",
    "current_state_transition",
    "current_state_protection",
    "current_state_urgency",
]


# ============================================================
# 6. FEATURES REALES DEL XGBOOST
# ============================================================

MODEL_FEATURES = [

    # Contexto
    "minute",
    "game_state_num",

    # Estado actual
    *STATE_FEATURES,

    # Events - 5 minutos
    "america_field_tilt_last5",
    "entries_final_third_diff_last5",
    "entries_box_diff_last5",
    "progressive_pass_diff_last5",
    "america_progressive_carries_last5",
    "america_progressive_passes_last5",
    "pressure_diff_last5",
    "recovery_height_diff_last5",

    # Events - 3 minutos
    "america_field_tilt_last3",
    "entries_final_third_diff_last3",
    "entries_box_diff_last3",
    "progressive_pass_diff_last3",
    "america_progressive_carries_last3",
    "america_progressive_passes_last3",
    "pressure_diff_last3",
    "recovery_height_diff_last3",

    # StatsBomb 360 - 5 minutos
    "america_local_superiority_10_last5",
    "america_pressure_share_5_last5",
    "america_nearest_opponent_distance_last5",
    "america_support_share_10_last5",

    # StatsBomb 360 - 3 minutos
    "america_local_superiority_10_last3",
    "america_pressure_share_5_last3",
    "america_nearest_opponent_distance_last3",
    "america_support_share_10_last3",
]


FEATURE_NAMES = {

    "minute":
        "Minuto del partido",

    "game_state_num":
        "Situación del marcador",

    "current_state_control":
        "Estado táctico actual",

    "current_state_pressure":
        "Estado táctico actual",

    "current_state_transition":
        "Estado táctico actual",

    "current_state_protection":
        "Estado táctico actual",

    "current_state_urgency":
        "Estado táctico actual",

    "america_field_tilt_last5":
        "Dominio territorial reciente",

    "entries_final_third_diff_last5":
        "Entradas al último tercio",

    "entries_box_diff_last5":
        "Entradas al área",

    "progressive_pass_diff_last5":
        "Balance de pases progresivos",

    "america_progressive_carries_last5":
        "Conducciones progresivas",

    "america_progressive_passes_last5":
        "Pases progresivos",

    "pressure_diff_last5":
        "Balance de presión",

    "recovery_height_diff_last5":
        "Altura de recuperación",

    "america_field_tilt_last3":
        "Dominio territorial de corto plazo",

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
        "Apoyo cercano de compañeros",

    "america_local_superiority_10_last3":
        "Superioridad numérica reciente",

    "america_pressure_share_5_last3":
        "Presión rival reciente",

    "america_nearest_opponent_distance_last3":
        "Proximidad del rival",

    "america_support_share_10_last3":
        "Apoyo reciente de compañeros",
}


# ============================================================
# 7. CARGA DE ARCHIVOS
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


def get_match_history():

    # En Streamlit Cloud usamos el archivo ligero.
    if MATCH_HISTORY_DEPLOY_PATH.exists():

        return load_csv(
            MATCH_HISTORY_DEPLOY_PATH
        )

    # En local permitimos fallback al archivo procesado.
    if MATCH_HISTORY_LOCAL_PATH.exists():

        return load_csv(
            MATCH_HISTORY_LOCAL_PATH
        )

    return pd.DataFrame()


# ============================================================
# 8. HELPERS
# ============================================================

def state_display(state):

    return STATE_DISPLAY.get(
        str(state),
        str(state),
    )


def safe_percent(value):

    try:

        value = float(value)

        if value <= 1:
            value *= 100

        return f"{value:.1f}%"

    except Exception:

        return "N/D"


def normalize_text(value):

    value = str(value).strip().lower()

    value = "".join(
        char
        for char in unicodedata.normalize(
            "NFD",
            value,
        )
        if unicodedata.category(char)
        !=
        "Mn"
    )

    return value


# ============================================================
# 9. CONTAR PARTIDOS JUGADOS VS RIVAL
#
# NO ESTÁ HARDCODEADO.
# Se calcula desde match_history.csv.
# ============================================================

def count_played_matches_vs_opponent(
    matches,
    opponent,
):

    if matches.empty:
        return 0

    data = matches.copy()


    # --------------------------------------------------------
    # Excluir partidos todavía no jugados
    # --------------------------------------------------------

    if (
        "home_score" in data.columns
        and
        "away_score" in data.columns
    ):

        home_score = pd.to_numeric(
            data["home_score"],
            errors="coerce",
        )

        away_score = pd.to_numeric(
            data["away_score"],
            errors="coerce",
        )

        data = data[
            home_score.notna()
            &
            away_score.notna()
        ].copy()


    target = normalize_text(
        opponent
    )


    # --------------------------------------------------------
    # Caso 1: tenemos opponent directamente
    # --------------------------------------------------------

    if (
        "opponent"
        in data.columns
    ):

        opponent_normalized = (
            data["opponent"]
            .astype(str)
            .map(normalize_text)
        )

        filtered = data[
            opponent_normalized.eq(
                target
            )
        ].copy()


    # --------------------------------------------------------
    # Caso 2: home_team / away_team
    # --------------------------------------------------------

    elif (
        "home_team" in data.columns
        and
        "away_team" in data.columns
    ):

        home_normalized = (
            data["home_team"]
            .astype(str)
            .map(normalize_text)
        )

        away_normalized = (
            data["away_team"]
            .astype(str)
            .map(normalize_text)
        )

        filtered = data[
            home_normalized.eq(target)
            |
            away_normalized.eq(target)
        ].copy()


    else:

        return 0


    if (
        "match_id"
        in filtered.columns
    ):

        return int(
            filtered["match_id"]
            .nunique()
        )


    return int(
        len(filtered)
    )


# ============================================================
# 10. RECOMENDACIÓN OPERATIVA
#
# OJO:
# Esto NO es una predicción adicional.
# Es una capa de decisión basada en el estado que predijo XGBoost.
# ============================================================

def coach_action_for_state(
    state,
    score_state,
):

    if state == "CONTROL":

        return (
            "Conservar el control de la siguiente fase. "
            "Mantener la posesión, mover al rival antes de progresar "
            "y proteger el equilibrio central."
        )


    if state == "PRESSURE":

        return (
            "Preparar una presión coordinada. "
            "Cerrar opciones interiores y utilizar un pase hacia atrás "
            "o un mal control rival como detonante."
        )


    if state == "TRANSITION":

        return (
            "Prepararse para una transición. "
            "Asegurar el primer pase tras recuperar, atacar los espacios "
            "libres y mantener equilibrio defensivo."
        )


    if state == "PROTECTION":

        if score_state > 0:

            return (
                "Gestionar la ventaja. "
                "Compactar zonas centrales, mantener balance defensivo "
                "y evitar pérdidas innecesarias."
            )

        return (
            "Proteger zonas centrales, reducir distancias entre líneas "
            "y limitar pérdidas de alto riesgo."
        )


    if state == "URGENCY":

        if score_state < 0:

            return (
                "Aumentar el riesgo ofensivo. "
                "Acelerar la progresión, ocupar el área con más jugadores "
                "y realizar contrapresión inmediata tras pérdida."
            )

        return (
            "Aumentar el ritmo ofensivo, ocupar antes el área "
            "y realizar contrapresión inmediata."
        )


    return (
        "Monitorear la siguiente fase táctica."
    )


# ============================================================
# 11. SHAP LOCAL
#
# Sale directamente del XGBoost para la predicción actual.
# ============================================================

def local_shap_reasons(
    model,
    matrix,
    predicted_index,
):

    dmatrix = xgb.DMatrix(
        matrix,
        feature_names=MODEL_FEATURES,
    )

    contributions = model.predict(
        dmatrix,
        pred_contribs=True,
        strict_shape=True,
    )

    n_features = len(
        MODEL_FEATURES
    )

    n_classes = len(
        STATE_LEVELS
    )


    if contributions.ndim == 3:

        if (
            contributions.shape[1]
            ==
            n_classes
        ):

            values = contributions[
                0,
                predicted_index,
                :n_features,
            ]

        elif (
            contributions.shape[2]
            ==
            n_classes
        ):

            values = contributions[
                0,
                :n_features,
                predicted_index,
            ]

        else:

            return []


    elif contributions.ndim == 2:

        flat = contributions[0]

        expected_size = (
            n_classes
            *
            (
                n_features + 1
            )
        )

        if (
            len(flat)
            !=
            expected_size
        ):

            return []

        reshaped = flat.reshape(
            n_classes,
            n_features + 1,
        )

        values = reshaped[
            predicted_index,
            :n_features,
        ]


    else:

        return []


    positive_indices = np.where(
        values > 0
    )[0]


    if len(positive_indices) > 0:

        ordered = positive_indices[
            np.argsort(
                values[
                    positive_indices
                ]
            )[::-1]
        ]

    else:

        ordered = np.argsort(
            np.abs(values)
        )[::-1]


    top_indices = ordered[:3]


    return [
        FEATURE_NAMES.get(
            MODEL_FEATURES[i],
            MODEL_FEATURES[i],
        )
        for i
        in top_indices
    ]


# ============================================================
# 12. DIVERSIDAD DE COMPARABLES
#
# Máximo 2 momentos del mismo partido.
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


    if (
        "match_id"
        not in
        pool.columns
    ):

        return (
            pool
            .head(n_neighbors)
            .copy()
        )


    selected_indices = []

    match_counts = {}


    for index, row in pool.iterrows():

        match_id = str(
            row["match_id"]
        )

        current_count = match_counts.get(
            match_id,
            0,
        )

        if (
            current_count
            >=
            max_per_match
        ):

            continue


        selected_indices.append(
            index
        )

        match_counts[
            match_id
        ] = (
            current_count + 1
        )


        if (
            len(selected_indices)
            >=
            n_neighbors
        ):

            break


    return (
        pool.loc[
            selected_indices
        ]
        .copy()
    )


# ============================================================
# 13. CONSTRUIR ESCENARIO
#
# El rival/localía ayudan a encontrar contexto histórico.
# Minuto, marcador y estado actual además entran DIRECTAMENTE
# al XGBoost.
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

    base = data.copy()


    # --------------------------------------------------------
    # Normalización
    # --------------------------------------------------------

    base["venue"] = (
        base["venue"]
        .astype(str)
        .str.upper()
    )


    base["current_tactical_state"] = (
        base["current_tactical_state"]
        .astype(str)
        .str.upper()
    )


    base["opponent"] = (
        base["opponent"]
        .astype(str)
    )


    base["minute"] = pd.to_numeric(
        base["minute"],
        errors="coerce",
    )


    base["game_state_num"] = pd.to_numeric(
        base["game_state_num"],
        errors="coerce",
    )


    base = base.dropna(
        subset=[
            "minute",
            "game_state_num",
        ]
    )


    # --------------------------------------------------------
    # Partidos vs rival disponibles para construir escenarios
    #
    # Este número puede ser menor que el histórico total
    # porque simulator_context usa TRAIN + VALIDATION.
    # --------------------------------------------------------

    opponent_mask = (
        base["opponent"]
        .astype(str)
        .map(normalize_text)
        .eq(
            normalize_text(opponent)
        )
    )


    exact_opponent_rows = base[
        opponent_mask
    ].copy()


    if (
        "match_id"
        in exact_opponent_rows.columns
    ):

        exact_opponent_matches = (
            exact_opponent_rows["match_id"]
            .nunique()
        )

    else:

        exact_opponent_matches = 0


    pool = base.copy()


    # --------------------------------------------------------
    # Local / visitante
    # --------------------------------------------------------

    venue_pool = pool[
        pool["venue"]
        ==
        venue
    ].copy()


    venue_filter_used = False


    if (
        len(venue_pool)
        >=
        30
    ):

        pool = venue_pool

        venue_filter_used = True


    # --------------------------------------------------------
    # Estado táctico actual
    # --------------------------------------------------------

    state_pool = pool[
        pool[
            "current_tactical_state"
        ]
        ==
        current_state
    ].copy()


    state_filter_used = False


    if (
        len(state_pool)
        >=
        30
    ):

        pool = state_pool

        state_filter_used = True


    # --------------------------------------------------------
    # Posesión
    # --------------------------------------------------------

    possession_available = False

    possession_filter_used = False


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


        if (
            possession.notna().sum()
            >=
            30
        ):

            possession_available = True


            max_possession = possession.max(
                skipna=True
            )


            if (
                max_possession
                <=
                1.5
            ):

                if (
                    possession_choice
                    ==
                    "América"
                ):

                    possession_filter = (
                        possession >= 0.55
                    )


                elif (
                    possession_choice
                    ==
                    "Rival"
                ):

                    possession_filter = (
                        possession <= 0.45
                    )


                else:

                    possession_filter = (
                        possession.between(
                            0.45,
                            0.55,
                        )
                    )


            else:

                if (
                    possession_choice
                    ==
                    "América"
                ):

                    possession_filter = (
                        possession >= 55
                    )


                elif (
                    possession_choice
                    ==
                    "Rival"
                ):

                    possession_filter = (
                        possession <= 45
                    )


                else:

                    possession_filter = (
                        possession.between(
                            45,
                            55,
                        )
                    )


            possession_pool = pool[
                possession_filter
            ].copy()


            if (
                len(possession_pool)
                >=
                20
            ):

                pool = possession_pool

                possession_filter_used = True


    # --------------------------------------------------------
    # Distancia histórica
    # --------------------------------------------------------

    minute_distance = (
        (
            pool["minute"]
            -
            minute
        ).abs()
        /
        10
    )


    score_distance = (
        pool["game_state_num"]
        -
        score_state
    ).abs()


    distance = (
        minute_distance
        +
        2.2
        *
        score_distance
    )


    # --------------------------------------------------------
    # Penalizaciones cuando no se hizo filtro duro
    # --------------------------------------------------------

    if (
        not venue_filter_used
    ):

        distance += np.where(
            pool["venue"]
            ==
            venue,
            0.0,
            1.25,
        )


    if (
        not state_filter_used
    ):

        distance += np.where(
            pool[
                "current_tactical_state"
            ]
            ==
            current_state,
            0.0,
            1.50,
        )


    # --------------------------------------------------------
    # Rival exacto obtiene ventaja de similitud.
    #
    # NO es una feature directa del XGBoost.
    # --------------------------------------------------------

    pool_opponent_normalized = (
        pool["opponent"]
        .astype(str)
        .map(normalize_text)
    )


    distance += np.where(
        pool_opponent_normalized.eq(
            normalize_text(opponent)
        ),
        0.0,
        1.10,
    )


    pool = pool.copy()

    pool["_distance"] = distance


    # --------------------------------------------------------
    # Máximo 2 momentos por partido
    # --------------------------------------------------------

    pool = select_diverse_neighbors(
        pool,
        n_neighbors=n_neighbors,
        max_per_match=2,
    )


    if pool.empty:

        raise ValueError(
            "No se encontraron situaciones históricas comparables."
        )


    # --------------------------------------------------------
    # Pesos de los comparables
    # --------------------------------------------------------

    weights = np.exp(
        -1.5
        *
        pool["_distance"]
        .to_numpy()
    )


    if (
        not np.isfinite(weights).all()
        or
        weights.sum() == 0
    ):

        weights = np.ones(
            len(pool)
        )


    weights = (
        weights
        /
        weights.sum()
    )


    # --------------------------------------------------------
    # Estimar features tácticas del escenario
    # --------------------------------------------------------

    feature_values = {}


    for feature in MODEL_FEATURES:

        if (
            feature
            in
            STATE_FEATURES
        ):

            continue


        if (
            feature
            not in
            pool.columns
        ):

            feature_values[
                feature
            ] = 0.0

            continue


        values = pd.to_numeric(
            pool[feature],
            errors="coerce",
        ).to_numpy()


        valid = np.isfinite(
            values
        )


        if valid.any():

            valid_weights = (
                weights[valid]
            )


            valid_weights = (
                valid_weights
                /
                valid_weights.sum()
            )


            feature_values[
                feature
            ] = np.sum(
                values[valid]
                *
                valid_weights
            )

        else:

            feature_values[
                feature
            ] = 0.0


    # --------------------------------------------------------
    # Inputs DIRECTOS del modelo
    # --------------------------------------------------------

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

        state_feature = (
            "current_state_"
            +
            state.lower()
        )


        feature_values[
            state_feature
        ] = (
            1.0
            if state == current_state
            else 0.0
        )


    # --------------------------------------------------------
    # Vector final de XGBoost
    # --------------------------------------------------------

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


    # --------------------------------------------------------
    # Similitud heurística
    #
    # NO es confianza del XGBoost.
    # --------------------------------------------------------

    mean_distance = float(
        pool["_distance"].mean()
    )


    similarity = (
        100
        /
        (
            1
            +
            mean_distance
        )
    )


    similarity = float(
        np.clip(
            similarity,
            0,
            100,
        )
    )


    # --------------------------------------------------------
    # Diagnósticos
    # --------------------------------------------------------

    if (
        "match_id"
        in pool.columns
    ):

        distinct_matches = int(
            pool["match_id"]
            .nunique()
        )

    else:

        distinct_matches = 0


    exact_opponent_moments = int(
        pool["opponent"]
        .astype(str)
        .map(normalize_text)
        .eq(
            normalize_text(opponent)
        )
        .sum()
    )


    diagnostic = {

        "exact_opponent_matches":
            int(
                exact_opponent_matches
            ),

        "historical_moments":
            int(
                len(pool)
            ),

        "distinct_matches":
            distinct_matches,

        "exact_opponent_moments":
            exact_opponent_moments,

        "average_minute":
            round(
                float(
                    pool["minute"]
                    .mean()
                ),
                1,
            ),

        "average_score_state":
            round(
                float(
                    pool[
                        "game_state_num"
                    ]
                    .mean()
                ),
                2,
            ),

        "venue_filter_used":
            venue_filter_used,

        "state_filter_used":
            state_filter_used,

        "possession_available":
            possession_available,

        "possession_filter_used":
            possession_filter_used,
    }


    return (
        feature_row,
        pool,
        similarity,
        diagnostic,
    )


# ============================================================
# 14. EVIDENCIA DEL MODELO CALCULADA DEL TEST
#
# NO escribimos 40.10 / 64.45 manualmente.
# ============================================================

def calculate_model_evidence(
    test_data,
):

    metrics = {
        "top1": None,
        "top2": None,
        "persistence": None,
        "balanced": None,
        "n": 0,
    }


    required = [
        "actual_future_state",
        "predicted_state",
    ]


    if (
        test_data.empty
        or
        not all(
            column
            in
            test_data.columns
            for column
            in required
        )
    ):

        return metrics


    evaluation = test_data.copy()


    evaluation = evaluation.dropna(
        subset=[
            "actual_future_state",
            "predicted_state",
        ]
    )


    if evaluation.empty:

        return metrics


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


    metrics["n"] = len(
        evaluation
    )


    # Top-1
    metrics["top1"] = float(
        (
            actual
            ==
            predicted
        ).mean()
    )


    # Top-2
    if (
        "second_state"
        in
        evaluation.columns
    ):

        second = (
            evaluation[
                "second_state"
            ]
            .astype(str)
            .str.upper()
        )


        metrics["top2"] = float(
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


    # Persistence baseline
    if (
        "current_state"
        in
        evaluation.columns
    ):

        current = (
            evaluation[
                "current_state"
            ]
            .astype(str)
            .str.upper()
        )


        metrics[
            "persistence"
        ] = float(
            (
                actual
                ==
                current
            ).mean()
        )


    # Balanced accuracy
    recalls = []


    for state in STATE_LEVELS:

        mask = (
            actual
            ==
            state
        )


        if (
            mask.sum()
            ==
            0
        ):

            continue


        recall = (
            predicted[
                mask
            ]
            ==
            state
        ).mean()


        recalls.append(
            float(recall)
        )


    if recalls:

        metrics[
            "balanced"
        ] = float(
            np.mean(
                recalls
            )
        )


    return metrics


# ============================================================
# 15. CARGAR DATOS
# ============================================================

simulator_data = load_csv(
    SIMULATOR_DATA_PATH
)

match_history = get_match_history()

coach_all = load_csv(
    COACH_ALL_PATH
)

alert_performance = load_csv(
    ALERT_PERFORMANCE_PATH
)

shap_global = load_csv(
    SHAP_GLOBAL_PATH
)

goal_summary = load_csv(
    GOAL_SUMMARY_PATH
)

goal_effects = load_csv(
    GOAL_EFFECTS_PATH
)


# ============================================================
# 16. CARGAR XGBOOST
# ============================================================

model = None


if MODEL_PATH.exists():

    try:

        model = load_model(
            MODEL_PATH
        )

    except Exception as error:

        st.error(
            f"No se pudo cargar el XGBoost: {error}"
        )


# ============================================================
# 17. SIDEBAR
# ============================================================

with st.sidebar:

    st.html(
        """
        <div class="sidebar-logo">

            <div class="sidebar-circle">
                CA
            </div>

            <div class="sidebar-title">
                COACHLENS
            </div>

            <div class="sidebar-subtitle">
                CLUB AMÉRICA
            </div>

        </div>
        """
    )


    st.divider()


    st.markdown(
        "**EQUIPO**"
    )

    st.write(
        "🦅 Club América — Varonil"
    )


    st.markdown(
        "**MODELO PREDICTIVO**"
    )

    st.write(
        "XGBoost multiclase"
    )


    st.markdown(
        "**HORIZONTE**"
    )

    st.write(
        "Próximos 5 minutos"
    )


    st.markdown(
        "**DATOS**"
    )

    st.write(
        "StatsBomb Events + 360"
    )


    st.divider()


    if model is not None:

        st.success(
            "● Modelo activo"
        )

    else:

        st.error(
            "● Modelo no disponible"
        )


    st.caption(
        "Evaluación con holdout temporal"
    )


# ============================================================
# 18. HEADER
# ============================================================

st.html(
    """
    <div class="america-header">

        <div class="header-kicker">
            Inteligencia Táctica · Club América
        </div>

        <div class="header-title">
            CoachLens
        </div>

        <div class="header-subtitle">
            Anticipa la siguiente fase táctica.
            Entiende por qué.
            Actúa antes de que cambie el partido.
        </div>

    </div>
    """
)


# ============================================================
# 19. PESTAÑAS
# ============================================================

(
    tab_simulator,
    tab_replay,
    tab_tactical,
    tab_why,
    tab_evidence,
) = st.tabs(
    [
        "🦅 Simulador de Escenarios",
        "⏪ Revisión Histórica",
        "📊 Inteligencia Táctica",
        "🧠 ¿Por qué?",
        "📈 Evidencia del Modelo",
    ]
)


# ============================================================
# 20. SIMULADOR
# ============================================================

with tab_simulator:

    st.html(
        """
        <div class="section-kicker">
            Simulador de decisiones
        </div>
        """
    )


    st.header(
        "Construye la situación del partido"
    )


    st.caption(
        "CoachLens completa el contexto táctico con situaciones "
        "históricas comparables y envía esas variables al modelo "
        "XGBoost entrenado."
    )


    if simulator_data.empty:

        st.error(
            "No se encontró simulator_context.csv."
        )


    elif model is None:

        st.error(
            "No se pudo cargar el modelo XGBoost."
        )


    else:

        opponents = sorted(
            simulator_data[
                "opponent"
            ]
            .dropna()
            .astype(str)
            .unique()
            .tolist()
        )


        # ====================================================
        # INPUTS
        # ====================================================

        col1, col2, col3 = (
            st.columns(3)
        )


        # ----------------------------------------------------
        # COLUMNA 1
        # ----------------------------------------------------

        with col1:

            venue_display = st.selectbox(
                "🏟️ Condición",
                [
                    "Local",
                    "Visitante",
                ],
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
                "🆚 Rival",
                options=opponents,
            )


        # ----------------------------------------------------
        # COLUMNA 2
        # ----------------------------------------------------

        with col2:

            minute = st.slider(
                "⏱️ Minuto del partido",
                min_value=1,
                max_value=90,
                value=60,
            )


            st.markdown(
                "**⚽ Marcador actual**"
            )


            score1, score2 = (
                st.columns(2)
            )


            with score1:

                america_goals = st.number_input(
                    "América",
                    min_value=0,
                    max_value=10,
                    value=0,
                    step=1,
                )


            with score2:

                opponent_goals = st.number_input(
                    opponent,
                    min_value=0,
                    max_value=10,
                    value=0,
                    step=1,
                )


        # ----------------------------------------------------
        # COLUMNA 3
        # ----------------------------------------------------

        with col3:

            possession_available_ui = False


            if (
                "america_possession_context"
                in
                simulator_data.columns
            ):

                possession_values = pd.to_numeric(
                    simulator_data[
                        "america_possession_context"
                    ],
                    errors="coerce",
                )


                possession_available_ui = (
                    possession_values
                    .notna()
                    .sum()
                    >=
                    30
                )


            if possession_available_ui:

                possession_choice = st.selectbox(
                    "🎯 Tendencia de posesión",
                    [
                        "América",
                        "Equilibrada",
                        "Rival",
                    ],
                    index=1,
                )

            else:

                possession_choice = (
                    "Equilibrada"
                )


                st.selectbox(
                    "🎯 Tendencia de posesión",
                    [
                        "No disponible"
                    ],
                    disabled=True,
                )


            current_state_display = st.selectbox(
                "🧭 Estado táctico actual",
                list(
                    STATE_DISPLAY.values()
                ),
            )


            current_state = (
                STATE_REVERSE[
                    current_state_display
                ]
            )


        # ====================================================
        # MARCADOR EXACTO -> DIFERENCIA QUE USA XGBOOST
        # ====================================================

        score_state = (
            int(america_goals)
            -
            int(opponent_goals)
        )


        score_label = (
            f"América {america_goals}"
            f" - "
            f"{opponent_goals} {opponent}"
        )


        if score_state > 0:

            score_situation = (
                f"América gana por "
                f"{score_state}"
            )

        elif score_state < 0:

            score_situation = (
                f"América pierde por "
                f"{abs(score_state)}"
            )

        else:

            score_situation = (
                "Empate"
            )


        # ====================================================
        # HISTORIAL TOTAL REAL
        # ====================================================

        total_historical_matches = (
            count_played_matches_vs_opponent(
                match_history,
                opponent,
            )
        )


        # ====================================================
        # SIMULACIÓN + XGBOOST
        # ====================================================

        try:

            (
                scenario_features,
                neighbors,
                similarity,
                diagnostic,
            ) = complete_scenario_context(

                simulator_data,

                venue=venue,

                opponent=opponent,

                minute=minute,

                score_state=score_state,

                possession_choice=
                    possession_choice,

                current_state=
                    current_state,

                n_neighbors=20,
            )


            # ------------------------------------------------
            # PREDICCIÓN REAL DEL MODELO
            # ------------------------------------------------

            dmatrix = xgb.DMatrix(
                scenario_features[
                    MODEL_FEATURES
                ],
                feature_names=
                    MODEL_FEATURES,
            )


            probabilities = np.asarray(
                model.predict(
                    dmatrix
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


            probs = probabilities[0]


            order = np.argsort(
                probs
            )[::-1]


            top1_index = int(
                order[0]
            )

            top2_index = int(
                order[1]
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
                probs[
                    top1_index
                ]
            )


            second_probability = float(
                probs[
                    top2_index
                ]
            )


            confidence_margin = (
                confidence
                -
                second_probability
            )


            # =================================================
            # ALERTA OPERATIVA
            #
            # Es una regla sobre el output del modelo.
            # =================================================

            if (
                confidence >= 0.60
                and
                confidence_margin >= 0.15
            ):

                alert = "ALTA"


            elif (
                confidence >= 0.45
                and
                confidence_margin >= 0.08
            ):

                alert = "MEDIA"


            else:

                alert = "VIGILAR"


            expected_change = (
                predicted_state
                !=
                current_state
            )


            if (
                alert == "ALTA"
                and
                expected_change
            ):

                decision = (
                    "PREPARAR CAMBIO AHORA"
                )


            elif (
                alert == "MEDIA"
                and
                expected_change
            ):

                decision = (
                    "PREPARAR SIGUIENTE FASE"
                )


            elif expected_change:

                decision = (
                    "MONITOREAR POSIBLE CAMBIO"
                )


            else:

                decision = (
                    "MANTENER PLAN ACTUAL"
                )


            # =================================================
            # EXPLICACIÓN REAL DEL XGBOOST
            # =================================================

            reasons = local_shap_reasons(
                model,
                scenario_features[
                    MODEL_FEATURES
                ],
                top1_index,
            )


            while len(reasons) < 3:

                reasons.append(
                    "Contexto táctico adicional"
                )


            coach_action = (
                coach_action_for_state(
                    predicted_state,
                    score_state,
                )
            )


            # =================================================
            # CONTEXTO
            # =================================================

            venue_text = (
                "🏠 LOCAL"
                if venue == "HOME"
                else "✈️ VISITANTE"
            )


            st.html(
                f"""
                <div class="context-row">

                    <span class="chip">
                        {venue_text}
                    </span>

                    <span class="chip">
                        vs {opponent}
                    </span>

                    <span class="chip">
                        {minute}'
                    </span>

                    <span class="chip">
                        {score_label}
                    </span>

                    <span class="chip">
                        {score_situation}
                    </span>

                    <span class="chip">
                        {state_display(current_state)}
                    </span>

                </div>
                """
            )


            # =================================================
            # MÉTRICAS PRINCIPALES
            # =================================================

            m1, m2, m3, m4 = (
                st.columns(4)
            )


            m1.metric(
                f"Partidos jugados vs {opponent}",
                total_historical_matches,
            )


            m2.metric(
                "Similitud del contexto",
                f"{similarity:.1f}%",
            )


            m3.metric(
                "Pronóstico",
                state_display(
                    predicted_state
                ),
            )


            m4.metric(
                "Confianza",
                safe_percent(
                    confidence
                ),
            )


            st.caption(
                "La similitud del contexto mide qué tan parecidos son "
                "los antecedentes históricos seleccionados. "
                "La confianza corresponde directamente a la probabilidad "
                "estimada por XGBoost."
            )


            st.divider()


            # =================================================
            # DECISIÓN
            # =================================================

            forecast_col, alert_col, coach_col = (
                st.columns(
                    [
                        1,
                        1,
                        1.7,
                    ]
                )
            )


            with forecast_col:

                st.html(
                    """
                    <div class="section-kicker">
                        Pronóstico a 5 minutos
                    </div>
                    """
                )


                st.html(
                    f"""
                    <div class="forecast">
                        {state_display(predicted_state)}
                    </div>
                    """
                )


                st.write(
                    "**Segunda posibilidad:**"
                )

                st.write(
                    state_display(
                        second_state
                    )
                )

                st.write(
                    safe_percent(
                        second_probability
                    )
                )


            with alert_col:

                st.html(
                    """
                    <div class="section-kicker">
                        Alerta táctica
                    </div>
                    """
                )


                if alert == "ALTA":

                    st.error(
                        "🔴 ALTA"
                    )

                elif alert == "MEDIA":

                    st.warning(
                        "🟠 MEDIA"
                    )

                else:

                    st.info(
                        "🔵 VIGILAR"
                    )


                st.write(
                    "**Decisión operativa**"
                )

                st.subheader(
                    decision
                )


            with coach_col:

                st.html(
                    """
                    <div class="section-kicker">
                        Recomendación al cuerpo técnico
                    </div>
                    """
                )


                st.info(
                    coach_action
                )


            # =================================================
            # EXPLICABILIDAD LOCAL
            # =================================================

            st.divider()


            st.html(
                """
                <div class="section-kicker">
                    Explicación del pronóstico
                </div>
                """
            )


            st.subheader(
                "¿Qué señales impulsan esta predicción?"
            )


            signal1, signal2, signal3 = (
                st.columns(3)
            )


            for column, number, reason in zip(
                [
                    signal1,
                    signal2,
                    signal3,
                ],
                [
                    "01",
                    "02",
                    "03",
                ],
                reasons,
            ):

                with column:

                    with st.container(
                        border=True
                    ):

                        st.caption(
                            f"SEÑAL {number}"
                        )

                        st.subheader(
                            reason
                        )


            st.caption(
                "Estas señales se obtienen mediante contribuciones SHAP "
                "del mismo XGBoost que produce el pronóstico."
            )


            # =================================================
            # DISTRIBUCIÓN DE PROBABILIDADES
            # =================================================

            st.divider()


            st.html(
                """
                <div class="section-kicker">
                    Distribución del pronóstico
                </div>
                """
            )


            probability_table = pd.DataFrame(
                {

                    "Estado": [
                        state_display(state)
                        for state
                        in STATE_LEVELS
                    ],

                    "Probabilidad":
                        probs * 100,
                }
            )


            probability_table = (
                probability_table
                .sort_values(
                    "Probabilidad",
                    ascending=True,
                )
            )


            fig_probs = px.bar(
                probability_table,
                x="Probabilidad",
                y="Estado",
                orientation="h",
                text="Probabilidad",
            )


            fig_probs.update_traces(
                marker_color=
                    AMERICA_YELLOW,

                texttemplate=
                    "%{text:.1f}%",

                textposition=
                    "outside",
            )


            fig_probs.update_layout(
                paper_bgcolor=
                    "rgba(0,0,0,0)",

                plot_bgcolor=
                    "rgba(0,0,0,0)",

                font_color=
                    "white",

                xaxis_title=
                    "Probabilidad (%)",

                yaxis_title=
                    "",

                showlegend=
                    False,

                height=
                    360,
            )


            st.plotly_chart(
                fig_probs,
                use_container_width=True,
            )


            # =================================================
            # TRANSPARENCIA
            # =================================================

            st.divider()


            with st.expander(
                "🔎 ¿Cómo construyó CoachLens este escenario?"
            ):

                transparency_table = pd.DataFrame(
                    {

                        "Dato": [

                            "Partidos jugados totales contra el rival",

                            "Partidos del rival disponibles en TRAIN + VALIDATION",

                            "Momentos históricos utilizados",

                            "Partidos distintos representados",

                            "Máximo de momentos por partido",

                            "Momentos exactos del rival entre los comparables",

                            "Minuto promedio de los comparables",

                            "Diferencia promedio de marcador",

                            "Localía utilizada como filtro",

                            "Estado táctico utilizado como filtro",

                            "Posesión disponible",

                            "Posesión utilizada como filtro",
                        ],

                        "Valor": [

                            total_historical_matches,

                            diagnostic[
                                "exact_opponent_matches"
                            ],

                            diagnostic[
                                "historical_moments"
                            ],

                            diagnostic[
                                "distinct_matches"
                            ],

                            2,

                            diagnostic[
                                "exact_opponent_moments"
                            ],

                            diagnostic[
                                "average_minute"
                            ],

                            diagnostic[
                                "average_score_state"
                            ],

                            diagnostic[
                                "venue_filter_used"
                            ],

                            diagnostic[
                                "state_filter_used"
                            ],

                            diagnostic[
                                "possession_available"
                            ],

                            diagnostic[
                                "possession_filter_used"
                            ],
                        ],
                    }
                )


                st.dataframe(
                    transparency_table,
                    hide_index=True,
                    use_container_width=True,
                )


                st.markdown(
                    "### Variables tácticas estimadas"
                )


                tactical_context = pd.DataFrame(
                    {

                        "Variable": [

                            "Dominio territorial",

                            "Balance de presión",

                            "Entradas al último tercio",

                            "Entradas al área",

                            "Pases progresivos",

                            "Altura de recuperación",

                            "Distancia al rival más cercano",

                            "Superioridad local",

                            "Apoyo de compañeros",
                        ],

                        "Valor estimado": [

                            scenario_features[
                                "america_field_tilt_last5"
                            ].iloc[0],

                            scenario_features[
                                "pressure_diff_last5"
                            ].iloc[0],

                            scenario_features[
                                "entries_final_third_diff_last5"
                            ].iloc[0],

                            scenario_features[
                                "entries_box_diff_last5"
                            ].iloc[0],

                            scenario_features[
                                "progressive_pass_diff_last5"
                            ].iloc[0],

                            scenario_features[
                                "recovery_height_diff_last5"
                            ].iloc[0],

                            scenario_features[
                                "america_nearest_opponent_distance_last5"
                            ].iloc[0],

                            scenario_features[
                                "america_local_superiority_10_last5"
                            ].iloc[0],

                            scenario_features[
                                "america_support_share_10_last5"
                            ].iloc[0],
                        ],
                    }
                )


                st.dataframe(
                    tactical_context.round(3),
                    hide_index=True,
                    use_container_width=True,
                )


                st.info(
                    "El historial total se utiliza únicamente como contexto "
                    "informativo. Para construir el escenario se usan solo "
                    "TRAIN + VALIDATION, evitando contaminar el conjunto TEST."
                )


        except Exception as error:

            st.error(
                f"Error en el simulador: {error}"
            )


# ============================================================
# 21. REVISIÓN HISTÓRICA
# ============================================================

with tab_replay:

    st.html(
        """
        <div class="section-kicker">
            Validación histórica
        </div>
        """
    )


    st.header(
        "Revisa una predicción histórica real"
    )


    if coach_all.empty:

        st.warning(
            "No se encontró coach_engine_all_test.csv."
        )


    else:

        replay = coach_all.copy()


        replay["match_id"] = (
            replay["match_id"]
            .astype(str)
            .str.replace(
                ".0",
                "",
                regex=False,
            )
        )


        match_table = (
            replay[
                [
                    "match_id",
                    "match_date",
                    "opponent",
                ]
            ]
            .drop_duplicates()
            .copy()
        )


        match_table["label"] = (
            match_table["match_date"]
            .astype(str)
            +
            " · América vs "
            +
            match_table["opponent"]
            .astype(str)
        )


        selected_match = st.selectbox(
            "Partido histórico",
            match_table[
                "label"
            ].tolist(),
        )


        selected_id = (
            match_table.loc[
                match_table["label"]
                ==
                selected_match,
                "match_id",
            ]
            .iloc[0]
        )


        match_data = replay[
            replay["match_id"]
            ==
            selected_id
        ].copy()


        minutes = sorted(
            pd.to_numeric(
                match_data["minute"],
                errors="coerce",
            )
            .dropna()
            .astype(int)
            .unique()
            .tolist()
        )


        selected_minute = st.select_slider(
            "Minuto",
            options=minutes,
            value=minutes[
                len(minutes) // 2
            ],
        )


        case = match_data[
            pd.to_numeric(
                match_data["minute"],
                errors="coerce",
            )
            ==
            selected_minute
        ].iloc[0]


        r1, r2, r3, r4 = (
            st.columns(4)
        )


        r1.metric(
            "Rival",
            case.get(
                "opponent",
                "N/D",
            ),
        )


        r2.metric(
            "Minuto",
            selected_minute,
        )


        r3.metric(
            "Estado actual",
            state_display(
                case.get(
                    "current_state",
                    "N/D",
                )
            ),
        )


        r4.metric(
            "Pronóstico",
            state_display(
                case.get(
                    "predicted_state",
                    "N/D",
                )
            ),
        )


        st.divider()


        c1, c2, c3 = (
            st.columns(3)
        )


        c1.metric(
            "Confianza",
            safe_percent(
                case.get(
                    "confidence",
                    0,
                )
            ),
        )


        c2.metric(
            "Estado real +5 min",
            state_display(
                case.get(
                    "actual_future_state",
                    "N/D",
                )
            ),
        )


        prediction_correct = (
            str(
                case.get(
                    "predicted_state",
                    "",
                )
            )
            ==
            str(
                case.get(
                    "actual_future_state",
                    "",
                )
            )
        )


        c3.metric(
            "Resultado",
            (
                "ACIERTO"
                if prediction_correct
                else "NO ACIERTO"
            ),
        )


# ============================================================
# 22. INTELIGENCIA TÁCTICA
# ============================================================

with tab_tactical:

    st.html(
        """
        <div class="section-kicker">
            Inteligencia táctica
        </div>
        """
    )


    st.header(
        "¿Cómo cambia América después de un gol?"
    )


    if not goal_summary.empty:

        metric_map = dict(
            zip(
                goal_summary["metric"],
                goal_summary["value"],
            )
        )


        g1, g2, g3, g4 = (
            st.columns(4)
        )


        g1.metric(
            "Goles anotados analizados",
            int(
                metric_map.get(
                    "America scoring events",
                    0,
                )
            ),
        )


        g2.metric(
            "Goles recibidos analizados",
            int(
                metric_map.get(
                    "America conceding events",
                    0,
                )
            ),
        )


        g3.metric(
            "Cambio táctico tras anotar",
            safe_percent(
                metric_map.get(
                    "Tactical change rate after scoring",
                    0,
                )
            ),
        )


        g4.metric(
            "Cambio táctico tras recibir",
            safe_percent(
                metric_map.get(
                    "Tactical change rate after conceding",
                    0,
                )
            ),
        )


    st.divider()


    if not goal_effects.empty:

        effects = goal_effects.copy()


        if (
            "ci_excludes_zero"
            in
            effects.columns
        ):

            significant = (
                effects[
                    "ci_excludes_zero"
                ]
                .astype(str)
                .str.lower()
                .isin(
                    [
                        "true",
                        "1",
                    ]
                )
            )

            effects = effects[
                significant
            ].copy()


        if not effects.empty:

            effects["label"] = (
                effects["goal_type"]
                .astype(str)
                .str.replace(
                    "AMERICA_",
                    "",
                    regex=False,
                )
                +
                " · "
                +
                effects["feature"]
                .astype(str)
            )


            fig_goal = px.bar(
                effects,
                x="mean_delta",
                y="label",
                orientation="h",
            )


            fig_goal.update_traces(
                marker_color=
                    AMERICA_YELLOW
            )


            fig_goal.update_layout(
                paper_bgcolor=
                    "rgba(0,0,0,0)",

                plot_bgcolor=
                    "rgba(0,0,0,0)",

                font_color=
                    "white",

                xaxis_title=
                    "Cambio promedio",

                yaxis_title=
                    "",
            )


            st.plotly_chart(
                fig_goal,
                use_container_width=True,
            )


            st.caption(
                "Estudio observacional alrededor de goles. "
                "No se interpreta como una estimación causal."
            )


# ============================================================
# 23. EXPLICABILIDAD GLOBAL
# ============================================================

with tab_why:

    st.html(
        """
        <div class="section-kicker">
            Inteligencia artificial explicable
        </div>
        """
    )


    st.header(
        "¿Qué variables utiliza más el modelo?"
    )


    if shap_global.empty:

        st.warning(
            "No se encontraron resultados SHAP."
        )


    else:

        shap_plot = (
            shap_global
            .sort_values(
                "mean_abs_shap",
                ascending=False,
            )
            .head(15)
            .copy()
        )


        shap_plot["label"] = (
            shap_plot["feature"]
            .map(
                FEATURE_NAMES
            )
            .fillna(
                shap_plot["feature"]
            )
        )


        fig_shap = px.bar(
            shap_plot.sort_values(
                "mean_abs_shap"
            ),
            x="mean_abs_shap",
            y="label",
            orientation="h",
        )


        fig_shap.update_traces(
            marker_color=
                AMERICA_YELLOW
        )


        fig_shap.update_layout(
            paper_bgcolor=
                "rgba(0,0,0,0)",

            plot_bgcolor=
                "rgba(0,0,0,0)",

            font_color=
                "white",

            xaxis_title=
                "Importancia SHAP promedio",

            yaxis_title=
                "",
        )


        st.plotly_chart(
            fig_shap,
            use_container_width=True,
        )


        st.info(
            "SHAP explica qué variables influyen en las predicciones "
            "del XGBoost. No implica que esas variables causen "
            "el cambio táctico."
        )


# ============================================================
# 24. EVIDENCIA DEL MODELO
# ============================================================

with tab_evidence:

    st.html(
        """
        <div class="section-kicker">
            Validación predictiva
        </div>
        """
    )


    st.header(
        "Evidencia del modelo XGBoost"
    )


    evidence = (
        calculate_model_evidence(
            coach_all
        )
    )


    e1, e2, e3, e4 = (
        st.columns(4)
    )


    e1.metric(
        "Exactitud Top-1",
        (
            safe_percent(
                evidence["top1"]
            )
            if evidence["top1"]
            is not None
            else "N/D"
        ),
    )


    e2.metric(
        "Exactitud Top-2",
        (
            safe_percent(
                evidence["top2"]
            )
            if evidence["top2"]
            is not None
            else "N/D"
        ),
    )


    e3.metric(
        "Baseline de persistencia",
        (
            safe_percent(
                evidence[
                    "persistence"
                ]
            )
            if evidence[
                "persistence"
            ]
            is not None
            else "N/D"
        ),
    )


    e4.metric(
        "Exactitud balanceada",
        (
            safe_percent(
                evidence[
                    "balanced"
                ]
            )
            if evidence[
                "balanced"
            ]
            is not None
            else "N/D"
        ),
    )


    st.caption(
        f"Métricas calculadas directamente sobre "
        f"{evidence['n']:,} observaciones del conjunto de prueba."
    )


    st.divider()


    if not alert_performance.empty:

        fig_alert = px.bar(
            alert_performance,
            x="alert_level",
            y="accuracy_pct",
            text="accuracy_pct",
        )


        fig_alert.update_traces(
            marker_color=
                AMERICA_YELLOW
        )


        fig_alert.update_layout(
            paper_bgcolor=
                "rgba(0,0,0,0)",

            plot_bgcolor=
                "rgba(0,0,0,0)",

            font_color=
                "white",

            xaxis_title=
                "Nivel de alerta",

            yaxis_title=
                "Exactitud (%)",
        )


        st.plotly_chart(
            fig_alert,
            use_container_width=True,
        )


    st.divider()


    st.markdown(
        """
### ¿Qué parte proviene realmente del modelo?

**Directamente del XGBoost**
- Estado táctico pronosticado.
- Probabilidad de cada uno de los cinco estados.
- Confianza del pronóstico.
- Segunda alternativa.
- Explicaciones SHAP.

**Construcción del escenario**
- Los comparables históricos estiman las variables Events y 360 que no introduce manualmente el usuario.
- El minuto, diferencia de marcador y estado táctico actual entran directamente al vector del modelo.

**Capa operativa**
- La alerta ALTA / MEDIA / VIGILAR utiliza umbrales sobre la confianza y el margen del modelo.
- La recomendación traduce el estado pronosticado a una acción comprensible para el cuerpo técnico.
- Esa recomendación no se presenta como una nueva predicción estadística.

**Control contra leakage**
- El simulador construye escenarios utilizando TRAIN + VALIDATION.
- TEST se conserva para evaluar el desempeño predictivo.
"""
    )


# ============================================================
# 25. FOOTER
# ============================================================

st.divider()


st.caption(
    "COACHLENS · CLUB AMÉRICA · "
    "XGBoost · StatsBomb Events + 360 · "
    "Inteligencia para decisiones tácticas"
)