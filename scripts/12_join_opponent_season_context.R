library(data.table)

# ============================================================
# 1. LEER ROLLING FEATURES
# ============================================================

dt <- fread(
  "data/processed/america_male/rolling_features.csv",
  encoding = "UTF-8"
)

# ============================================================
# 2. LEER TEAM SEASON STATS
# ============================================================

team_stats <- fread(
  "data/processed/america_male/team_season_stats.csv",
  encoding = "UTF-8"
)

# ============================================================
# 3. SELECCIONAR VARIABLES ÚTILES
# ============================================================

context_cols <- c(
  "team_name",
  "season_name",

  "team_season_possession",
  "team_season_directness",
  "team_season_pace_towards_goal",
  "team_season_ppda",
  "team_season_aggression",

  "team_season_pressures_pg",
  "team_season_counterpressures_pg",
  "team_season_pressure_regains_pg",

  "team_season_deep_progressions_pg",
  "team_season_deep_completions_pg",

  "team_season_np_xg_pg",
  "team_season_np_xg_conceded_pg",

  "team_season_shots_pg",
  "team_season_shots_conceded_pg",

  "team_season_obv_pg",
  "team_season_obv_conceded_pg",

  "team_season_transition_possession_rate",
  "team_season_time_in_transition_pg",
  "team_season_transition_xg_pg",

  "team_season_avg_team_possession_directness"
)

context_cols <- context_cols[
  context_cols %in% names(team_stats)
]

team_context <- unique(
  team_stats[
    ,
    ..context_cols
  ]
)

# ============================================================
# 4. RENOMBRAR PARA QUE QUEDE CLARO QUE ES DEL RIVAL
# ============================================================

setnames(
  team_context,
  old = "team_name",
  new = "opponent"
)

setnames(
  team_context,
  old = "season_name",
  new = "season"
)

otras <- setdiff(
  names(team_context),
  c("opponent", "season")
)

setnames(
  team_context,
  otras,
  paste0("opponent_", otras)
)

# ============================================================
# 5. UNIR CON NUESTRA TABLA DE AMÉRICA
# ============================================================

dt_context <- merge(
  dt,
  team_context,
  by = c(
    "opponent",
    "season"
  ),
  all.x = TRUE
)

# ============================================================
# 6. REVISAR MATCHING
# ============================================================

cat(
  "Filas originales:",
  nrow(dt),
  "\n"
)

cat(
  "Filas después del join:",
  nrow(dt_context),
  "\n"
)

cat(
  "Filas con contexto rival:",
  sum(!is.na(dt_context$opponent_team_season_possession)),
  "\n"
)

cat(
  "Filas sin contexto rival:",
  sum(is.na(dt_context$opponent_team_season_possession)),
  "\n"
)

# ============================================================
# 7. GUARDAR
# ============================================================

fwrite(
  dt_context,
  "data/processed/america_male/rolling_features_with_opponent_context.csv",
  bom = TRUE
)

cat(
  "\nContexto de rival unido correctamente\n"
)


dt_context[
  !is.na(opponent_team_season_possession),
  .(
    opponent,
    season,
    opponent_team_season_possession,
    opponent_team_season_ppda,
    opponent_team_season_np_xg_pg,
    opponent_team_season_obv_pg
  )
][1:20]