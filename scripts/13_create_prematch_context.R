library(data.table)

# ============================================================
# 1. LEER DATOS
# ============================================================

dt <- fread(
  "data/processed/america_male/rolling_features.csv",
  encoding = "UTF-8"
)

team_stats <- fread(
  "data/processed/america_male/team_season_stats.csv",
  encoding = "UTF-8"
)

cat("Rolling rows:", nrow(dt), "\n")
cat("Team-season rows:", nrow(team_stats), "\n")


# ============================================================
# 2. CREAR AÑO INICIAL DE TEMPORADA
#
# Ejemplo:
# 2025/2026 -> 2025
# ============================================================

dt[
  ,
  season_start := as.integer(
    substr(season, 1, 4)
  )
]

team_stats[
  ,
  season_start := as.integer(
    substr(season_name, 1, 4)
  )
]


# ============================================================
# 3. EL PERFIL DE UNA TEMPORADA SE USARÁ
#    PARA PREDECIR LA SIGUIENTE
#
# Ejemplo:
# Tigres 2025/26 -> contexto para 2026/27
# ============================================================

team_stats[
  ,
  prediction_season_start :=
    season_start + 1
]


# ============================================================
# 4. QUEDARNOS CON VARIABLES TÁCTICAS
# ============================================================

id_cols <- c(
  "team_name",
  "prediction_season_start"
)

feature_cols <- grep(
  "^team_season_",
  names(team_stats),
  value = TRUE
)

profile <- team_stats[
  ,
  c(
    id_cols,
    feature_cols
  ),
  with = FALSE
]


# ============================================================
# 5. RENOMBRAR
# ============================================================

setnames(
  profile,
  "team_name",
  "opponent"
)

feature_cols_profile <- grep(
  "^team_season_",
  names(profile),
  value = TRUE
)

setnames(
  profile,
  feature_cols_profile,
  paste0(
    "opponent_previous_",
    feature_cols_profile
  )
)


# ============================================================
# 6. JOIN SIN LEAKAGE
# ============================================================

dt_prematch <- merge(
  dt,
  profile,
  by.x = c(
    "opponent",
    "season_start"
  ),
  by.y = c(
    "opponent",
    "prediction_season_start"
  ),
  all.x = TRUE
)


# ============================================================
# 7. VALIDAR
# ============================================================

cat(
  "Filas antes:",
  nrow(dt),
  "\n"
)

cat(
  "Filas después:",
  nrow(dt_prematch),
  "\n"
)

context_cols <- grep(
  "^opponent_previous_team_season_",
  names(dt_prematch),
  value = TRUE
)

cat(
  "Variables históricas agregadas:",
  length(context_cols),
  "\n"
)


# ============================================================
# 8. GUARDAR
# ============================================================

fwrite(
  dt_prematch,
  "data/processed/america_male/prematch_context.csv",
  bom = TRUE
)

cat(
  "\nprematch_context.csv guardado correctamente\n"
)




