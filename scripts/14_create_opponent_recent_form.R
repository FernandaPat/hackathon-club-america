library(data.table)

# ============================================================
# 1. LEER DATOS
# ============================================================

league_matches <- fread(
  "data/raw/america_male/matches_liga_mx.csv",
  encoding = "UTF-8"
)

america_matches <- fread(
  "data/processed/america_male/america_matches.csv",
  encoding = "UTF-8"
)

prematch <- fread(
  "data/processed/america_male/prematch_context.csv",
  encoding = "UTF-8"
)

cat("Liga MX matches:", nrow(league_matches), "\n")
cat("America matches:", nrow(america_matches), "\n")
cat("Prematch rows:", nrow(prematch), "\n")


# ============================================================
# 2. DETECTAR COLUMNAS DE EQUIPO
# ============================================================

home_col <- if (
  "home_team" %in% names(league_matches)
) {
  "home_team"
} else {
  "home_team.home_team_name"
}

away_col <- if (
  "away_team" %in% names(league_matches)
) {
  "away_team"
} else {
  "away_team.away_team_name"
}


# ============================================================
# 3. SOLO PARTIDOS YA JUGADOS
# ============================================================

league_matches[, match_date := as.IDate(match_date)]

league_played <- league_matches[
  !is.na(home_score) &
  !is.na(away_score) &
  !is.na(match_date)
]

cat(
  "Partidos jugados disponibles:",
  nrow(league_played),
  "\n"
)


# ============================================================
# 4. PASAR A FORMATO LARGO
# UNA FILA POR EQUIPO POR PARTIDO
# ============================================================

home_rows <- league_played[
  ,
  .(
    match_id,
    match_date,

    team = get(home_col),
    opponent_match = get(away_col),

    goals_for = home_score,
    goals_against = away_score,

    venue_match = "Home"
  )
]


away_rows <- league_played[
  ,
  .(
    match_id,
    match_date,

    team = get(away_col),
    opponent_match = get(home_col),

    goals_for = away_score,
    goals_against = home_score,

    venue_match = "Away"
  )
]


team_matches <- rbindlist(
  list(
    home_rows,
    away_rows
  )
)


# ============================================================
# 5. RESULTADO Y PUNTOS
# ============================================================

team_matches[
  ,
  result := fifelse(
    goals_for > goals_against,
    "W",
    fifelse(
      goals_for == goals_against,
      "D",
      "L"
    )
  )
]

team_matches[
  ,
  points := fifelse(
    result == "W",
    3,
    fifelse(
      result == "D",
      1,
      0
    )
  )
]

team_matches[
  ,
  goal_diff :=
    goals_for -
    goals_against
]


# ============================================================
# 6. ORDEN CRONOLÓGICO
# ============================================================

setorder(
  team_matches,
  team,
  match_date,
  match_id
)


# ============================================================
# 7. CREAR VARIABLES DEL PARTIDO ANTERIOR
# IMPORTANTE:
# shift() evita usar el partido actual
# ============================================================

team_matches[
  ,
  prev_points := shift(points),
  by = team
]

team_matches[
  ,
  prev_goals_for := shift(goals_for),
  by = team
]

team_matches[
  ,
  prev_goals_against := shift(goals_against),
  by = team
]

team_matches[
  ,
  prev_goal_diff := shift(goal_diff),
  by = team
]

team_matches[
  ,
  prev_win := shift(as.integer(result == "W")),
  by = team
]

team_matches[
  ,
  prev_draw := shift(as.integer(result == "D")),
  by = team
]

team_matches[
  ,
  prev_loss := shift(as.integer(result == "L")),
  by = team
]


# ============================================================
# 8. FORMA ÚLTIMO 1 PARTIDO
# ============================================================

team_matches[
  ,
  opponent_points_last1 := prev_points
]

team_matches[
  ,
  opponent_goals_for_last1 := prev_goals_for
]

team_matches[
  ,
  opponent_goals_against_last1 := prev_goals_against
]

team_matches[
  ,
  opponent_goal_diff_last1 := prev_goal_diff
]

team_matches[
  ,
  opponent_win_last1 := prev_win
]

team_matches[
  ,
  opponent_draw_last1 := prev_draw
]

team_matches[
  ,
  opponent_loss_last1 := prev_loss
]


# ============================================================
# 9. FORMA ÚLTIMOS 3 PARTIDOS
# ============================================================

team_matches[
  ,
  opponent_points_last3 :=
    frollsum(
      prev_points,
      n = 3,
      align = "right",
      fill = NA,
      na.rm = TRUE
    ),
  by = team
]

team_matches[
  ,
  opponent_goals_for_last3 :=
    frollsum(
      prev_goals_for,
      n = 3,
      align = "right",
      fill = NA,
      na.rm = TRUE
    ),
  by = team
]

team_matches[
  ,
  opponent_goals_against_last3 :=
    frollsum(
      prev_goals_against,
      n = 3,
      align = "right",
      fill = NA,
      na.rm = TRUE
    ),
  by = team
]

team_matches[
  ,
  opponent_goal_diff_last3 :=
    opponent_goals_for_last3 -
    opponent_goals_against_last3
]

team_matches[
  ,
  opponent_wins_last3 :=
    frollsum(
      prev_win,
      n = 3,
      align = "right",
      fill = NA,
      na.rm = TRUE
    ),
  by = team
]

team_matches[
  ,
  opponent_draws_last3 :=
    frollsum(
      prev_draw,
      n = 3,
      align = "right",
      fill = NA,
      na.rm = TRUE
    ),
  by = team
]

team_matches[
  ,
  opponent_losses_last3 :=
    frollsum(
      prev_loss,
      n = 3,
      align = "right",
      fill = NA,
      na.rm = TRUE
    ),
  by = team
]


# ============================================================
# 10. FORMA ÚLTIMOS 5 PARTIDOS
# ============================================================

team_matches[
  ,
  opponent_points_last5 :=
    frollsum(
      prev_points,
      n = 5,
      align = "right",
      fill = NA,
      na.rm = TRUE
    ),
  by = team
]

team_matches[
  ,
  opponent_goals_for_last5 :=
    frollsum(
      prev_goals_for,
      n = 5,
      align = "right",
      fill = NA,
      na.rm = TRUE
    ),
  by = team
]

team_matches[
  ,
  opponent_goals_against_last5 :=
    frollsum(
      prev_goals_against,
      n = 5,
      align = "right",
      fill = NA,
      na.rm = TRUE
    ),
  by = team
]

team_matches[
  ,
  opponent_goal_diff_last5 :=
    opponent_goals_for_last5 -
    opponent_goals_against_last5
]

team_matches[
  ,
  opponent_wins_last5 :=
    frollsum(
      prev_win,
      n = 5,
      align = "right",
      fill = NA,
      na.rm = TRUE
    ),
  by = team
]

team_matches[
  ,
  opponent_draws_last5 :=
    frollsum(
      prev_draw,
      n = 5,
      align = "right",
      fill = NA,
      na.rm = TRUE
    ),
  by = team
]

team_matches[
  ,
  opponent_losses_last5 :=
    frollsum(
      prev_loss,
      n = 5,
      align = "right",
      fill = NA,
      na.rm = TRUE
    ),
  by = team
]


# ============================================================
# 11. FORMA PONDERADA POR RECENCIA
#
# Pesos:
# último partido       1.00
# hace 2 partidos      0.75
# hace 3 partidos      0.50
# hace 4 partidos      0.30
# hace 5 partidos      0.15
# ============================================================

weights <- c(
  0.15,
  0.30,
  0.50,
  0.75,
  1.00
)

weighted_roll <- function(x) {

  if (length(x) < 5 || all(is.na(x))) {
    return(NA_real_)
  }

  valid <- !is.na(x)

  if (!any(valid)) {
    return(NA_real_)
  }

  sum(
    x[valid] * weights[valid]
  ) /
    sum(weights[valid])
}


team_matches[
  ,
  opponent_points_weighted :=
    frollapply(
      prev_points,
      n = 5,
      FUN = weighted_roll,
      align = "right",
      fill = NA
    ),
  by = team
]

team_matches[
  ,
  opponent_goals_for_weighted :=
    frollapply(
      prev_goals_for,
      n = 5,
      FUN = weighted_roll,
      align = "right",
      fill = NA
    ),
  by = team
]

team_matches[
  ,
  opponent_goals_against_weighted :=
    frollapply(
      prev_goals_against,
      n = 5,
      FUN = weighted_roll,
      align = "right",
      fill = NA
    ),
  by = team
]

team_matches[
  ,
  opponent_goal_diff_weighted :=
    opponent_goals_for_weighted -
    opponent_goals_against_weighted
]


# ============================================================
# 12. NÚMERO DE PARTIDOS ANTERIORES DISPONIBLES
# ============================================================

team_matches[
  ,
  matches_before :=
    seq_len(.N) - 1,
  by = team
]

team_matches[
  ,
  opponent_history_available :=
    pmin(
      matches_before,
      5
    )
]


# ============================================================
# 13. TABLA FINAL PRE-PARTIDO
# ============================================================

recent_form <- team_matches[
  ,
  .(
    match_id,
    match_date,
    opponent = team,

    opponent_history_available,

    # ---- Last 1
    opponent_points_last1,
    opponent_goals_for_last1,
    opponent_goals_against_last1,
    opponent_goal_diff_last1,
    opponent_win_last1,
    opponent_draw_last1,
    opponent_loss_last1,

    # ---- Last 3
    opponent_points_last3,
    opponent_goals_for_last3,
    opponent_goals_against_last3,
    opponent_goal_diff_last3,
    opponent_wins_last3,
    opponent_draws_last3,
    opponent_losses_last3,

    # ---- Last 5
    opponent_points_last5,
    opponent_goals_for_last5,
    opponent_goals_against_last5,
    opponent_goal_diff_last5,
    opponent_wins_last5,
    opponent_draws_last5,
    opponent_losses_last5,

    # ---- Recency weighted
    opponent_points_weighted,
    opponent_goals_for_weighted,
    opponent_goals_against_weighted,
    opponent_goal_diff_weighted
  )
]


# ============================================================
# 14. PREPARAR PARTIDOS DE AMÉRICA
# ============================================================

america_matches[
  ,
  match_date := as.IDate(match_date)
]

america_keys <- unique(
  america_matches[
    ,
    .(
      match_id,
      match_date,
      opponent
    )
  ]
)


# ============================================================
# 15. UNIR FORMA AL PARTIDO DE AMÉRICA
# ============================================================

america_recent_form <- merge(
  america_keys,
  recent_form,
  by = c(
    "match_id",
    "match_date",
    "opponent"
  ),
  all.x = TRUE
)


# ============================================================
# 16. VALIDACIÓN
# ============================================================

cat("\n=====================================\n")
cat("MULTI-HORIZON RECENT FORM\n")
cat("=====================================\n")

cat(
  "Partidos América:",
  nrow(america_recent_form),
  "\n"
)

cat(
  "Con al menos 1 partido histórico:",
  sum(
    america_recent_form$opponent_history_available >= 1,
    na.rm = TRUE
  ),
  "\n"
)

cat(
  "Con al menos 3 partidos históricos:",
  sum(
    america_recent_form$opponent_history_available >= 3,
    na.rm = TRUE
  ),
  "\n"
)

cat(
  "Con 5 partidos históricos:",
  sum(
    america_recent_form$opponent_history_available >= 5,
    na.rm = TRUE
  ),
  "\n"
)


# ============================================================
# 17. UNIR AL DATASET MINUTO A MINUTO
# ============================================================

form_cols <- setdiff(
  names(america_recent_form),
  c(
    "match_date",
    "opponent"
  )
)

form_context <- america_recent_form[
  ,
  ..form_cols
]

final_context <- merge(
  prematch,
  form_context,
  by = "match_id",
  all.x = TRUE
)


# ============================================================
# 18. VERIFICAR QUE NO SE DUPLICARON FILAS
# ============================================================

cat(
  "Filas antes:",
  nrow(prematch),
  "\n"
)

cat(
  "Filas después:",
  nrow(final_context),
  "\n"
)


# ============================================================
# 19. GUARDAR
# ============================================================

fwrite(
  final_context,
  "data/processed/america_male/model_context.csv",
  bom = TRUE
)

cat(
  "\nmodel_context.csv guardado correctamente\n"
)













final_context[
  !is.na(opponent_points_last3),
  .(
    match_id,
    match_date,
    opponent,

    opponent_points_last1,
    opponent_points_last3,
    opponent_points_last5,

    opponent_goal_diff_last1,
    opponent_goal_diff_last3,
    opponent_goal_diff_last5,

    opponent_points_weighted,
    opponent_history_available
  )
][1:30]