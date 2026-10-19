library(data.table)

# ============================================================
# 1. LEER DATASET DE 5 MIN
# ============================================================

dt5 <- fread(
  "data/processed/america_male/model_with_tactical_states.csv",
  encoding = "UTF-8"
)

# ============================================================
# 2. LEER DATASET DE 3 MIN
# ============================================================

dt3 <- fread(
  "data/processed/america_male/model_with_tactical_states_3min.csv",
  encoding = "UTF-8"
)

setorder(dt5, match_id, minute)
setorder(dt3, match_id, minute)


# ============================================================
# 3. RESUMEN GENERAL
# ============================================================

general_summary <- data.table(

  dataset = c(
    "5 minutes",
    "3 minutes"
  ),

  rows = c(
    nrow(dt5),
    nrow(dt3)
  ),

  columns = c(
    ncol(dt5),
    ncol(dt3)
  ),

  matches = c(
    uniqueN(dt5$match_id),
    uniqueN(dt3$match_id)
  )
)

View(general_summary)


# ============================================================
# 4. DUPLICADOS MATCH-MINUTE
# ============================================================

duplicates_5 <- dt5[
  ,
  .N,
  by = .(
    match_id,
    minute
  )
][
  N > 1
]

duplicates_3 <- dt3[
  ,
  .N,
  by = .(
    match_id,
    minute
  )
][
  N > 1
]

duplicate_summary <- data.table(

  dataset = c(
    "5 minutes",
    "3 minutes"
  ),

  duplicated_match_minutes = c(
    nrow(duplicates_5),
    nrow(duplicates_3)
  )
)

View(duplicate_summary)


# ============================================================
# 5. COBERTURA DE ESTADOS TÁCTICOS
# ============================================================

state_coverage_5 <- dt5[
  ,
  .(
    observations = .N,

    classified =
      sum(
        !is.na(tactical_state)
      ),

    missing =
      sum(
        is.na(tactical_state)
      ),

    coverage_pct =
      round(
        mean(
          !is.na(tactical_state)
        ) * 100,
        2
      )
  )
]

state_coverage_3 <- dt3[
  ,
  .(
    observations = .N,

    classified =
      sum(
        !is.na(tactical_state_3min)
      ),

    missing =
      sum(
        is.na(tactical_state_3min)
      ),

    coverage_pct =
      round(
        mean(
          !is.na(tactical_state_3min)
        ) * 100,
        2
      )
  )
]

state_coverage <- rbind(
  data.table(
    window = "5 minutes",
    state_coverage_5
  ),

  data.table(
    window = "3 minutes",
    state_coverage_3
  )
)

View(state_coverage)


# ============================================================
# 6. DISTRIBUCIÓN DE ESTADOS 5 MIN
# ============================================================

distribution_5 <- dt5[
  !is.na(tactical_state),
  .N,
  by = tactical_state
]

distribution_5[
  ,
  percentage :=
    round(
      N / sum(N) * 100,
      2
    )
]

setorder(
  distribution_5,
  -N
)

View(distribution_5)


# ============================================================
# 7. DISTRIBUCIÓN DE ESTADOS 3 MIN
# ============================================================

distribution_3 <- dt3[
  !is.na(tactical_state_3min),
  .N,
  by = tactical_state_3min
]

distribution_3[
  ,
  percentage :=
    round(
      N / sum(N) * 100,
      2
    )
]

setorder(
  distribution_3,
  -N
)

View(distribution_3)


# ============================================================
# 8. MISSINGNESS VARIABLES IMPORTANTES
# ============================================================

important_features <- c(

  "america_field_tilt_last5",

  "pressure_diff_last5",

  "xg_diff_last5",

  "obv_diff_last5",

  "progressive_pass_diff_last5",

  "entries_final_third_diff_last5",

  "entries_box_diff_last5",

  "recovery_height_diff_last5",

  "america_pressure_share_5_last5",

  "america_local_superiority_10_last5",

  "america_nearest_opponent_distance_last5",

  "opponent_points_last5",

  "opponent_points_weighted"
)

important_features <- important_features[
  important_features %in%
    names(dt5)
]


missing_summary <- rbindlist(
  lapply(
    important_features,
    function(col) {

      data.table(

        variable = col,

        missing =
          sum(
            is.na(dt5[[col]])
          ),

        missing_pct =
          round(
            mean(
              is.na(dt5[[col]])
            ) * 100,
            2
          )
      )
    }
  )
)

setorder(
  missing_summary,
  -missing_pct
)

View(missing_summary)


# ============================================================
# 9. COBERTURA 360
# ============================================================

coverage_360 <- dt5[
  ,
  .(

    total_minutes = .N,

    america_with_360 =
      sum(
        america_360_events > 0,
        na.rm = TRUE
      ),

    opponent_with_360 =
      sum(
        opponent_360_events > 0,
        na.rm = TRUE
      ),

    america_coverage_pct =
      round(
        mean(
          america_360_events > 0,
          na.rm = TRUE
        ) * 100,
        2
      ),

    opponent_coverage_pct =
      round(
        mean(
          opponent_360_events > 0,
          na.rm = TRUE
        ) * 100,
        2
      )
  )
]

View(coverage_360)


# ============================================================
# 10. PARTIDOS CON MENOR COBERTURA 360
# ============================================================

coverage_by_match <- dt5[
  ,
  .(

    minutes = .N,

    america_360_pct =
      round(
        mean(
          america_360_events > 0,
          na.rm = TRUE
        ) * 100,
        2
      ),

    opponent_360_pct =
      round(
        mean(
          opponent_360_events > 0,
          na.rm = TRUE
        ) * 100,
        2
      )

  ),
  by = .(
    match_id,
    opponent
  )
]

setorder(
  coverage_by_match,
  america_360_pct
)

View(coverage_by_match)


# ============================================================
# 11. REVISAR FECHAS
# ============================================================

if (
  "match_date" %in%
  names(dt5)
) {

  dt5[
    ,
    match_date :=
      as.IDate(
        match_date
      )
  ]

  date_summary <- dt5[
    ,
    .(
      first_match =
        min(
          match_date,
          na.rm = TRUE
        ),

      last_match =
        max(
          match_date,
          na.rm = TRUE
        ),

      matches =
        uniqueN(
          match_id
        )
    )
  ]

  View(date_summary)
}


# ============================================================
# 12. PARTIDOS POR TEMPORADA
# ============================================================

matches_by_season <- unique(
  dt5[
    ,
    .(
      match_id,
      season,
      opponent
    )
  ]
)[
  ,
  .(
    matches = .N
  ),
  by = season
]

View(matches_by_season)


# ============================================================
# 13. VALIDAR GAME STATE
# ============================================================

game_state_check <- dt5[
  ,
  .N,
  by = game_state
][
  order(game_state)
]

View(game_state_check)


# ============================================================
# 14. SCORE / GAME STATE CONSISTENCY
# ============================================================

if (
  all(
    c(
      "america_score",
      "opponent_score"
    ) %in%
      names(dt5)
  )
) {

  score_check <- dt5[
    ,
    .(
      match_id,
      minute,
      america_score,
      opponent_score,
      game_state
    )
  ]

  View(score_check)
}


# ============================================================
# 15. GUARDAR RESUMEN DE AUDITORÍA
# ============================================================

fwrite(
  missing_summary,
  "outputs/models/audit_missing_values.csv",
  bom = TRUE
)

fwrite(
  coverage_by_match,
  "outputs/models/audit_360_coverage.csv",
  bom = TRUE
)

cat("\n=====================================\n")
cat("QUALITY AUDIT COMPLETADA\n")
cat("=====================================\n")

cat("5min rows:", nrow(dt5), "\n")
cat("3min rows:", nrow(dt3), "\n")

cat(
  "Duplicados 5min:",
  nrow(duplicates_5),
  "\n"
)

cat(
  "Duplicados 3min:",
  nrow(duplicates_3),
  "\n"
)

check5 <- fread(
  "data/processed/america_male/model_with_tactical_states.csv",
  encoding = "UTF-8",
  na.strings = c("", "NA")
)

check5_distribution <- check5[
  !is.na(tactical_state),
  .N,
  by = tactical_state
]

check5_distribution[
  ,
  percentage := round(
    N / sum(N) * 100,
    2
  )
]

View(check5_distribution)