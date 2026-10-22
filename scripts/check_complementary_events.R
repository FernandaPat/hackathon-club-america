# ============================================================
# COACHLENS
# VALIDACION DE SUSTITUCIONES Y BALON PARADO
# ============================================================

options(
  stringsAsFactors = FALSE,
  scipen = 999
)

cat("\n")
cat("============================================================\n")
cat(" COACHLENS - VALIDACION DE EVENTOS\n")
cat("============================================================\n\n")


# ============================================================
# LOCALIZAR RAIZ DEL PROYECTO
# ============================================================

args <- commandArgs(
  trailingOnly = FALSE
)

file_arg <- grep(
  "^--file=",
  args,
  value = TRUE
)

if (length(file_arg) > 0) {

  script_path <- normalizePath(
    sub(
      "^--file=",
      "",
      file_arg[1]
    )
  )

  script_dir <- dirname(
    script_path
  )

  project_root <- normalizePath(
    file.path(
      script_dir,
      ".."
    )
  )

} else {

  project_root <- normalizePath(
    ".."
  )

}

cat(
  "Proyecto:",
  project_root,
  "\n\n"
)


# ============================================================
# RUTAS
# ============================================================

male_path <- file.path(
  project_root,
  "data",
  "processed",
  "america_male",
  "events_america_clean.csv"
)

female_path <- file.path(
  project_root,
  "data",
  "processed",
  "america_female",
  "events_america_female_clean.csv"
)


# ============================================================
# FUNCIONES
# ============================================================

read_events <- function(
  path,
  label
) {

  cat("\n")
  cat("------------------------------------------------------------\n")
  cat(label, "\n")
  cat("------------------------------------------------------------\n")

  cat(
    "Archivo:",
    path,
    "\n"
  )

  if (!file.exists(path)) {

    cat(
      "ERROR: archivo no encontrado.\n"
    )

    return(NULL)
  }

  cat(
    "Leyendo...\n"
  )

  data <- read.csv(
    path,
    check.names = FALSE,
    stringsAsFactors = FALSE
  )

  cat(
    "Filas:",
    format(
      nrow(data),
      big.mark = ","
    ),
    "\n"
  )

  cat(
    "Columnas:",
    ncol(data),
    "\n"
  )

  return(
    data
  )
}


show_values <- function(
  data,
  column,
  n = 20
) {

  cat(
    "\n--- ",
    column,
    " ---\n",
    sep = ""
  )

  if (!column %in% names(data)) {

    cat(
      "COLUMNA NO DISPONIBLE\n"
    )

    return(
      invisible(NULL)
    )
  }

  values <- as.character(
    data[[column]]
  )

  values <- values[
    !is.na(values)
    &
    trimws(values) != ""
  ]

  if (length(values) == 0) {

    cat(
      "SIN VALORES\n"
    )

    return(
      invisible(NULL)
    )
  }

  counts <- sort(
    table(values),
    decreasing = TRUE
  )

  print(
    head(
      counts,
      n
    )
  )

  invisible(
    counts
  )
}


analyse_events <- function(
  data,
  label
) {

  if (is.null(data)) {
    return(NULL)
  }

  cat("\n\n")
  cat("============================================================\n")
  cat(" ANALISIS ", label, "\n", sep = "")
  cat("============================================================\n")


  # ==========================================================
  # COLUMNAS RELEVANTES
  # ==========================================================

  relevant_columns <- names(data)[
    grepl(
      paste(
        c(
          "match",
          "minute",
          "team",
          "player",
          "type",
          "substitution",
          "replacement",
          "pass",
          "shot",
          "outcome",
          "pattern",
          "corner",
          "free",
          "set_piece"
        ),
        collapse = "|"
      ),
      names(data),
      ignore.case = TRUE
    )
  ]

  cat("\nCOLUMNAS RELEVANTES:\n")

  print(
    relevant_columns
  )


  # ==========================================================
  # TIPOS DE EVENTO
  # ==========================================================

  cat("\n")
  cat("============================================================\n")
  cat(" TIPOS DE EVENTO\n")
  cat("============================================================\n")

  show_values(
    data,
    "type.name",
    50
  )


  # ==========================================================
  # SUSTITUCIONES
  # ==========================================================

  cat("\n")
  cat("============================================================\n")
  cat(" SUSTITUCIONES\n")
  cat("============================================================\n")

  substitution_ok <- FALSE

  if ("type.name" %in% names(data)) {

    substitutions <- data[
      data[["type.name"]] == "Substitution",
      ,
      drop = FALSE
    ]

    cat(
      "Total sustituciones:",
      nrow(substitutions),
      "\n"
    )

    if (nrow(substitutions) > 0) {

      substitution_ok <- TRUE

      substitution_columns <- c(
        "match_id",
        "minute",
        "second",
        "team.name",
        "player.name",
        "substitution.replacement.name",
        "substitution.outcome.name"
      )

      substitution_columns <- substitution_columns[
        substitution_columns %in% names(substitutions)
      ]

      cat(
        "\nColumnas disponibles:\n"
      )

      print(
        substitution_columns
      )

      if (length(substitution_columns) > 0) {

        cat(
          "\nPrimeras 10 sustituciones:\n"
        )

        print(
          head(
            substitutions[
              ,
              substitution_columns,
              drop = FALSE
            ],
            10
          )
        )
      }
    }

  } else {

    cat(
      "No existe type.name.\n"
    )
  }


  # ==========================================================
  # BALON PARADO
  # ==========================================================

  cat("\n")
  cat("============================================================\n")
  cat(" BALON PARADO\n")
  cat("============================================================\n")


  show_values(
    data,
    "set_piece_phase",
    30
  )

  show_values(
    data,
    "play_pattern.name",
    30
  )


  # Buscar columnas pass.type
  pass_type_columns <- names(data)[
    grepl(
      "pass.*type",
      names(data),
      ignore.case = TRUE
    )
  ]

  cat(
    "\nColumnas de tipo de pase:\n"
  )

  print(
    pass_type_columns
  )

  if (length(pass_type_columns) > 0) {

    for (column in pass_type_columns) {

      show_values(
        data,
        column,
        30
      )
    }
  }


  # Buscar columnas shot.type
  shot_type_columns <- names(data)[
    grepl(
      "shot.*type",
      names(data),
      ignore.case = TRUE
    )
  ]

  cat(
    "\nColumnas de tipo de tiro:\n"
  )

  print(
    shot_type_columns
  )

  if (length(shot_type_columns) > 0) {

    for (column in shot_type_columns) {

      show_values(
        data,
        column,
        30
      )
    }
  }


  # ==========================================================
  # BUSQUEDA TEXTUAL DE CORNER / FREE KICK / SET PIECE
  # ==========================================================

  cat("\n")
  cat("============================================================\n")
  cat(" REFERENCIAS DE BALON PARADO\n")
  cat("============================================================\n")


  candidate_columns <- names(data)[
    grepl(
      "type|pattern|phase|pass|shot",
      names(data),
      ignore.case = TRUE
    )
  ]


  set_piece_hits <- list()


  for (column in candidate_columns) {

    values <- as.character(
      data[[column]]
    )

    hits <- grepl(
      "corner|free kick|free_kick|set piece|set_piece",
      values,
      ignore.case = TRUE
    )

    hits[
      is.na(hits)
    ] <- FALSE

    if (any(hits)) {

      found_values <- unique(
        values[
          hits
          &
          !is.na(values)
        ]
      )

      set_piece_hits[[column]] <- found_values
    }
  }


  if (length(set_piece_hits) == 0) {

    cat(
      "No se encontraron referencias textuales.\n"
    )

  } else {

    for (column in names(set_piece_hits)) {

      cat(
        "\nColumna:",
        column,
        "\n"
      )

      print(
        set_piece_hits[[column]]
      )
    }
  }


  # ==========================================================
  # VALIDACION AUTOMATICA
  # ==========================================================

  set_piece_ok <- FALSE


  if ("set_piece_phase" %in% names(data)) {

    values <- as.character(
      data[["set_piece_phase"]]
    )

    valid <- (
      !is.na(values)
      &
      trimws(values) != ""
    )

    if (any(valid)) {

      set_piece_ok <- TRUE
    }
  }


  if (
    !set_piece_ok
    &&
    length(set_piece_hits) > 0
  ) {

    set_piece_ok <- TRUE
  }


  cat("\n")
  cat("============================================================\n")
  cat(" VALIDACION AUTOMATICA - ", label, "\n", sep = "")
  cat("============================================================\n")


  cat(
    "Sustituciones disponibles: ",
    ifelse(
      substitution_ok,
      "SI",
      "NO"
    ),
    "\n",
    sep = ""
  )


  cat(
    "Balon parado identificable: ",
    ifelse(
      set_piece_ok,
      "SI",
      "NO"
    ),
    "\n",
    sep = ""
  )


  cat(
    "player.name disponible: ",
    ifelse(
      "player.name" %in% names(data),
      "SI",
      "NO"
    ),
    "\n",
    sep = ""
  )


  cat(
    "team.name disponible: ",
    ifelse(
      "team.name" %in% names(data),
      "SI",
      "NO"
    ),
    "\n",
    sep = ""
  )


  cat(
    "match_id disponible: ",
    ifelse(
      "match_id" %in% names(data),
      "SI",
      "NO"
    ),
    "\n",
    sep = ""
  )


  cat("\n")
}


# ============================================================
# EJECUTAR
# ============================================================

male <- read_events(
  male_path,
  "VARONIL"
)

analyse_events(
  male,
  "VARONIL"
)


female <- read_events(
  female_path,
  "FEMENIL"
)

analyse_events(
  female,
  "FEMENIL"
)


cat("\n")
cat("============================================================\n")
cat(" FIN DE VALIDACION\n")
cat("============================================================\n\n")