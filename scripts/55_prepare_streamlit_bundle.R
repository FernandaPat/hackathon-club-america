# ============================================================
# 55_prepare_streamlit_bundle.R
#
# PREPARA LOS ARCHIVOS NECESARIOS PARA STREAMLIT
# ============================================================

library(data.table)


# ============================================================
# 1. RUTAS
# ============================================================

project_root <- normalizePath(
  ".",
  winslash = "/",
  mustWork = TRUE
)

bundle_dir <- file.path(
  project_root,
  "streamlit_bundle"
)

dashboard_data_dir <- file.path(
  bundle_dir,
  "dashboard",
  "data"
)

models_dir <- file.path(
  bundle_dir,
  "outputs",
  "models"
)

female_models_dir <- file.path(
  models_dir,
  "female",
  "change_model"
)


# ============================================================
# 2. CREAR CARPETAS
# ============================================================

dirs_to_create <- c(
  bundle_dir,
  dashboard_data_dir,
  models_dir,
  female_models_dir
)

for (dir_path in dirs_to_create) {

  if (!dir.exists(dir_path)) {

    dir.create(
      dir_path,
      recursive = TRUE
    )
  }
}


# ============================================================
# 3. ARCHIVOS DASHBOARD DATA
# ============================================================

dashboard_files <- c(
  "simulator_context.csv",
  "match_history.csv",
  "female_simulator_context.csv",
  "female_match_history.csv",
  "female_change_metrics.csv",
  "female_change_model_features.csv",
  "female_change_selection.csv"
)


# ============================================================
# 4. COPIAR DASHBOARD DATA
# ============================================================

for (file_name in dashboard_files) {

  source_file <- file.path(
    project_root,
    "dashboard",
    "data",
    file_name
  )

  destination_file <- file.path(
    dashboard_data_dir,
    file_name
  )


  if (!file.exists(source_file)) {

    stop(
      paste(
        "Falta el archivo:",
        source_file
      )
    )
  }


  success <- file.copy(
    from = source_file,
    to = destination_file,
    overwrite = TRUE
  )


  if (!success) {

    stop(
      paste(
        "No se pudo copiar:",
        file_name
      )
    )
  }


  cat(
    "Copiado:",
    file_name,
    "\n"
  )
}


# ============================================================
# 5. MODELO VARONIL
# ============================================================

male_model_source <- file.path(
  project_root,
  "outputs",
  "models",
  "blend_xgboost_model.json"
)

male_model_destination <- file.path(
  models_dir,
  "blend_xgboost_model.json"
)


if (!file.exists(male_model_source)) {

  stop(
    paste(
      "No existe el modelo varonil:",
      male_model_source
    )
  )
}


file.copy(
  male_model_source,
  male_model_destination,
  overwrite = TRUE
)


cat(
  "Copiado modelo varonil\n"
)


# ============================================================
# 6. MODELO FEMENIL
# ============================================================

female_model_source <- file.path(
  project_root,
  "outputs",
  "models",
  "female",
  "change_model",
  "female_change_model.json"
)

female_model_destination <- file.path(
  female_models_dir,
  "female_change_model.json"
)


if (!file.exists(female_model_source)) {

  stop(
    paste(
      "No existe el modelo femenil:",
      female_model_source
    )
  )
}


file.copy(
  female_model_source,
  female_model_destination,
  overwrite = TRUE
)


cat(
  "Copiado modelo femenil\n"
)


# ============================================================
# 7. COPIAR app.py
# ============================================================

app_source <- file.path(
  project_root,
  "dashboard",
  "app.py"
)

app_destination <- file.path(
  bundle_dir,
  "dashboard",
  "app.py"
)


if (!file.exists(app_source)) {

  stop(
    "No existe dashboard/app.py"
  )
}


file.copy(
  app_source,
  app_destination,
  overwrite = TRUE
)


cat(
  "Copiado app.py\n"
)


# ============================================================
# 8. COPIAR CSS
# ============================================================

styles_source <- file.path(
  project_root,
  "dashboard",
  "styles"
)

styles_destination <- file.path(
  bundle_dir,
  "dashboard",
  "styles"
)


dir.create(
  styles_destination,
  recursive = TRUE,
  showWarnings = FALSE
)


css_source <- file.path(
  styles_source,
  "america.css"
)

css_destination <- file.path(
  styles_destination,
  "america.css"
)


if (file.exists(css_source)) {

  file.copy(
    css_source,
    css_destination,
    overwrite = TRUE
  )

  cat(
    "Copiado america.css\n"
  )
}


# ============================================================
# 9. COPIAR CONFIG
# ============================================================

config_source <- file.path(
  project_root,
  "dashboard",
  "config"
)

config_destination <- file.path(
  bundle_dir,
  "dashboard",
  "config"
)


dir.create(
  config_destination,
  recursive = TRUE,
  showWarnings = FALSE
)


config_files <- list.files(
  config_source,
  full.names = TRUE
)


for (config_file in config_files) {

  file.copy(
    config_file,
    file.path(
      config_destination,
      basename(config_file)
    ),
    overwrite = TRUE
  )
}


cat(
  "Copiado config/\n"
)


# ============================================================
# 10. REQUIREMENTS
# ============================================================

requirements_source <- file.path(
  project_root,
  "dashboard",
  "requirements.txt"
)

requirements_destination <- file.path(
  bundle_dir,
  "dashboard",
  "requirements.txt"
)


if (file.exists(requirements_source)) {

  file.copy(
    requirements_source,
    requirements_destination,
    overwrite = TRUE
  )

  cat(
    "Copiado requirements.txt\n"
  )
}


# ============================================================
# 11. VALIDACIÓN FINAL
# ============================================================

cat("\n")
cat("============================================\n")
cat("STREAMLIT BUNDLE LISTO\n")
cat("============================================\n\n")


cat(
  "Carpeta creada:\n",
  bundle_dir,
  "\n\n"
)


cat(
  "Archivos incluidos:\n\n"
)


bundle_files <- list.files(
  bundle_dir,
  recursive = TRUE
)


print(
  bundle_files
)


cat("\n")
cat("TODO CORRECTO.\n")