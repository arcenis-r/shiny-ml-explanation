################################################################################
# Script name: shiny-ml-explanation-funs.R
# Author: Arcenis Rojas
# E-mail: arcenis.rojas@tutanota.com
# Date created: 3/1/2022
#
# Script description: Contains helper functions for the ML Explanation - SHAP
#   vs LIME shiny app
#
################################################################################

# Notes ========================================================================



# TODO Items ===================================================================



# Modeling functions ===========================================================
gen_wflow <- function(algo_name, train_data) {
  mod_rec <- recipe(Class ~ ., train_data) %>%
    step_nzv(all_predictors()) %>%
    step_YeoJohnson(all_numeric_predictors())
  
  if (algo_name %in% "Logistic Regression") {
    mod_rec <- mod_rec %>% step_dummy(all_nominal_predictors(), one_hot = FALSE)
  } else {
    mod_rec <- mod_rec %>% step_dummy(all_nominal_predictors(), one_hot = TRUE)
  }
  
  mod_def <- switch(
    algo_name,
    "Decision Tree" = decision_tree(cost_complexity = tune()) %>%
      set_engine("rpart"),
    "Random Forest" = rand_forest(trees = tune(), mtry = tune()) %>%
      set_engine("ranger"),
    "Boosted Tree" = boost_tree(
      trees = tune(), 
      mtry = tune(),
      learn_rate = tune()
    ) %>%
      set_engine("xgboost"),
    "Multi-Layer Perceptron (NN)" = mlp(
      hidden_units = tune(),
      penalty = tune(),
      dropout = tune(),
      epochs = tune()
    )%>%
      set_engine("nnet"),
    "Logistic Regression" = logistic_reg() %>% set_engine("glm")
  )
  
  mod_def <- mod_def %>% set_mode("classification")
  
  workflow() %>% add_recipe(mod_rec) %>% add_model(mod_def)
}

choose_params <- function(wf, df, seed) {
  set.seed = seed
  finalize(parameters(wf), df)
}

tune_mod <- function(wf, folds, params, seed) {
  tune_bayes(
    wf,
    resamples = folds,
    param_info = params,
    initial = 5,
    iter = 20,
    metrics = metric_set(roc_auc),
    control = control_bayes(no_improve = 5, seed = seed)
  )
}


# Extracting objects ===========================================================

# Get the index number of a recipe step 
get_rec_step_num <- function(rec, step_name_str) {
  map_lgl(
    rec$steps, 
    ~ .x %>% 
      attributes() %>% 
      pluck("class") %>% 
      map_lgl(~ str_detect(.x, step_name_str)) %>% 
      any()
  ) %>% 
    which()
}


# Plotting functions ===========================================================

# Create a ggplot2 theme to use throughout the project
ml_eval_theme <- function() {
  theme_bw() + 
    theme(
      plot.title = element_text(hjust = 0.5, size = 18),
      plot.subtitle = element_text(hjust = 0.5, size = 14),
      strip.text = element_text(size = 12, color = "white"),
      strip.background = element_rect(fill = "#17468F"),
      axis.text = element_text(size = 10)
    )
}

plot_chi_sq <- function(df) {
  factor_names <- df %>% select_if(is.factor) %>% names()
  
  chi_sq_dat <- crossing(var1 = factor_names, var2 = factor_names) %>%
    mutate(
      chi_sq_results = map2(
        var1,
        var2,
        ~ select(df, any_of(c(.x, .y))) %>%
          table() %>%
          chisq.test() %>%
          broom::tidy()
      )
    ) %>%
    unnest(chi_sq_results) %>%
    select(var1, var2, p.value) %>%
    pivot_wider(names_from = var2, values_from = p.value) %>%
    column_to_rownames("var1")
  
  chi_sq_dat[!upper.tri(chi_sq_dat)] <- NA

  chi_sq_dat %>%
    rownames_to_column("var1") %>%
    pivot_longer(-var1, names_to = "var2", values_to = "p.value") %>%
    drop_na(p.value) %>%
    ggplot(aes(fct_rev(var2), var1, color = p.value)) +
    geom_point(size = 3) +
    # scale_color_viridis_c(direction = -1) +
    scale_color_gradient(low = "red", high = "gray") +
    labs(
      title = "Chi-square Plot of Categorical Variables",
      color = "P-value"
    ) +
    ml_eval_theme() +
    theme(
      axis.title = element_blank(),
      axis.text.x = element_text(angle = 45, hjust = 1),
      panel.border = element_blank(),
      axis.line = element_line()
    )
}

plot_corr <- function(df) {
  df %>%
    select_if(is.numeric) %>%
    corrr::correlate(method = "spearman", use = "pairwise.complete.obs") %>%
    corrr::rearrange(absolute = FALSE) %>%
    corrr::shave() %>%
    corrr::rplot(colors = c("red", "white", "blue")) +
    labs(
      title = "Correlation Plot of Numeric Variables",
      color = "Correlation"
    ) +
    ml_eval_theme() +
    theme(
      axis.text.x = element_text(angle = 45, hjust = 1),
      panel.border = element_blank(),
      axis.line = element_line()
    )
}

plot_class_bal <- function(df, class_col) {
  class_col <- enquo(class_col)
  
  total_n <- df %>% drop_na(!!class_col) %>% nrow()
  
  df %>%
    select(!!class_col) %>%
    ggplot(aes(!!class_col, group = !!class_col)) +
    geom_bar(fill = "blue") +
    ml_eval_theme() +
    labs(
      title = "Group Counts",
      subtitle = str_c("Total obs: ", total_n),
      x = "",
      y = ""
    )
}
