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

# Build a workflow object given an algorithm name (from the UI) and training 
# data
gen_wflow <- function(algo_name, train_data, dep_var) {
  dep_var <- enquo(dep_var)
  
  mod_rec <- recipe(train_data) %>%
    update_role(!!dep_var, new_role = "outcome") %>%
    update_role(-all_outcomes(), new_role = "predictor") %>%
    step_nzv(all_predictors(freq_cut = )) %>%
    step_YeoJohnson(all_numeric_predictors())
  
  if (algo_name %in% "Logistic Regression") {
    mod_rec <- mod_rec %>% step_dummy(all_nominal_predictors(), one_hot = FALSE)
  } else {
    mod_rec <- mod_rec %>% step_dummy(all_nominal_predictors(), one_hot = TRUE)
  }
  
  mod_rec <- mod_rec %>%
    themis::step_smote(!!dep_var, over_ratio = tune(), skip = TRUE, seed = 732)
  
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

# Finalize the parameters for a model in a workflow object
choose_params <- function(x, df, seed) {
  set.seed = seed
  finalize(parameters(x), df)
}

# Tune a model using Bayesian optimization
tune_mod <- function(wf, algo_name, folds, params, seed) {
  # if (algo_name %in% "Logistic Regression") {
  #   return(
  #     fit_resamples(
  #       wf,
  #       resamples = folds,
  #       metrics = metric_set(roc_auc)
  #     )
  #   )
  # }
  
  tune_bayes(
    wf,
    resamples = folds,
    param_info = params,
    initial = 5,
    iter = 20,
    metrics = metric_set(roc_auc),
    control = control_bayes(no_improve = 10, seed = seed)
  )
}

# Calculate sensitivity/specificity associated with different probability
# thresholds
get_cutoff_threshold_data <- function(wflow_eval, tr, est) {
  tr <- enquo(tr)
  est <- enquo(est)
  
  preds <- wflow_eval %>% collect_predictions() %>% select(!!tr, !!est)
  
  threshold_data <- preds %>%
    probably::threshold_perf(
      !!tr,
      !!est,
      thresholds = seq(0, 1, by = 0.001)
    ) %>%
    pivot_wider(
      id_cols = (.threshold), 
      names_from = .metric, 
      values_from = .estimate
    )
  
  list(preds = preds, threshold_data = threshold_data)
}

# Get the optimal cutoff or the best cutoff corresponding with the user
# preference
get_best_cutoff <- function(threshold_dat, err_pref) {
  if ((err_pref - 3) < 0) {
    # Cutoff to reduce FPR (high sensitivity)
    threshold_dat %>% 
      arrange(desc(sens)) %>%
      slice(1:(which(.$j_index == max(j_index))[1] - 1)) %>%
      mutate(rank_sens = ntile(sens, 2)) %>%
      filter(rank_sens == abs(err_pref - 3)) %>%
      slice_max(j_index, with_ties = FALSE) %>%
      pull(.threshold)
  } else if ((err_pref - 3) > 0) {
    # Cutoff to reduce FNR (high specificity)
    threshold_dat %>% 
      arrange(desc(spec)) %>%
      slice(1:(which(.$j_index == max(j_index))[1] - 1)) %>%
      mutate(rank_spec = ntile(spec, 2)) %>%
      filter(rank_spec == abs(err_pref - 3)) %>%
      slice_max(j_index, with_ties = FALSE) %>%
      pull(.threshold)
  } else {
    # optimal cutoff
    threshold_dat %>%
      arrange(desc(.threshold)) %>%
      slice_max(j_index, with_ties = FALSE) %>%
      pull(.threshold)
  }
}


# Model evaluation functions ===================================================

# Store a 'metrics' object with the metrics to calculate
ys_metrics <- metric_set(mcc, bal_accuracy, sensitivity, specificity)

# Get metrics using custom cutoff
get_ys_metrics <- function(pred_data, cutoff, tr, est, outcome_1, outcome_0) {
  tr <- enquo(tr)
  est <- enquo(est)
  
  preds <- pred_data %>%
    mutate(
      pred_class = if_else(!!est > cutoff, outcome_1, outcome_0) %>% 
        factor(levels = c(outcome_1, outcome_0))
    )
  
  rocauc_metrics <- roc_auc(preds, !!tr, !!est)
  
  roc_curve_dat <- roc_curve(preds, !!tr, !!est)
  
  add_metrics <- ys_metrics(
    preds,
    truth = !!tr,
    estimate = pred_class
  ) %>%
    bind_rows(rocauc_metrics) %>%
    select(-.estimator) %>%
    pivot_wider(names_from = .metric, values_from = .estimate) %>%
    mutate(cutoff = cutoff)
  
  list(cutoff = cutoff, ys_metrics = add_metrics, roc_curve_dat = roc_curve_dat)
}

# Wrapper for a predicting on Tidy objects
tidy_pred <- function(object, newdata) {
  predict.model_fit(object, new_data = newdata, type = "prob") %>%
    pull(.pred_good)
}


# Get permutative variable importance given a finalized model workflow
get_var_imp <- function(wflow, ref_class, pred_fun) {
  mod <- extract_fit_parsnip(wflow)
  train_dat <- extract_mold(wflow) %>% pluck("predictors")
  target_dat <- extract_mold(wflow) %>% pluck("outcomes")
  
  vi_df <- vip::vi_permute(
    mod, 
    train = train_dat, 
    target = target_dat, 
    metric = "auc",
    pred_wrapper = pred_fun,
    reference_class = ref_class,
    nsim = 10,
    paralell = TRUE
  )
}

# Get a dataframe of SHAP values for each combination of feature and observation
get_shap <- function(wflow, pred_fun) {
  fastshap::explain(
    extract_fit_parsnip(wflow),
    X = extract_mold(wflow) %>% pluck("predictors") %>% as.matrix(),
    feature_names = extract_mold(wflow) %>% 
      pluck("predictors") %>% 
      colnames(),
    pred_wrapper = pred_fun,
    nsim = 10
  )
}

# Get a SHAP summary dataframe that contains feature observations, predictions,
# and permutative importance
get_shap_summary <- function(vi, shap_df, feat_df, max_features = 10) {
  vi <- vi %>%
    set_names(colnames(.) %>% str_to_lower()) %>%
    slice_max(importance, n = max_features, with_ties = FALSE)
  
  shap_df <- shap_df %>%
    as_tibble() %>%
    mutate(id = row_number()) %>%
    pivot_longer(cols = -id, names_to = "variable", values_to = "shap_value")
  
  feat_df <- feat_df %>% 
    mutate(id = row_number()) %>%
    pivot_longer(cols = -id, names_to = "variable", values_to = "feature_value")
  
  left_join(shap_df, feat_df, by = c("variable", "id")) %>%
    right_join(vi, by = "variable") %>%
    mutate(
      variable = str_c(
        variable, 
        " (", format(round(importance, 3), nsmall = 3), ")"
      )
    ) %>%
    arrange(desc(importance))
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

# Generate a plot of chi-squared significance values
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

# Generate a correlation plot
plot_corr <- function(df) {
  df %>%
    select_if(is.numeric) %>%
    corrr::correlate(method = "spearman", use = "pairwise.complete.obs") %>%
    corrr::rearrange(absolute = FALSE) %>%
    corrr::shave() %>%
    corrr::rplot(colors = c("red", "lightgray", "blue")) +
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

# Generate a barplot showing class group sizes
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

# Create a plot of the ROC Curve
plot_roc <- function(roc_df, auc) {
  roc_df %>%
    mutate(fpr = 1 - specificity) %>%
    ggplot(aes(x = fpr, y = sensitivity)) +
    geom_path(color = "black") +
    geom_abline(slope = 1, intercept = 0, lty = 2, color = "red") +
    ml_eval_theme() +
    labs(
      title = "ROC Curve",
      subtitle = str_c("AUC: ", scales::percent(auc, accuracy = .1)),
      y = "True Positive Rate (Sensitivity)",
      x = "False Positive Rate (1 - Specificity)"
    )
}

# Create a variable importance plot
plot_var_imp <- function(vi_df, algo_name, region_name, max_features = 10) {
  vi_plot_df <- vi_df %>% 
    slice_max(Importance, n = max_features) %>%
    arrange(desc(Importance))
  
  vi_plot_df %>% 
    vip::vip() +
    ml_eval_theme() +
    labs(
      title = paste("Variable Importance:", algo_name),
      subtitle = str_c("(", region_name, ")")
    )
}

# Create a SHAP summary plot
plot_shap_summary <- function(shap_summary_df, algo_name, region_name) {
  region_string <- paste0("(", region_name, ")")
  
  shap_summary_df %>%
    ggplot(
      aes(shap_value, fct_rev(fct_inorder(variable)), color = feature_value)
    ) +
    geom_jitter(height = 0.2, alpha = 0.8) +
    geom_vline(xintercept = 0, size = 1) +
    scale_color_gradient(low = "blue", high = "red") +
    labs(
      title = paste("SHAP Summary Plot:", algo_name),
      subtitle = region_string,
      x = "SHAP Value",
      y = NULL,
      color = "Feature Value"
    ) + 
    ml_eval_theme()
}
