################################################################################
# Script name: lime-explainer.R
# Author: Arcenis Roajs
# E-mail: arcenis.rojas@tutanota.com
# Date created: 2/27/2022
#
# Script description: Learn to build a LIME explainer using Tidy workflow
#
################################################################################

# Notes ========================================================================



# TODO Items ===================================================================



# Set up the environment =======================================================
# Clear the workspace
rm(list = ls())

# Load required packages
library(tidyverse)
library(tidymodels)
library(lime)

# Load / declare helper functions


################################ Program start #################################

# Load the lending_club data
data(lending_club)


# Split lending_club into train/test ===========================================

set.seed(555)
lc_split <- initial_split(lending_club, strata = "Class")
lc_train <- training(lc_split)
lc_test <- testing(lc_split)


# Create model workflow ========================================================

# Write a recipe that normalizes all numeric explanatory variables
lc_rec <- recipe(lc_train, Class ~ .) %>%
  step_nzv(all_predictors()) %>%
  step_normalize(all_numeric_predictors())

# Create model object
lc_mod <- rand_forest(mtry = 5, trees = 500) %>%
  set_engine("ranger") %>%
  set_mode("classification")

# Create the workflow
lc_wflow <- workflow() %>% add_recipe(lc_rec) %>% add_model(lc_mod)


# Fit the model and get predictions ============================================

set.seed(384)
lc_fit <- lc_wflow %>% fit(lc_train)
lc_fit %>% predict(lc_test %>% slice(1:5))

lc_lime <- lime(lc_train, extract_fit_parsnip(lc_fit))

# Register and set up the parallel backend
cl <- parallel::makePSOCKcluster(parallel::detectCores(logical = FALSE) - 1)
doParallel::registerDoParallel(cl)
parallel::clusterEvalQ(cl, set.seed(228))

lc_explanation <- explain(
  lc_test, 
  lc_lime, 
  n_labels = 1, 
  n_features = 10,
  n_permutations = 500
)

# Close the connection to the cluster
parallel::stopCluster(cl)

plot_explanations(lc_explanation)
plot_features(lc_explanation, cases = 1:2)
