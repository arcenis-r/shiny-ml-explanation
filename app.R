################################################################################
# Script name: app.R
# Author: Arcenis Rojas
# E-mail: arcenis.rojas@tutanota.com
# Date created: 3/1/2022
#
# Script description: Creates a Shiny application that allows the user to train
#   a model using a user-selected algorithm along with other inputs then
#   displays data about 
#
################################################################################

# Notes ========================================================================
# Consider adding SMOTE to the pre-processing


# TODO Items ===================================================================
# TODO: Generate LIME values
# TODO: Calculate importance scores for both SHAP and LIME
# TODO: Generate plots for SHAP vs LIME (summary plots and max "good" and max
#   "bad" plots... 3 plots for each method)
# TODO: Create bar plot containing 3 types of variable importance measures
# TODO: Create partial dependence plots for the top 4 numeric variables in terms
#   of permutative variable importance


# Set up the environment =======================================================
# Clear the workspace
rm(list = ls())

# Load required packages
library(tidyverse)
library(tidymodels)
library(shiny)

# Load / declare helper functions
source("./scripts/shiny-ml-explanation-funs.R")


################################ Program start #################################

# Load Lending Club data from 'modeldata'
data(lending_club)

# Merge the state-region crosswalk data to allow users to filter by region
lc_data <- lending_club %>% 
  left_join(
    readr::read_rds("./data/state-region-crosswalk.Rds"),
    by = c("addr_state" = "state_abb")
  )

# Define UI for application ====================================================

ui <- fluidPage(
  tags$head(
    tags$style(
      HTML("hr {border-top: 1px solid #000000;}"),
      ".expSummRow{height:500px;}"
    )
  ),
  titlePanel(
    h1("Opening The 'Black Box'", align = "center"),
    windowTitle = "Opening The 'Black Box'"
  ),
  wellPanel(
    style = "background: lightblue",
    tags$div(
      class = "row",
      
      # Allow the user to filter data by Region
      tags$div(
        class = "col-xs-3",
        selectInput(
          "region_filter",
          "Filter by Region",
          c("All", lc_data %>% distinct(region_name) %>% pull(region_name)),
          selected = "All"
        )  # end Regioin input
      ),  # end Region input column
      
      # Allow the user to select an algorithm for the model
      tags$div(
        class = "col-xs-3",
        selectInput(
          "algo",
          "Select a Model Algorithm",
          c(
            "Decision Tree",
            "Random Forest",
            "Boosted Tree",
            "Multi-Layer Perceptron (NN)",
            "Logistic Regression"
          ),
          selected = "Decision Tree"
        )  # end algo input
      ),  # end algo input column
      
      # Allow the user to express a misclassification preference
      tags$div(
        class = "col-xs-3",
        radioButtons(
          "misclass_pref",
          "Misclassification Preference",
          choiceNames = c(
            "Avoid False Positives - Strong",
            "Avoid False Positives - Moderate",
            "Neutral (Optimal Cutoff)",
            "Avoid False Negatives - Moderate",
            "Avoid False Negatives - Strong"
          ),
          choiceValues = c(1:5),
          selected = "3"
        )  # end Misclassification Preference radio buttons
      ),  # end Misclassification Preference column
      
      # Create buttons for different actions
      tags$div(
        class = "col-xs-3",
        br(),
        actionButton("train_mod", "Train Model", width = "75%")
        # br(), br(),
        # actionButton("dl_pdf", "Download PDF", width = "75%"),
        # br(), br(),
        # actionButton("dl_mod_obj", "Download Model Objects", width = "75%")
      )
    )  # end wellPanel row formatting
  ),  # end wellPanel (inputs)
  
  tabsetPanel(
    id = "tabs",
    tabPanel(
      "Exploratory Data Analysis",
      
      # Put 3 EDA plots into one row
      fluidRow(
        column(4, plotOutput("chi_sq_plot")),
        column(3, plotOutput("class_bal_plot")),
        column(5, plotOutput("corr_plot"))
      ), # end row of plots
      
      tags$hr(),
      
      # Put {skimr} tables for factors and numerics each in its own row
      fluidRow(tableOutput("skim_fct")),
      tags$hr(),
      fluidRow(tableOutput("skim_num"))
    ),  # end EDA panel def
    
    tabPanel(
      "Model Tuning",
      # Create the UI for Logistic Regression
      conditionalPanel(
        condition = "input.algo %in% 'Logistic Regression'",
        textOutput("lr_tuning_text")
      ),
      conditionalPanel(
        condition = "!input.algo %in% 'Logistic Regression'",
        tableOutput("best_mod")
      )
    ),  # end Tuning panel def
    
    tabPanel(
      "Model Evaluation",
      column(
        width = 3,
        tags$div(align = "center", tableOutput("mod_metrics_table"))
      ),
      # tags$div(align = "center", tableOutput("mod_metrics_table")),
      column(width = 6, plotOutput("roc_plot"))
    ),  # end Model Evaluation panel def
    
    tabPanel(
      "SHAP",
      plotOutput("shap_summary_plot"),
      fluidRow(
        column(width = 6, plotOutput("shap_contrib_good")),
        column(width = 6, plotOutput("shap_contrib_bad"))
      )
    ),  # end SHAP panel def
    
    # tabPanel(
    #   "LIME",
    #   verbatimTextOutput("lime_str")
    #   plotOutput("lime_summary_plot"),
    #   plotOutput("lime_contrib_good"),
    #   plotOutput("lime_contrib_bad")
    # ),  # end LIME panel def
    
    tabPanel(
      "Variable Importance",
      plotOutput("var_imp_plot")
    ),  # end Variable Importance panel def
    
    tabPanel(
      "Partial Dependence",
      plotOutput("pdp_plot")
    )  # end PD and Var Imp panel def
  )  # end 'tabsetPanel'
)  # end UI definition


# Define server logic ==========================================================

server <- function(input, output) {
  # Wrangle the data -----------------------------------------------------------
  
  # Filter the data if a region is chosen
  filter_data <- reactive({
    if (!input$region_filter %in% "All") {
      return(filter(lc_data, region_name %in% input$region_filter))
    }
    
    lc_data
  })
  
  # Final data wrangling
  mod_data <- reactive({
    filter_data() %>%
      select(-region_name) %>%
      mutate(
        across(where(is.factor), fct_drop),
        Class = fct_relevel(Class, c("good", "bad"))
      )
  })
  
  
  # Perform pre-processing and model tuning ------------------------------------
  
  # Capture the user inputs when the "Train Model" button is clicked
  mod_inputs <- eventReactive(
    input$train_mod,
    {
      if (input$algo %in% "Logistic_Regression") {
        hideTab("tabs", "Model Tuning")
      } else {
        showTab("tabs", "Model Tuning")
      }
      
      list(
        df = mod_data(),
        region = input$region_filter,
        algo = input$algo, 
        err_pref = input$misclass_pref
      )
    }
  )
  
  # Split the data into training and testing sets
  tt_split <- reactive({
    set.seed(200)
    initial_split(mod_inputs()$df, 3/4, strata = Class)
  })
  
  # Split training data into cross-validation folds
  cv_folds <- reactive({
    set.seed(479)
    vfold_cv(training(tt_split()), v = 5, strata = Class)
  })
  
  # Build a model workflow
  mod_wflow <- reactive({
    gen_wflow(mod_inputs()$algo, training(tt_split()), Class)
  })
  
  # Set the hyperparameters for the model
  hp_set <- reactive({
    choose_params(mod_wflow(), training(tt_split()) %>% select(-Class), 500)
  })
  
  # Tune the model and store the results
  tune_results <- reactive({
    withProgress(
      message = "Resampling and tuning model hyperparameters",
      tune_mod(mod_wflow(), mod_inputs()$algo, cv_folds(), hp_set(), 384)
    )
  })
  
  
  # Train a final model and generate evaluation objects ------------------------
  
  # Select the best model
  best_mod <- reactive({select_best(tune_results(), metric = "roc_auc")})
  
  # Fit the model with the best parameters to the entire training dataset
  final_wflow <- reactive({
    set.seed(9474)
    finalize_workflow(mod_wflow(), best_mod()) %>%
      fit(data = training(tt_split()))
  })
  
  # Evaluate the model with the test data and store a 'last_fit' object
  final_wflow_eval <- reactive({
    set.seed(845)
    last_fit(
      final_wflow(),
      split = tt_split(),
      metrics = metric_set(roc_auc)
    )
  })
  
  # Get table of probability cutoff thresholds
  cutoff_thresh <- reactive({
    get_cutoff_threshold_data(final_wflow_eval(), Class, .pred_good)
  })
  
  # Calculate the best cutoff given the user's preference
  opt_cutoff <- reactive({
    pluck(cutoff_thresh(), "threshold_data") %>% 
      get_best_cutoff(as.numeric(mod_inputs()$err_pref))
  })
  
  # Generate a dataframe of model metrics
  model_metrics <- reactive({
    withProgress(
      message = "Calculating model metrics",
      get_ys_metrics(
        cutoff_thresh()$preds, 
        opt_cutoff(),
        Class,
        .pred_good,
        "good",
        "bad"
      )
    )
  })
  
  # Get the row numbers of the observations that have the highest probabilities
  # of being predicted as "good" and "bad" respectively
  max_pred_good_row <- reactive({
    collect_predictions(final_wflow_eval()) %>% 
      slice_max(.pred_good) %>%
      slice(1) %>% 
      pull(.row)
  })
  
  max_pred_bad_row <- reactive({
    collect_predictions(final_wflow_eval()) %>% 
      slice_max(.pred_bad) %>%
      slice(1) %>% 
      pull(.row)
  })
  
  
  # Generate Variable Importance, SHAP, and LIME objects -----------------------
  
  # Get a matrix of training features for the model
  training_features <- reactive({
    extract_mold(final_wflow_eval()) %>% pluck("predictors")
  })
  
  # Generate Permutative variable importance values
  var_imp <- reactive({
    set.seed(8405)
    withProgress(
      message = "Calculating permutative variable importance",
      get_var_imp(final_wflow_eval(), "good", tidy_pred)
    )
  })
  
  # Calculate SHAP values
  shap <- reactive({
    set.seed(44)
    withProgress(
      message = "Calculating SHAP values",
      get_shap(final_wflow_eval(), tidy_pred)
    )
  })
  
  # Generate SHAP variable importance
  shap_var_imp <- reactive({
    withProgress(
      message = "Calculating SHAP variable importance",
      get_shap_imp(shap())
    )
  })
  
  # Calculate LIME values
  # lime_expl <- reactive({
  #   set.seed(27)
  #   withProgress(
  #     message = "Calculating LIME values",
  #     get_lime(
  #       training(tt_split()), 
  #       testing(tt_split()), 
  #       extract_fit_parsnip(final_wflow_eval()), 
  #       nfeatures = 10
  #     )
  #   )
  # })
  
  
  # Store additional plots and tables for output -------------------------------
  
  # Store the EDA plots and tables as reactive objects to use for display and
  # inclusion in the PDF
  chi_sq_plot <- reactive({plot_chi_sq(mod_data() %>% select(-Class))})
  corr_plot <- reactive({plot_corr(mod_data() %>% select(-Class))})
  class_bal_plot <- reactive({plot_class_bal(mod_data(), Class)})
  violin_plot <- reactive({plot_violins(mod_data())})
  skim_fct <- reactive({
    mod_data() %>% select(where(is.factor)) %>% skimr::skim()
  })
  skim_num <- reactive({
    mod_data() %>% select(where(is.numeric)) %>% skimr::skim()
  })
  
  # Store ROC plot
  roc_plot <- reactive({
    plot_roc(model_metrics()$roc_curve_dat, model_metrics()$ys_metrics$roc_auc)
  })
  
  # Model metrics table
  model_metrics_table <- reactive({model_metrics()$ys_metrics})
  
  # Variable Importance Plot
  var_imp_plot <- reactive({
    plot_var_imp(
      var_imp(), 
      shap_var_imp(), 
      mod_inputs()$algo, 
      mod_inputs()$region, 
      20
    )
  })
  
  # SHAP summary plot
  shap_summary_plot <- reactive({
    get_shap_summary(var_imp(), shap(), training_features(), 20) %>%
      plot_shap_summary(mod_inputs()$algo, mod_inputs()$region)
  })
  
  # SHAP contribution plots
  shap_contrib_good <- reactive({
    plot_shap_contributions(shap(), mod_inputs()$algo, max_pred_good_row(), 20)
  })
  
  shap_contrib_bad <- reactive({
    plot_shap_contributions(shap(), mod_inputs()$algo, max_pred_bad_row(), 20)
  })
  
  # LIME summary plot
  # lime_summary_plot <- reactive({lime::plot_explanations(lime_expl())})
  
  # LIME contribution plots
  # lime_contrib_good <- reactive({
  #   lime::plot_features(lime_expl(), cases = max_pred_good_row())
  # })
  
  # lime_contrib_bad <- reactive({
  #   lime::plot_features(lime_expl(), cases = max_pred_bad_row())
  # })
  
  # Partial dependence plot
  pdp_plots <- reactive({
    # First, get the numeric variable with the highest permutative variable
    # importance
    pdp_var <- mod_inputs()$df %>%
      select(where(is.numeric)) %>%
      colnames() %>%
      enframe("var_num", "Variable") %>%
      left_join(var_imp(), by = "Variable") %>%
      slice_max(abs(Importance), n = 1, with_ties = FALSE) %>%
      pull(Variable)
    
    print(pdp_var)
    
    plot_pdp(final_wflow(), !! rlang::sym(pdp_var), mod_inputs()$algo)
  })
  
  
  # Create output objects for display ------------------------------------------
  
  # EDA tab
  output$chi_sq_plot <- renderPlot(chi_sq_plot())
  output$corr_plot <- renderPlot(corr_plot())
  output$class_bal_plot <- renderPlot(class_bal_plot())
  output$skim_fct <- renderTable(skim_fct())
  output$skim_num <- renderTable(skim_num())
  
  # Model Tuning tab
  output$tuning_notes <- renderPrint(tune_notes())
  
  output$lr_tuning_text <- renderText({
    if (mod_inputs()$algo %in% "Logistic Regression") {
      "Logistic Regression does not get tuned, but resampling is still done."
    }
  })
  
  output$best_mod <- renderTable({
    if (!mod_inputs()$algo %in% "Logistic Regression") {
      show_best(tune_results())
    }
  })
  
  # Model Evaluation tab
  output$mod_metrics_table <- renderTable(
    model_metrics_table() %>%
      pivot_longer(everything(), names_to = "Metric", values_to = "Value")
  )
  output$roc_plot <- renderPlot(roc_plot())
  
  # SHAP tab
  output$shap_summary_plot <- renderPlot(shap_summary_plot())
  output$shap_contrib_good <- renderPlot(shap_contrib_good())
  output$shap_contrib_bad <- renderPlot(shap_contrib_bad())
  
  # LIME tab
  # output$lime_str <- renderPrint(str(lime_expl()))
  # output$lime_summary_plot <- renderPlot(lime_summary_plot())
  # output$lime_contrib_good <- renderPlot(lime_contrib_good())
  # output$lime_contrib_bad <- renderPlot(lime_contrib_bad())
  
  # Variable Importance tab
  output$var_imp_plot <- renderPlot(var_imp_plot())
  
  # Partial Dependence tab
  output$pdp_plot <- renderPlot(pdp_plots())
} # end 'server' function


# Run the application ==========================================================
shinyApp(ui = ui, server = server)