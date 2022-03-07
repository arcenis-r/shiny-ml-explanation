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



# TODO Items ===================================================================
# TODO: Write code for finalizing the model 
# TODO: Generate model metrics
# TODO: Implement cut-off selection based on user preference
# TODO: Learn to create LIME summary plot (using SHAP variable importance)


# Set up the environment =======================================================
# Clear the workspace
rm(list = ls())

# Load required packages
library(tidyverse)
library(tidymodels)
library(lime)
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
  tags$head(tags$style(HTML("hr {border-top: 1px solid #000000;}"))),
  titlePanel(
    h1("Explaining ML - SHAP vs LIME", align = "center"),
    windowTitle = "Explaining ML - SHAP vs LIME"
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
            "Reduce FPR - Strong",
            "Reduce FPR - Moderate",
            "Neutral",
            "Reduce FNR - Moderate",
            "Reduce FNR - Strong"
          ),
          choiceValues = c(1:5),
          selected = "3"
        )  # end Misclassification Preference radio buttons
      ),  # end Misclassification Preference column
      
      # Create buttons for different actions
      tags$div(
        class = "col-xs-3",
        br(),
        actionButton("train_mod", "Train Model", width = "75%"),
        br(), br(),
        actionButton("dl_pdf", "Download PDF", width = "75%"),
        br(), br(),
        actionButton("dl_mod_obj", "Download Model Objects", width = "75%")
      )
    )  # end wellPanel row formatting
  ),  # end wellPanel (inputs)
  
  tabsetPanel(
    tabPanel(
      "Exploratory Data Analysis",
      
      # Put 3 EDA plots into one row
      fluidRow(
        column(4, plotOutput("class_bal_plot")),
        column(4, plotOutput("chi_sq_plot")),
        column(4, plotOutput("corr_plot"))
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
      tags$div(align = "center", tableOutput("mod_metrics_table")),
      plotOutput("roc_plot")
    ),  # end Model Evaluation panel def
    
    tabPanel(
      "SHAP",
      # tableOutput("shap_table")
      # verbatimTextOutput("shap_class")
      plotOutput("shap_summary_plot")
    ),  # end SHAP panel def
    
    tabPanel(
      "LIME"
    ),  # end LIM panel def
    
    tabPanel(
      "Partial Dependence Plot & Variable Importance",
      plotOutput("var_imp_plot")
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
  mod_wflow <- reactive({gen_wflow(input$algo, training(tt_split()))})
  
  # Set the hyperparameters for the model
  hp_set <- reactive({
    choose_params(mod_wflow(), training(tt_split()) %>% select(-Class), 500)
  })
  
  # Tune the model and store the results
  tune_results <- reactive({
    withProgress(
      message = "Tuning model hyperparameters",
      tune_mod(mod_wflow(), mod_inputs()$algo, cv_folds(), hp_set(), 384)
    )
  })
  
  
  # Train a final model and generate evaluation objects ------------------------
  
  # Select the best model
  best_mod <- reactive({select_best(tune_results(), metric = "roc_auc")})
  
  # Fit the model with the best parameters to the entire training dataset
  final_wflow <- reactive({finalize_workflow(mod_wflow(), best_mod())})
  
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
    get_ys_metrics(
      cutoff_thresh()$preds, 
      opt_cutoff(),
      Class,
      .pred_good,
      "good",
      "bad"
    )
  })
  
  
  # Generate Variable Importance, SHAP, and LIME objects -----------------------
  
  # Get a matrix of training features for the model
  training_features <- reactive({
    set.seed(8405)
    extract_mold(final_wflow_eval()) %>% pluck("predictors")
  })
  
  # Generate variable importance values
  set.seed(8405)
  var_imp <- reactive({
    withProgress(
      message = "Calculating permutative variable importance",
      get_var_imp(final_wflow_eval(), "good", tidy_pred)
    )
  })
  
  # Get SHAP values
  shap <- reactive({
    set.seed(44)
    withProgress(
      message = "Calculating SHAP values",
      get_shap(final_wflow_eval(), tidy_pred)
    )
  })
  
  
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
    plot_var_imp(var_imp(), mod_inputs()$algo, mod_inputs()$region, 10)
  })
  
  # Store SHAP plot
  shap_summary_plot <- reactive({
    get_shap_summary(var_imp(), shap(), training_features(), 10) %>%
      plot_shap_summary(mod_inputs()$algo, mod_inputs()$region)
  })
  
  
  # Create output objects for display ------------------------------------------
  
  # EDA tab
  output$chi_sq_plot <- renderPlot(chi_sq_plot())
  output$corr_plot <- renderPlot(corr_plot())
  output$class_bal_plot <- renderPlot(class_bal_plot())
  output$skim_fct <- renderTable(skim_fct())
  output$skim_num <- renderTable(skim_num())
  
  # Model Tuning tab
  output$lr_tuning_text <- renderText({
    if (mod_inputs()$algo %in% "Logistic Regression") {
      "Logistic Regression does not get tuned."
    }
  })
  
  output$best_mod <- renderTable({
    if (!mod_inputs()$algo %in% "Logistic Regression") {
      show_best(tune_results())
    }
  })
  
  # Model Evaluation tab
  output$mod_metrics_table <- renderTable(model_metrics_table())
  
  output$roc_plot <- renderPlot(roc_plot())
  
  # SHAP tab
  # output$shap_class <- renderPrint(str(shap_summary_plot()))
  # output$shap_table <- renderPlot({as.data.frame(shap())})
  output$shap_summary_plot <- renderPlot(shap_summary_plot())
  
  # Partial Dependence Plot & Variable Importance tab
  output$var_imp_plot <- renderPlot(var_imp_plot())
} # end 'server' function


# Run the application ==========================================================
shinyApp(ui = ui, server = server)