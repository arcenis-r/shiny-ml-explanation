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
            "Support Vector Machine",
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
            "Avoid FN - Strong",
            "Avoid FN - Moderate",
            "Neutral",
            "Avoid FP - Moderate",
            "Avoid FP - Strong"
          ),
          choiceValues = c(
            "avoid_fn_strong",
            "avoid_fn_moderate",
            "neutral",
            "avoid_fp_moderate",
            "avoid_fp_strong"
          ),
          selected = "neutral"
        )  # end Misclassification Preference radio buttons
      ),  # end Misclassification Preference column
      
      # Create buttons for different actions
      tags$div(
        class = "col-xs-3",
        actionButton("train_mod", "Train Model", width = "100%"),
        actionButton("dl_pdf", "Download PDF", width = "100%"),
        actionButton("dl_mod_obj", "Download Model Objects", width = "100%")
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
      "SHAP vs LIME"
    ),  # end Explanation panel def
    
    tabPanel(
      "Model Evaluation"
    ),  # end Evaluation panel def
    
    tabPanel(
      "Model Tuning"
    )  # end Tuning panel def
  )  # end 'tabsetPanel'
)  # end UI definition


# Define server logic ==========================================================

server <- function(input, output) {
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
      mutate(across(where(is.factor), fct_drop))
  })
  
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
  
  # Create the output for the EDA tab
  output$chi_sq_plot <- renderPlot(chi_sq_plot())
  output$corr_plot <- renderPlot(corr_plot())
  output$class_bal_plot <- renderPlot(class_bal_plot())
  output$skim_fct <- renderTable(skim_fct())
  output$skim_num <- renderTable(skim_num())
} # end 'server' function


# Run the application ==========================================================
shinyApp(ui = ui, server = server)