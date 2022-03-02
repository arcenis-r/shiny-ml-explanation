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

data(lending_club)

lc_data <- lending_club %>% 
  left_join(
    readr::read_rds("./data/state-region-crosswalk.Rds"),
    by = c("addr_state" = "state_abb")
  )

# Define UI for application ====================================================

ui <- fluidPage(
  tags$head(tags$style(HTML("hr {border-top: 1px solid #000000;}"))),
  titlePanel("ML Explanation - LIME vs SHAP"),
  fluidRow(
    column(
      4,
      
      # Allow the user to filter the data by Region
      selectInput(
        "region_filter",
        "Filter by Region",
        c("All", lc_data %>% distinct(region_name) %>% pull(region_name))
      )
    )  # end Region input column
  ),  # end row of inputs
  tabsetPanel(
    tabPanel(
      "Exploratory Data Analysis",
      column(4, plotOutput("class_bal_plot")),
      column(4, plotOutput("chi_sq_plot")),
      column(4, plotOutput("corr_plot")),
      tags$hr(),
      column(12, tableOutput("lc_skim_fct")),
      tags$hr(),
      column(12, tableOutput("lc_skim_num"))
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
  
  filter_data <- reactive({
    if (!input$region_filter %in% "All") {
      return(filter(lc_data, region_name %in% input$region_filter))
    }
    
    lc_data
  })
  
  mod_data <- reactive({
    filter_data() %>%
      select(-region_name) %>%
      mutate(across(where(is.factor), fct_drop))
  })
  
  output$chi_sq_plot <- renderPlot(plot_chi_sq(mod_data() %>% select(-Class)))
  
  output$corr_plot <- renderPlot(plot_corr(mod_data() %>% select(-Class)))
  
  output$class_bal_plot <- renderPlot(plot_class_bal(mod_data(), Class))
  
  output$lc_skim_fct <- renderTable(
    skimr::skim(mod_data() %>% select(where(is.factor)))
  )
  
  output$lc_skim_num <- renderTable(
    skimr::skim(mod_data() %>% select(where(is.numeric)))
  )
} # end 'server' function

# Run the application ==========================================================
shinyApp(ui = ui, server = server)