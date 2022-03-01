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
library(shiny)
library(tidyverse)
library(tidymodels)
library(lime)

# Load / declare helper functions


################################ Program start #################################

# Define UI for application ====================================================

ui <- fluidPage(
  titlePanel("ML Explanation - LIME vs SHAP"),
  sidebarLayout(
    sidebarPanel(
      # Subset sample
      
      # Select algorithm
      
      # FN/FP slider
      
      # Download buttons
    ),  # close 'sidebarPanel'
    tabsetPanel(
      tabPanel(
        "Exploratory Data Analysis", 
        plotOutput("plot")
      ),  # close EDA panel
      
      tabPanel(
        "SHAP vs LIME", 
        verbatimTextOutput("summary")
      ),  # close Explanation panel
      
      tabPanel(
        "Model Evaluation", 
        tableOutput("table")
      ),  # close Evaluation panel
      
      tabPanel(
        "Model Tuning",
        tableOutput("table")
      )  # close Tuning panel
    )  # close 'tabsetPanel'
  )  # close 'sidebarLayout'
)  # close 'fluidPage'


# Define server logic ==========================================================

server <- function(input, output) {} # close 'server' function

# Run the application ==========================================================
shinyApp(ui = ui, server = server)