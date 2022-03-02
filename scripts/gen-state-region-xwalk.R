################################################################################
# Script name: gen-state-region-xwalk.R
# Author: Arcenis Rojas
# E-mail: arcenis.rojas@tutanota.com
# Date created: 2/27/2022
#
# Script description: Generate a crosswalk to go from state name to state
#   abbreviation to state name to Census region
#
################################################################################

# Notes ========================================================================
# Census region comes from the 2020 Census crosswalk file and state
# abbreviations come from the stored R object.


# TODO Items ===================================================================



# Set up the environment =======================================================
# Clear the workspace
rm(list = ls())

# Load required packages
library(readxl)
library(dplyr)

# Load / declare helper functions


################################ Program start #################################

# Store the name of the census crosswalk file
census_xwalk_file <- file.path("./data", "state-geocodes-v2020.xlsx")

# Download the crosswalk file if it's not in the working directory
if (!file.exists(file.path("./data", census_xwalk_file))) {
  download.file(
    stringr::str_c(
      "https://www2.census.gov/programs-surveys/popest/geographies/2020/",
      basename(census_xwalk_file)
    ),
    destfile = census_xwalk_file
  )
}

# Read in the downloaded excel file
census_mapping <- readxl::read_excel(census_xwalk_file, skip = 5)

# Create the crosswalk
state_region_xwalk <- census_mapping %>%
  janitor::clean_names() %>%
  filter(!state_fips %in% "00") %>%
  select(region, state_name = "name") %>%
  left_join(
    census_mapping %>%
      janitor::clean_names() %>%
      filter(division %in% "0", state_fips %in% "00") %>%
      select(region, region_name = "name"),
    by = "region"
  ) %>%
  left_join(
    tibble(
      state_abb = c(state.abb, "DC"),
      state_name = c(state.name, "District of Columbia")
    ),
    by = "state_name"
  ) %>%
  select(-c(state_name, region)) %>%
  mutate(
    region_name = stringr::str_remove(region_name, "Region") %>% 
      stringr::str_trim("both"),
    state_abb = as_factor(state_abb)
  ) %>%
  arrange(region_name, state_abb)

# Store the crosswalk to the working directory
readr::write_rds(state_region_xwalk, "./data/state-region-crosswalk.Rds")

# Remove the Census file
file.remove(census_xwalk_file)
