library(tidyverse)
library(readxl)

rm(list = ls())

regions_data_path <- "путь к папке с данными"

# data =================================================================
setwd(regions_data_path)

### 
data_1 <- read_excel("Данные.xlsx", 
                   sheet = "Лист", col_types = c("text", 
                                                            "numeric", "numeric", "numeric", 
                                                            "numeric", "numeric", "numeric", 
                                                            "numeric"),
                   na = "-")

data_2 <- read_excel("Данные.xlsx", 
                     sheet = "Лист 2", col_types = c("text", 
                                                   "numeric", "numeric", "numeric", 
                                                   "numeric", "numeric", "numeric", 
                                                   "numeric"),
                     na = "-")

data_3 <- read_excel("Данные.xlsx", 
                     sheet = "Лист 3", col_types = c("text", 
                                                   "numeric", "numeric", "numeric", 
                                                   "numeric", "numeric", "numeric", 
                                                   "numeric"),
                     na = "-")

## Creating long panel ---------------------------------------------------------

### List of all Rosstat datasets
regions_datasets_names <- ls()[sapply(ls(), function(x) any(is.data.frame(get(x))))]

regions_datasets <- lapply(regions_datasets_names, 
                           function(x) get(x))

names(regions_datasets) <- regions_datasets_names

### Manipulations
regions_datasets <- lapply(regions_datasets, rename, "region_rus" = "...1")


### Reshape
regions_datasets <- regions_datasets %>%
  names(.) %>%
  #walk(~ pivot_longer(regions_datasets[[.]], -region_rus, names_to = "year", values_to = .)) # Invisible output
  map(~ pivot_longer(regions_datasets[[.]], -region_rus, names_to = "year", values_to = .)) # Prints output to console


regions_datasets[[1]]

regions_datasets <- lapply(regions_datasets, mutate, 
                           year = str_replace_all(year, "/20..", ""))

### Joining
regions_rosstat <- regions_datasets %>% reduce(full_join, by = c("region_rus", "year"))


# Saving dataset ===============================================================
setwd(dirname(rstudioapi::getActiveDocumentContext()$path))

regions_rosstat %>% write_csv("regions_rosstat.csv")
