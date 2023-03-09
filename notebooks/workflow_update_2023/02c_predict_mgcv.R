#!/usr/bin/env Rscript
# Code to read a pre-fitted GAM and use it to predict values for new data

args <- commandArgs(trailingOnly = TRUE)
suppressMessages(library(mgcv))
suppressMessages(library(ggplot2))
suppressMessages(library(dplyr))
suppressMessages(library(viridis))

# Unpack args
par <- args[1]

# Read datasets
df <- read.csv("gam_data.csv")
grid_df_yr <- read.csv("grid_df_yr.csv")

# Determine model to fit
if (par == "TOC_mgpl") {
    model <- readRDS("best_toc_model.rds")
} else if (par == "NO3_NO2_ugpl") {
    model <- readRDS("best_no3_model.rds")
} else {
    stop("Could not identify model to use.")
}

# Predict grid points (with std. errs.). NOTE: Std. errors are not currently used in
# the Python script. If you want to use them, change to 'type = 'link' and modify 
# the code to transform them correctly. See here:
# https://stats.stackexchange.com/a/33328/5467
preds <- predict.gam(model, grid_df_yr, se.fit = TRUE, , type = "response")
grid_df_yr$predicted <- preds$fit
grid_df_yr$stderr <- preds$se.fit

write.table(grid_df_yr,
    "preds.csv",
    sep = ",",
    row.names = FALSE,
    col.names = TRUE,
    quote = FALSE
)
