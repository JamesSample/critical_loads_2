#!/usr/bin/env Rscript
# Code to read a pre-fitted GAM and use it to predict values for new data

args <- commandArgs(trailingOnly = TRUE)
suppressMessages(library(mgcv))
suppressMessages(library(ggplot2))
suppressMessages(library(dplyr))
suppressMessages(library(viridis))

# Unpack args
par <- args[1]
model_type <- args[2]

# Read datasets
df <- read.csv("gam_data.csv")
grid_df_yr <- read.csv("grid_df_yr.csv")

# Determine model to fit
if ((model_type == "tensor") && (par == "TOC_mgpl")) {
    model <- readRDS("toc_tensor_model.rds")
} else if ((model_type == "spline") && (par == "TOC_mgpl")) {
    model <- readRDS("toc_spline_model.rds")
} else if ((model_type == "tensor") && (par == "NO3_NO2_ugpl")) {
    model <- readRDS("no3_tensor_model.rds")
} else if ((model_type == "spline") && (par == "NO3_NO2_ugpl")) {
    model <- readRDS("no3_spline_model.rds")
} else {
    stop("Could not identify model to use.")
}

# Predict grid points (with std. errs.)
preds <- predict.gam(model, grid_df_yr, se.fit = TRUE)
grid_df_yr$predicted <- preds$fit
grid_df_yr$stderr <- preds$se.fit

write.table(grid_df_yr,
    "preds.csv",
    sep = ",",
    row.names = FALSE,
    col.names = TRUE,
    quote = FALSE
)
