#' Suraj Udasi
#' Jan 27, 2024
#' A2 code for building models to predicting household spending trends

# libraries
library(rpart)
library(caret)
library(plyr)
library(dplyr)
library(MLmetrics)
library(vtreat)
library(rpart.plot)
library(ggplot2)
library(ggthemes)
library(maps)
library(data.table)
library(DataExplorer)
library(pROC)
library(knitr)
library(webshot)
library(psych)
library(viridis)


# Options
options(scipen=999)

# WD
setwd("~/Visualizing_Analyzing_Data_with_R/personalFiles/A2_Suraj Udasi")

# data - training
allTrainingFiles <- list.files(path = '~/Visualizing_Analyzing_Data_with_R/personalFiles/A2_Suraj Udasi/A2_Household_Spend/studentTables',
                               pattern = 'training',
                               full.names = T)

# Load the files and arrange them with a left join
bby_train <- lapply(allTrainingFiles, read.csv)
bby_train <- join_all(bby_train, by='tmpID', type='left')
bby_train_clean <- data.frame(bby_train) # to train the first linear model and test performance

# data - testing
allTestingFiles <- list.files(path = '~/Visualizing_Analyzing_Data_with_R/personalFiles/A2_Suraj Udasi/A2_Household_Spend/studentTables',
                              pattern = 'testing',
                              full.names = T)

# Load the files and arrange them with a left join
bby_test <- lapply(allTestingFiles, read.csv)
bby_test <- join_all(bby_test, by='tmpID', type='left')

# data - prospects
allProspectsFiles <- list.files(path = '~/Visualizing_Analyzing_Data_with_R/personalFiles/A2_Suraj Udasi/A2_Household_Spend/studentTables',
                                pattern = 'prospects',
                                full.names = T)

# Load the files and arrange them with a left join
bby_prospects <- lapply(allProspectsFiles, read.csv)
bby_prospects <- join_all(bby_prospects, by='tmpID', type='left')


# EDA
### Perform robust exploratory data analysis, drop any columns that you think don't make sense, 
# or are unethical to use in use case.  
# You can build visuals, tables, summaries and explore the data's overall integrity.

head(bby_train_clean)
str(bby_train_clean)

# Checking for outliers
ggplot(data = bby_train_clean, aes(y=yHat)) + 
  geom_boxplot() +
  scale_fill_brewer(palette = "Set2") +
  labs(title = "Household Spends",
       y = "yHat ($)") +
  theme_minimal() +
  theme(legend.position = "top")

# Let's drop the outliers in the top demi decile
quantile(bby_train_clean$yHat, probs = seq(.1,.95, by = .05))
dropAmt <- tail(quantile(bby_train_clean$yHat, probs = seq(.1,.95, by = .05)), 1)
bby_train_clean <- subset(bby_train_clean, bby_train_clean$yHat<dropAmt)

ggplot(data = bby_train_clean, aes(y=yHat)) + 
  geom_boxplot() +
  scale_fill_brewer(palette = "Set2") +
  labs(title = "Household Spends",
       y = "yHat ($)") +
  theme_minimal() +
  theme(legend.position = "top")

# Data Cleaning

# Defining a function to remove dollar signs and spaces, and convert to numeric
correctFormat <- function(column) {
  column <- gsub("\\$", "", column)  # Remove dollar signs
  column <- gsub(" ", "", column)    # Remove spaces
  return(as.numeric(column))         # Convert to numeric
}

# Apply the function to the columns with '$' sign
bby_train_clean$EstHomeValue <- correctFormat(bby_train_clean$EstHomeValue)
bby_train_clean$LandValue <- correctFormat(bby_train_clean$LandValue)

# Create a vector of numeric variable names
numeric_vars <- names(bby_train_clean)[sapply(bby_train_clean, is.numeric)]
summary(bby_train_clean[numeric_vars])
# Create a vector of categorical variable names
categorical_vars <- names(bby_train_clean)[sapply(bby_train_clean, is.character)]
sapply(bby_train_clean[categorical_vars], table)

# Feature Selection

#Count missing values in each column
missing_values <- sapply(bby_train_clean, function(x) {
  sum(is.na(x) | x == "" | x == " " | is.null(x)) 
})

missingValuesDF <- as.data.frame(missing_values)
missingValuesDF

# drop columns with missing values

# Calculate the threshold for 50% missing data
threshold <- 0.50 * nrow(bby_train_clean)

# Identify columns where missing values are more than 50%
dropCols <- names(missing_values[missing_values > threshold])
dropCols

# Drop these columns from the dataset
bby_train_clean <- bby_train_clean[, !(names(bby_train_clean) %in% dropCols)]
names(bby_train_clean)

#Drop columns with low relevance
bby_train_clean <- subset(bby_train_clean, select = 
                            -c(tmpID, EthnicDescription, BroadEthnicGroupings, 
                               lat, lon, county, city, TelephonesFullPhone, 
                               FirstName,LastName, fips, stateFips, state, 
                               PartiesDescription, ReligionsDescription, 
                               GunOwner, Veteran))


# Function to get unique entries and their counts for each column
get_unique_counts <- function(data) {
  lapply(data, function(x) {
    if (is.numeric(x)) {
      return(NULL)  # Skip numeric columns for unique value counts
    } else {
      return(table(x))  # Return a table of counts for unique values
    }
  })
}

# Apply the function to the bby_train_clean df
uniqueCounts <- get_unique_counts(bby_train_clean)
uniqueCounts

# Updated the vector of numeric variable names
numeric_vars <- names(bby_train_clean)[sapply(bby_train_clean, is.numeric)]
summary(bby_train_clean[numeric_vars])
# Update the vector of categorical variable names
categorical_vars <- names(bby_train_clean)[sapply(bby_train_clean, is.character)]
sapply(bby_train_clean[categorical_vars], table)

# Univariate Analysis

# Histograms for numeric variables
histograms <- lapply(numeric_vars, function(var) {
  ggplot(bby_train_clean, aes(x = get(var))) +
    geom_histogram() +
    labs(title = paste("Histogram of", var))
})

# Bar plots for categorical variables
bar_plots <- lapply(categorical_vars, function(var) {
  ggplot(bby_train_clean, aes(x = get(var))) +
    geom_bar() +
    labs(title = paste("Bar Plot of", var))
})

# SAMPLE
### Using the training data, create a validation set.  

# Using the Training data and Partitioning 80% Train set / 20% validation set
splitPercent_clean  <- round(nrow(bby_train_clean) %*% .8)
totalRecords_clean <- 1:nrow(bby_train_clean)
totalRecords_clean
set.seed(2017)
idx_clean          <- sample(totalRecords_clean, splitPercent_clean)

trainSet_clean <- bby_train_clean[idx_clean, ]
validationSet_clean  <- bby_train_clean[-idx_clean, ]

# Dimensions
dim(trainSet_clean)
dim(validationSet_clean)

# MODIFY 
### Using the training, create a design treatment plan.
#1 Identify the informative features (x variables)
#2 Identify the target variable (y variable)

# Get the column names of our data frame
names(bby_train_clean)

# Example of some variables and building a plan
infoFeatures <- names(bby_train_clean)[!names(bby_train_clean) %in% "yHat"]
infoFeatures
tarVariable <- 'yHat'
Plan <- designTreatmentsN(dframe      = bby_train_clean,
                          varlist     = infoFeatures,
                          outcomename = tarVariable)

# Apply the plan to all sections of the data
treatedTrain_0 <- prepare(Plan, trainSet_clean)
treatedValidation_0 <- prepare(Plan, validationSet_clean) 


# MODEL
### Fitting a linear model 1 - Cleaned all missing values, removed unwanted columns, removed outliers
fit <- lm(yHat~., treatedTrain_0)
fit_summary <- summary(fit)
fit_summary

# Model 2 - Parsimony Model of Original Model 

# First get the variable and p-values
pVals_clean <- data.frame(varNames = names(na.omit(coef(fit))),
                          pValues = summary(fit)$coefficients[,4])

keeps_clean <- subset(pVals_clean$varNames, pVals_clean$pValues<0.1) # Determine which variable names to keep 

treatedTrainParsimony_clean <- treatedTrain_0[,names(treatedTrain_0) %in% keeps_clean] # Remove unwanted columns

treatedTrainParsimony_clean$yHat <- treatedTrain_0$yHat # Append the dependent y-variable

# Refit the model
fit_parsi <- lm(yHat ~ ., treatedTrainParsimony_clean)
fit_parsi_summary <- summary(fit_parsi)
fit_parsi_summary

# ASSESS
### Make predictions
LinearTrainPredictions      <- predict(fit, treatedTrain_0)
LinearValidationPredictions <- predict(fit, treatedValidation_0)

LinearParsiTrainPredictions      <- predict(fit_parsi, treatedTrain_0)
LinearParsiValidationPredictions <- predict(fit_parsi, treatedValidation_0)

#Organize training set preds for Models
# Original Linear Model
Linear_trainingResults <-data.frame(actuals    = treatedTrain_0$yHat,
                                    predicted      = LinearTrainPredictions,
                                    residualErrors = treatedTrain_0$yHat-LinearTrainPredictions )
head(Linear_trainingResults)

Linear_validationResults <-data.frame(actuals  = treatedValidation_0$yHat,
                                      predicted      = LinearValidationPredictions,
                                      residualErrors = treatedValidation_0$yHat-LinearValidationPredictions )
head(Linear_validationResults)

# Parsimony model of original linear model
Linear_parsi_trainingResults <-data.frame(actuals        = treatedTrain_0$yHat,
                                          predicted      = LinearParsiTrainPredictions,
                                          residualErrors = treatedTrain_0$yHat-LinearParsiTrainPredictions )
head(Linear_parsi_trainingResults)

Linear_parsi_validationResults <-data.frame(actuals        = treatedValidation_0$yHat,
                                            predicted      = LinearParsiValidationPredictions,
                                            residualErrors = treatedValidation_0$yHat-LinearParsiValidationPredictions )
head(Linear_parsi_validationResults)

# RMSE
(Linear_trainRMSE <- MLmetrics::RMSE(Linear_trainingResults$predicted, 
                                     Linear_trainingResults$actuals))

(Linear_validationRMSE <- MLmetrics::RMSE(Linear_validationResults$predicted, 
                                          Linear_validationResults$actuals))

(Linear_parsi_tainRMSE <- MLmetrics::RMSE(Linear_parsi_trainingResults$predicted, 
                                          Linear_parsi_trainingResults$actuals))

(Linear_parsi_validationRMSE <- MLmetrics::RMSE(Linear_parsi_validationResults$predicted, 
                                                Linear_parsi_validationResults$actuals))

# MAPE
(Linear_trainMAPE <- MLmetrics::MAPE(Linear_trainingResults$predicted, 
                                     Linear_trainingResults$actuals))

(Linear_validationMAPE <- MLmetrics::MAPE(Linear_validationResults$predicted, 
                                          Linear_validationResults$actuals))

(Linear_parsi_trainMAPE <- MLmetrics::MAPE(Linear_parsi_trainingResults$predicted, 
                                           Linear_parsi_trainingResults$actuals))

(Linear_parsi_validationMAPE <- MLmetrics::MAPE(Linear_parsi_validationResults$predicted, 
                                                Linear_parsi_validationResults$actuals))


# Print the metrics
cat("Linear Original Training RMSE:", Linear_trainRMSE, "\n")
cat("Linear Original Validation RMSE:", Linear_validationRMSE, "\n")
cat("Linear Original Training MAPE:", Linear_trainMAPE, "\n")
cat("Linear Original Validation MAPE:", Linear_validationMAPE, "\n")

cat("Linear Parimony Training RMSE:", Linear_parsi_tainRMSE, "\n")
cat("Linear Parimony Validation RMSE:", Linear_parsi_validationRMSE, "\n")
cat("Linear Parimony Training MAPE:", Linear_parsi_trainMAPE, "\n")
cat("Linear Parimony Validation MAPE:", Linear_parsi_validationMAPE, "\n") 


# EDA
summary(bby_train)

names(bby_train)

#Drop columns with low relevance
bby_train <- subset(bby_train, select = 
                      -c(tmpID, MosaicZ4, HorseOwner, FirstName, LastName,
                         TelephonesFullPhone, fips, stateFips))

# Function to get unique entries and their counts for each column
get_unique_counts <- function(data) {
  lapply(data, function(x) {
    if (is.numeric(x)) {
      return(NULL)  # Skip numeric columns for unique value counts
    } else {
      return(table(x))  # Return a table of counts for unique values
    }
  })
}

# Apply the function to the bby_train df
uniqueCounts <- get_unique_counts(bby_train)
uniqueCounts

# Data Cleaning

# Defining a function to remove dollar signs and spaces, and convert to numeric
correctFormat <- function(column) {
  column <- gsub("\\$", "", column)  # Remove dollar signs
  column <- gsub(" ", "", column)    # Remove spaces
  return(as.numeric(column))         # Convert to numeric
}

# Apply the function to the columns with '$' sign
bby_train$EstHomeValue <- correctFormat(bby_train$EstHomeValue)
bby_train$LandValue <- correctFormat(bby_train$LandValue)
bby_train$HomePurchasePrice <- correctFormat(bby_train$HomePurchasePrice)

# Apply the function to the testing and prospects set
bby_test$EstHomeValue <- correctFormat(bby_test$EstHomeValue)
bby_test$LandValue <- correctFormat(bby_test$LandValue)
bby_test$HomePurchasePrice <- correctFormat(bby_test$HomePurchasePrice)

bby_prospects$EstHomeValue <- correctFormat(bby_prospects$EstHomeValue)
bby_prospects$LandValue <- correctFormat(bby_prospects$LandValue)
bby_prospects$HomePurchasePrice <- correctFormat(bby_prospects$HomePurchasePrice)

unique_counts <- get_unique_counts(bby_train)
unique_counts

# # Checking for outliers
# ggplot(data = bby_train, aes(y=yHat)) + geom_boxplot() + theme_gdocs()
# 
# # Let's drop the outliers in the top demi decile
# quantile(bby_train$yHat, probs = seq(.1,.95, by = .05))
# dropAmt <- tail(quantile(bby_train$yHat, probs = seq(.1,.95, by = .05)), 1)
# bby_train <- subset(bby_train, bby_train$yHat<dropAmt)
# 
# 
# ggplot(data = bby_train, aes(y=yHat)) + geom_boxplot() + theme_gdocs()

# SAMPLE
### Using the training data, create a validation set.  

# Using the Training data and Partitioning 80% Train set / 20% validation set
splitPercent  <- round(nrow(bby_train) %*% .8)
totalRecords <- 1:nrow(bby_train)
totalRecords
set.seed(2017)
idx          <- sample(totalRecords, splitPercent)

trainSet <- bby_train[idx, ]
validationSet  <- bby_train[-idx, ]

# Dimensions
dim(trainSet)
dim(validationSet)

# MODIFY 
### Using the training, create a design treatment plan.
#1 Identify the informative features (x variables)
#2 Identify the target variable (y variable)

# Get the column names of our data frame
names(bby_train)

# Example of some variables and building a plan
informartiveFeatures <- names(bby_train)[!names(bby_train) %in% "yHat"]
informartiveFeatures
targetVariable <- 'yHat'
dataPlan <- designTreatmentsN(dframe      = bby_train,
                              varlist     = informartiveFeatures,
                              outcomename = targetVariable)

# Apply the plan to all sections of the data
treatedTrain <- prepare(dataPlan, trainSet)
treatedValidation <- prepare(dataPlan, validationSet) 
treatedTest <- prepare(dataPlan, bby_test) # this is the data set from repeating the read.csv section but with the test CSV files
treatedProspects <- prepare(dataPlan, bby_prospects) #this is the data set from repeating read.csv but with the prospect CSV files

# MODEL
### Fitting a linear model 1 - Cleaned all missing values, removed unwanted columns, removed outliers
fit1 <- lm(yHat~., treatedTrain)
fit1_summary <- summary(fit1)
fit1_summary

# Model 2 - Parsimony Model of Model 1

# First get the variable and p-values
pVals <- data.frame(varNames = names(na.omit(coef(fit1))),
                    pValues = summary(fit1)$coefficients[,4])

keeps <- subset(pVals$varNames, pVals$pValues<0.05) # Determine which variable names to keep 

treatedTrainParsimony <- treatedTrain[,names(treatedTrain) %in% keeps] # Remove unwanted columns

treatedTrainParsimony$yHat <- treatedTrain$yHat # Append the dependent y-variable

# Refit the model
fit2 <- lm(yHat ~ ., treatedTrainParsimony)
fit2_summary <- summary(fit2)
fit2_summary

# Model 3 - Random Forest

# Set up cross-validation
train_control_rf <- trainControl(method="cv", number=10, search="random")

# Define the tuning grid for Random Forest
tune_grid_rf <- expand.grid(
  .mtry = c(2, sqrt(ncol(treatedTrain)-1), (ncol(treatedTrain)-1)/3),
  .splitrule = "variance",
  .min.node.size = c(5, 10)
)

# Train the Random Forest model
fit_rf <- train(
  yHat ~ ., 
  data = treatedTrain,
  method = "ranger", # 'ranger' is a faster implementation of random forest
  trControl = train_control_rf,
  tuneGrid = tune_grid_rf,
  importance = 'impurity' # Option to get variable importance
)

# Check the results
print(fit_rf)
plot(fit_rf)

# Model 4 - XGBoost
# Set up cross-validation
train_control_xgb <- trainControl(method = "cv", number = 10, search = "grid")

# Define the tuning grid for XGBoost
tune_grid_xgb <- expand.grid(
  nrounds = 100,  # Number of boosting rounds (consider starting lower for faster runs)
  eta = c(0.01, 0.1),  # Learning rate
  max_depth = c(3, 6, 9),  # Max depth of trees
  gamma = 0,  # Minimum loss reduction required to make a further partition on a leaf node
  colsample_bytree = 0.8,  # Subsample ratio of columns when constructing each tree
  min_child_weight = 1,  # Minimum sum of instance weight needed in a child
  subsample = 0.8  # Subsample ratio of the training instances
)

# Train the XGBoost model
fit_xgb <- train(
  yHat ~ ., 
  data = treatedTrain,
  method = "xgbTree",
  trControl = train_control_xgb,
  tuneGrid = tune_grid_xgb,
  metric = "RMSE"
)

# Check the results
print(fit_xgb)

# Model 5 - Decision Tree
# Set up cross-validation
train_control_dt <- trainControl(method="cv", number=10)

# Define the tuning grid for Decision Tree
# Note: 'cp' is the complexity parameter, used to control tree complexity
# Lower 'cp' values mean more complex trees
tune_grid_dt <- expand.grid(.cp = seq(0.001, 0.1, by = 0.001))

# Train the Decision Tree model
fit_dt <- train(
  yHat ~ ., 
  data = treatedTrain,
  method = "rpart",
  trControl = train_control_dt,
  tuneGrid = tune_grid_dt
)

# Check the results
fit_dt

# Plot the decision tree
plot(fit_dt)

# ASSESS
### Make predictions
lm1TrainPredictions      <- predict(fit1, treatedTrain)
lm1ValidationPredictions <- predict(fit1, treatedValidation)

lm2TrainPredictions      <- predict(fit2, treatedTrain)
lm2ValidationPredictions <- predict(fit2, treatedValidation)

RFTrainPredictions      <- predict(fit_rf, treatedTrain)
RFValidationPredictions <- predict(fit_rf, treatedValidation)

XGBTrainPredictions <- predict(fit_xgb, treatedTrain)
XGBValidationPredictions <- predict(fit_xgb, treatedValidation)

DTTrainPredictions <- predict(fit_dt, treatedTrain)
DTValidationPredictions <- predict(fit_dt, treatedValidation)

# Next, calculate the RMSE for these sections.  Look for consistency.  At this point you could go back and adjust the model by adding or subtracting variables.  

#Organize training set preds for Models
# Linear Model 1
lm1_trainingResults <-data.frame(actuals    = treatedTrain$yHat,
                                 predicted      = lm1TrainPredictions,
                                 residualErrors = treatedTrain$yHat-lm1TrainPredictions )
head(lm1_trainingResults)

lm1_validationResults <-data.frame(actuals  = treatedValidation$yHat,
                                   predicted      = lm1ValidationPredictions,
                                   residualErrors = treatedValidation$yHat-lm1ValidationPredictions )
head(lm1_validationResults)

# Linear Model 2
lm2_trainingResults <-data.frame(actuals        = treatedTrain$yHat,
                                 predicted      = lm2TrainPredictions,
                                 residualErrors = treatedTrain$yHat-lm2TrainPredictions )
head(lm2_trainingResults)

lm2_validationResults <-data.frame(actuals        = treatedValidation$yHat,
                                   predicted      = lm2ValidationPredictions,
                                   residualErrors = treatedValidation$yHat-lm2ValidationPredictions )
head(lm2_validationResults)


# Random Forest 
RF_trainingResults <-data.frame(actuals        = treatedTrain$yHat,
                                predicted      = RFTrainPredictions,
                                residualErrors = treatedTrain$yHat-RFTrainPredictions )
head(RF_trainingResults)

RF_validationResults <-data.frame(actuals        = treatedValidation$yHat,
                                  predicted      = RFValidationPredictions,
                                  residualErrors = treatedValidation$yHat-RFValidationPredictions )
head(RF_validationResults)

#XGBoost Model
XGB_trainingResults <- data.frame(
  actuals = treatedTrain$yHat,
  predicted = XGBTrainPredictions,
  residualErrors = treatedTrain$yHat - XGBTrainPredictions
)

XGB_validationResults <- data.frame(
  actuals = treatedValidation$yHat,
  predicted = XGBValidationPredictions,
  residualErrors = treatedValidation$yHat - XGBValidationPredictions
)

# Decision Tree
DT_trainingResults <- data.frame(
  actuals = treatedTrain$yHat,
  predicted = DTTrainPredictions,
  residualErrors = treatedTrain$yHat - DTTrainPredictions)

DT_validationResults <- data.frame(
  actuals = treatedValidation$yHat,
  predicted = DTValidationPredictions,
  residualErrors = treatedValidation$yHat - DTValidationPredictions)

# R-squared

Lm1_adj_Rsquared <- fit1_summary$adj.r.squared
Lm2_adj_Rsquared <- fit2_summary$adj.r.squared
RF_Rsquared <- max(fit_rf$results$Rsquared)
XGB_Rsquared <- max(fit_xgb$results$Rsquared)

# Find the row index in the results where the RMSE is the smallest
DTBestMod <- which.min(fit_dt$results$RMSE)
best_cp <- fit_dt$results$cp[DTBestMod] # Extract the cp value for the best model
DT_Rsquared <- fit_dt$results$Rsquared[DTBestMod] # Extract the Rsquared value for the best model
DT_Rsquared

# RMSE
(lm1_trainRMSE <- MLmetrics::RMSE(lm1_trainingResults$predicted, 
                                  lm1_trainingResults$actuals))

(lm1_validationRMSE <- MLmetrics::RMSE(lm1_validationResults$predicted, 
                                       lm1_validationResults$actuals))

(lm2_trainRMSE <- MLmetrics::RMSE(lm2_trainingResults$predicted, 
                                  lm2_trainingResults$actuals))

(lm2_validationRMSE <- MLmetrics::RMSE(lm2_validationResults$predicted, 
                                       lm2_validationResults$actuals))


(RF_trainRMSE <- MLmetrics::RMSE(RF_trainingResults$predicted, 
                                 RF_trainingResults$actuals))

(RF_validationRMSE <- MLmetrics::RMSE(RF_validationResults$predicted, 
                                      RF_validationResults$actuals))

(XGB_trainRMSE <- MLmetrics::RMSE(XGB_trainingResults$predicted, 
                                  XGB_trainingResults$actuals))

(XGB_validationRMSE <- MLmetrics::RMSE(XGB_validationResults$predicted, 
                                       XGB_validationResults$actuals))

(DT_trainRMSE <- MLmetrics::RMSE(DT_trainingResults$predicted, 
                                 DT_trainingResults$actuals))

(DT_validationRMSE <- MLmetrics::RMSE(DT_validationResults$predicted, 
                                      DT_validationResults$actuals))



# MAPE
(lm1_trainMAPE <- MLmetrics::MAPE(lm1_trainingResults$predicted, 
                                  lm1_trainingResults$actuals))

(lm1_validationMAPE <- MLmetrics::MAPE(lm1_validationResults$predicted, 
                                       lm1_validationResults$actuals))

(lm2_trainMAPE <- MLmetrics::MAPE(lm2_trainingResults$predicted, 
                                  lm2_trainingResults$actuals))

(lm2_validationMAPE <- MLmetrics::MAPE(lm2_validationResults$predicted, 
                                       lm2_validationResults$actuals))

(RF_trainMAPE <- MLmetrics::MAPE(RF_trainingResults$predicted, 
                                 RF_trainingResults$actuals))

(RF_validationMAPE <- MLmetrics::MAPE(RF_validationResults$predicted, 
                                      RF_validationResults$actuals))

(XGB_trainMAPE <- MLmetrics::MAPE(XGB_trainingResults$predicted, 
                                  XGB_trainingResults$actuals))

(XGB_validationMAPE <- MLmetrics::MAPE(XGB_validationResults$predicted, 
                                       XGB_validationResults$actuals))

(DT_trainMAPE <- MLmetrics::MAPE(DT_trainingResults$predicted, 
                                 DT_trainingResults$actuals))

(DT_validationMAPE <- MLmetrics::MAPE(DT_validationResults$predicted, 
                                      DT_validationResults$actuals))

# Create a data frame to hold the model comparison metrics
model_comparison <- data.frame(
  Metric = c("Rsquared", "Train_RMSE", "Validation_RMSE", "Train_MAPE", "Validation_MAPE"),
  Linear_1 = c(Lm1_adj_Rsquared, lm1_trainRMSE, lm1_validationRMSE, lm1_trainMAPE, lm1_validationMAPE),
  Linear_2 = c(Lm2_adj_Rsquared, lm2_trainRMSE, lm2_validationRMSE, lm2_trainMAPE, lm2_validationMAPE),
  Random_Forest = c(RF_Rsquared, RF_trainRMSE, RF_validationRMSE, RF_trainMAPE, RF_validationMAPE),
  XG_Boost      = c(XGB_Rsquared, XGB_trainRMSE, XGB_validationRMSE, XGB_trainMAPE, XGB_validationMAPE),
  Decision_Tree = c(DT_Rsquared, DT_trainRMSE, DT_validationRMSE, DT_trainMAPE, DT_validationMAPE)
)
model_comparison

# ASSESS the 5 models on a test set
### Make predictions
lm1TestPredictions      <- predict(fit1, treatedTest)

lm2TestPredictions      <- predict(fit2, treatedTest)

RFTestPredictions      <- predict(fit_rf, treatedTest)

XGBTestPredictions <- predict(fit_xgb, treatedTest)

DTTestPredictions <- predict(fit_dt, treatedTest)

#Organize testing set preds for Models
# Linear Model 1
lm1_testingResults <-data.frame(actuals    = treatedTest$yHat,
                                predicted      = lm1TestPredictions,
                                residualErrors = treatedTest$yHat-lm1TestPredictions )
head(lm1_testingResults)

# Linear Model 2
lm2_testingResults <-data.frame(actuals        = treatedTest$yHat,
                                predicted      = lm2TestPredictions,
                                residualErrors = treatedTest$yHat-lm2TestPredictions )
head(lm2_testingResults)

# Random Forest 
RF_testingResults <-data.frame(actuals        = treatedTest$yHat,
                               predicted      = RFTestPredictions,
                               residualErrors = treatedTest$yHat-RFTestPredictions )
head(RF_testingResults)

#XGBoost Model
XGB_testingResults <- data.frame(
  actuals = treatedTest$yHat,
  predicted = XGBTestPredictions,
  residualErrors = treatedTest$yHat - XGBTestPredictions
)

head(XGB_testingResults)

# Decision Tree
DT_testingResults <- data.frame(
  actuals = treatedTest$yHat,
  predicted = DTTestPredictions,
  residualErrors = treatedTest$yHat - DTTestPredictions)

head(DT_testingResults)


# RMSE
(lm1_testRMSE <- MLmetrics::RMSE(lm1_testingResults$predicted, 
                                 lm1_testingResults$actuals))

(lm2_testRMSE <- MLmetrics::RMSE(lm2_testingResults$predicted, 
                                 lm2_testingResults$actuals))

(RF_testRMSE <- MLmetrics::RMSE(RF_testingResults$predicted, 
                                RF_testingResults$actuals))

(XGB_testRMSE <- MLmetrics::RMSE(XGB_testingResults$predicted, 
                                 XGB_testingResults$actuals))

(DT_testRMSE <- MLmetrics::RMSE(DT_testingResults$predicted, 
                                DT_testingResults$actuals))

# MAPE
(lm1_testMAPE <- MLmetrics::MAPE(lm1_testingResults$predicted, 
                                 lm1_testingResults$actuals))

(lm2_testMAPE <- MLmetrics::MAPE(lm2_testingResults$predicted, 
                                 lm2_testingResults$actuals))

(RF_testMAPE <- MLmetrics::MAPE(RF_testingResults$predicted, 
                                RF_testingResults$actuals))

(XGB_testMAPE <- MLmetrics::MAPE(XGB_testingResults$predicted, 
                                 XGB_testingResults$actuals))

(DT_testMAPE <- MLmetrics::MAPE(DT_testingResults$predicted, 
                                DT_testingResults$actuals))


# Compare the training, validation and test set RMSE.  Look for consistency.

# Create a data frame to hold the model comparison metrics and compare it to test set
model_comp_test <- data.frame(
  Metric = c("Rsquared", "Train_RMSE", "Validation_RMSE", "Test_RMSE","Train_MAPE", "Validation_MAPE", "Test_MAPE"),
  Linear_1 = c(Lm1_adj_Rsquared, lm1_trainRMSE, lm1_validationRMSE, lm1_testRMSE, lm1_trainMAPE, lm1_validationMAPE, lm1_testMAPE),
  Linear_2 = c(Lm2_adj_Rsquared, lm2_trainRMSE, lm2_validationRMSE, lm2_testRMSE, lm2_trainMAPE, lm2_validationMAPE, lm2_testMAPE),
  Random_Forest = c(RF_Rsquared, RF_trainRMSE, RF_validationRMSE, RF_testRMSE, RF_trainMAPE, RF_validationMAPE, RF_testMAPE),
  XG_Boost      = c(XGB_Rsquared, XGB_trainRMSE, XGB_validationRMSE, XGB_testRMSE, XGB_trainMAPE, XGB_validationMAPE, XGB_testMAPE),
  Decision_Tree = c(DT_Rsquared, DT_trainRMSE, DT_validationRMSE, DT_testRMSE, DT_trainMAPE, DT_validationMAPE, DT_testMAPE)
)
model_comp_test

# The best model you have based on the RMSE score in the test set that is also fairly consistent among other data sections - Liner_2 (fit2)
# Using the best possible model make predictions on the prospect set.
prospectPredictions_rf <- predict(fit_rf, treatedProspects)
prospectPredictions_xgb <- predict(fit_xgb, treatedProspects)

# Create a data frame to store predictions from all models
predictions_df <- data.frame(tmpID = bby_prospects$tmpID)

# Add predictions from each model to the data frame
predictions_df$Predictions_rf <- prospectPredictions_rf
predictions_df$Predictions_xgb <- prospectPredictions_xgb
head(predictions_df)

# Calculate the mean prediction value across all models and add it as a new column
predictions_df$meanPrediction_rf_xgb <- rowMeans(predictions_df[, -1])  # Exclude the tmpID column from mean calculation
head(predictions_df)

# Column bind the predictions to the prospect CSV; finish the case submissions. 
# Convert predictions to a data frame
# Join the predictions back to your original bby_prospects dataset
prospectPredictions_df <- merge(bby_prospects, predictions_df, by = "tmpID", all.x = TRUE)
head(prospectPredictions_df)


bby_prospects_to_submit <- select(prospectPredictions_df, tmpID, meanPrediction_rf_xgb)
head(bby_prospects_to_submit)
# Write to CSV
write.csv(bby_prospects_to_submit, "prospect_predictions_SU.csv", row.names = FALSE)


str(prospectPredictions_df)
summary(prospectPredictions_df)


#Keep only columns of importance 
bby_prospects_predicted <- subset(prospectPredictions_df, select = 
                                    c(ResidenceHHGenderDescription, PresenceOfChildrenCode,
                                      HomeOwnerRenter, MosaicZ4, MedianEducationYears, 
                                      OccupationIndustry, Gender, Age, storeVisitFrequency,
                                      meanPrediction_rf_xgb))
names(bby_prospects_predicted)
quantile(bby_prospects_predicted$meanPrediction_rf_xgb, probs = seq(0.1, 0.95, by = 0.05))

spendingQuantiles <- quantile(bby_prospects_predicted$meanPrediction_rf_xgb, probs = seq(0.1, 0.95, by = 0.05))

bby_prospects_predicted$SpendingCategory <- cut(
  bby_prospects_predicted$meanPrediction_rf_xgb,
  breaks = c(-Inf, spendingQuantiles, Inf),
  labels = c("Lowest 10%", "10%-15%", "15%-20%", "20%-25%", "25%-30%", "30%-35%", "35%-40%", "40%-45%", 
             "45%-50%", "50%-55%", "55%-60%", "60%-65%", "65%-70%", "70%-75%", "75%-80%", "80%-85%", 
             "85%-90%", "90%-95%", "Top 5%"),
  include.lowest = TRUE
)

# Filter the dataset for the top 5% spending prospects
top5Percent_prospects <- subset(bby_prospects_predicted, SpendingCategory == "Top 5%")
head(top5Percent_prospects)

# Write to CSV
write.csv(top5Percent_prospects, "top5Percent_prospects_analysis.csv", row.names = FALSE)


# Summary Statistics
summary_statistics <- top5Percent_prospects %>% 
  summarise(
    MeanAge = mean(Age, na.rm = TRUE),
    MedianAge = median(Age, na.rm = TRUE),
    MeanSpending = mean(meanPrediction_rf_xgb, na.rm = TRUE),
    MedianSpending = median(meanPrediction_rf_xgb, na.rm = TRUE)
  )

# Segmentation Analysis
# Assuming 'MosaicZ4' is the column with Mosaic codes:
table(top5Percent_prospects$MosaicZ4) # To see how many categories and their frequencies
mosaic_table <- table(top5Percent_prospects$MosaicZ4)
# Filter the table for codes with frequencies > 1 and < 200
filtered_mosaic <- mosaic_table[mosaic_table > 1 & mosaic_table < 200]
# Filter the original dataframe to keep only the rows with the selected Mosaic codes
mosaic_data <- top5Percent_prospects[top5Percent_prospects$MosaicZ4 %in% names(filtered_mosaic), ]
mosaic_data


# Visualizations
# Histogram of predicted spending
ggplot(top5Percent_prospects, aes(x = meanPrediction_rf_xgb)) +
  geom_histogram(binwidth = 10, fill = "steelblue", color = "black") +
  theme_minimal() +
  labs(x = "Predicted Spending", y = "Count", title = "Distribution of Predicted Spending of Top 5%")

# Create the bar chart
ggplot(mosaic_data, aes(x = MosaicZ4)) +
  geom_bar(fill = "steelblue", color = "black") +
  theme_minimal(base_size = 12) +
  theme(axis.text.x = element_text( vjust = 0.5, hjust=1)) +
  labs(title = "Frequency of Mosaic Codes",
       x = "Mosaic Category",
       y = "Frequency") +
  theme(plot.title = element_text(hjust = 0.5)) # Center the plot title



# Box plot of spending by Mosaic codes
ggplot(mosaic_data, aes(x = MosaicZ4, y = meanPrediction_rf_xgb)) + 
  geom_boxplot() +
  theme(axis.text.x = element_text(hjust = 1)) +
  labs(title = "Predicted Spend by Mosaic Code for Top 5% of Prospects", 
       x = "Mosaic Code", 
       y = "Predicted Spend")



# Correlation
correlation_matrix <- cor(top5Percent_prospects %>% 
                            select(Age, meanPrediction_rf_xgb, storeVisitFrequency), use = "complete.obs")



# End