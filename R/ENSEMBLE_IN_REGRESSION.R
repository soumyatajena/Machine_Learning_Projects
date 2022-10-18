# ENSEMBLE ---------------------------------------------------------------------

rm(list = ls())
setwd("C:/Users/soumy/Soumyata_Jena/GSU/FallSemester/BigDataAnalytics/Projects")

library(adabag)
library(rpart) 
library(caret)
#install.packages("klaR")
library(klaR)


rawData = read.csv('media prediction and its cost.csv')
print(dim(rawData))
#[1] 60428    40


#methods

#encode_ordinal <- function(x, order = unique(x)) {
#  x <- as.numeric(factor(x, levels = order, exclude = NULL))
#  x
#}

#summary of dataset
summary(rawData)

#checking for null values in columns
lapply(rawData,function(x) { length(which(is.na(x)))})
#There are no null values in the dataset

#finding numerical and categorical data
str(rawData)
numCols = unlist(lapply(rawData, is.numeric))         # Identify numeric columns
catCols = unlist(lapply(rawData, is.character))
sum(catCols) #17 categorical columns
sum(numCols) #23 numerical columns

# Heat map of correlation matrix
#install.packages("reshape2")
library(reshape2)

# creating correlation matrix
corrMmat <- round(cor(rawData[numCols]),2)

# reduce the size of correlation matrix
meltedCorrMat <- melt(corrMmat)
head(meltedCorrMat)

# plotting the correlation heatmap
library(ggplot2)
ggplot(data = meltedCorrMat, aes(x=Var2, y=Var1,
                                 fill=value)) +
  geom_tile() +
  geom_text(aes(Var2, Var1, label = value),
            color = "black", size = 4)

#Based on the heatmap we can reduce highly correlated columns with values >= 0.8
rawData = within(rawData, rm('salad_bar', 'gross_weight', 'avg_cars_at.home.approx..1', 'meat_sqft','store_sales.in.millions.', 'store_cost.in.millions.', 'store_sqft','food_category', 'food_department', 'food_family', 'gender', 'marital_status',
                             'brand_name', 'avg..yearly_income', 'education', 'member_card', 'houseowner', 
                             'sales_country', 'occupation','recyclable_package', 'low_fat'))

#again checking numerical and categorical columns
numCols = unlist(lapply(rawData, is.numeric))         # Identify numeric columns
catCols = unlist(lapply(rawData, is.character))
sum(catCols) #5 categorical columns
sum(numCols) #14 numerical columns

#deciding encoding techniques for the categorical columns
#1 story type
unique(rawData$store_type)
# [1] Deluxe Supermarket  Supermarket         Gourmet Supermarket   Small Grocery       Mid-Size Grocery 
# we can use ordinal encoding here 

# rawData$store_type = encode_ordinal(rawData$store_type,c("Mid-Size Grocery","Small Grocery","Gourmet Supermarket","Supermarket","Deluxe Supermarket"))
rawData$store_type = encode_ordinal(rawData$store_type)
rawData$store_city <- encode_ordinal(rawData$store_city)
rawData$store_state <- encode_ordinal(rawData$store_state)
rawData$media_type <- encode_ordinal(rawData$media_type)
rawData$promotion_name <- encode_ordinal(rawData$promotion_name)


#splitting the dataset
set.seed(2)
colnames(rawData) <- make.names(colnames(rawData))
train.index <- sample(c(1:dim(rawData)[1]), dim(rawData)[1]*0.7)
train.df <- rawData[train.index, ]
valid.df <- rawData[-train.index, ]

typeof(rawData)

# Build an Ensemble Model with Multiple Types of Models
# Defining the training controls for multiple models
fitControl <- trainControl(
  method = "cv",
  number = 5,
  savePredictions = 'final',
  classProbs = T)

#Defining the predictors and outcome
test.features = subset(valid.df, select=-c(cost))
test.target = subset(valid.df, select=cost)[,1]

# RANDOM FOREST ----------------------------------------------------------------

library(randomForest)
#Training a random forest model
rf <- randomForest(cost ~ ., data =  train.df, ntree = 50, 
                      mtry = 4, nodesize = 5, importance = TRUE) 
#Predicting using random forest model
predictions.rf = as.numeric(predict(rf, newdata = test.features))
#Mean Absolute Error
print(mean(abs(test.target - predictions.rf)))
# RMSE
print(sqrt(mean((test.target - predictions.rf)^2)))
# R2
print(cor(test.target, predictions.rf) ^ 2)

plot(test.target, predictions.rf, col = c("red", "blue"), 
     main = 'RF Real vs Predicted')
abline(0, 1, lwd = 2)

# BAGGING ----------------------------------------------------------------------
# done to reduce variance
rf_bagging = randomForest(cost ~ ., data = train.df, mtry = 19, 
                          importance = TRUE, ntrees = 100)
predictions.bag.rf = as.numeric(predict(rf_bagging, newdata = test.features))
#Mean Absolute Error
print(mean(abs(test.target - predictions.bag.rf)))
# RMSE
print(sqrt(mean((test.target - predictions.rf)^2)))
# R2
print(cor(test.target, predictions.bag.rf) ^ 2)

plot(test.target, predictions.bag.rf, col = c("red", "blue"), 
     main = 'RF Real vs Predicted')
abline(0, 1, lwd = 2)

# BOOSTING ---------------------------------------------------------------------
# focused on reducing the bias

library(gbm)
rf_boosting = gbm(cost ~ ., data = train.df, distribution = "gaussian", 
                    n.trees = 5000, interaction.depth = 4, shrinkage = 0.01)

predictions.boost.rf = as.numeric(predict(rf_boosting, newdata = test.features))
#Mean Absolute Error
print(mean(abs(test.target - predictions.boost.rf)))
# RMSE
print(sqrt(mean((test.target - predictions.rf)^2)))
# R2
print(cor(test.target, predictions.boost.rf) ^ 2)

plot(test.target, predictions.boost.rf, col = c("red", "blue"), 
     main = 'RF Real vs Predicted')
abline(0, 1, lwd = 2)


# XGBOOST ----------------------------------------------------------------------

#install.packages('xgboost')     # for fitting the adaboost model

#install.packages('caret')       # for general data preparation and model fitting

library(xgboost)
library(caret)

#define predictor and response variables in training set
train_x = data.matrix(train.df[, -ncol(train.df)])
train_y = train.df[,ncol(train.df)]

#define predictor and response variables in testing set
test_x = data.matrix(valid.df[, -ncol(valid.df)])
test_y = valid.df[, ncol(valid.df)]

#define final training and testing sets
xgb_train = xgb.DMatrix(data = train_x, label = train_y)
xgb_test = xgb.DMatrix(data = test_x, label = test_y)

#defining a watchlist
watchlist = list(train=xgb_train, test=xgb_test)

#fit XGBoost model and display training and testing data at each iteartion
model = xgb.train(data = xgb_train, max.depth = 5, watchlist=watchlist, nrounds = 100)
print(model)

#define final model
model_xgboost = xgboost(data = xgb_train, max.depth = 5, nrounds = 86, verbose = 0)
summary(model_xgboost)

#use model to make predictions on test data
pred_y = predict(model_xgboost, xgb_test)

#Mean Absolute Error
print(mean(abs(test.target - pred_y)))
# RMSE
print(sqrt(mean((test.target - predictions.rf)^2)))
# R2
print(cor(test.target, pred_y) ^ 2)

plot(test_y, pred_y, col = c("red", "blue"), type = "p",
     main = 'XGBoost Real vs Predicted')
abline(0, 1, lwd = 2)


# performance metrics on the test data

#Mean Absolute Error
print(mean(abs(test_y - pred_y)))
# RMSE
print(sqrt(mean((test.target - predictions.rf)^2)))
# R2
print(cor(test_y, pred_y) ^ 2)

#output
# RMSE
#1.076552
# R2
#0.9952771

# Ensemble using Averaging -------------------------------------------------------------------------------


lm <- lm(cost ~ .,train.df)
pred.lm <- predict(lm,valid.df)

rf <- randomForest(cost ~ ., data =  train.df, ntree = 50, 
                   mtry = 4, nodesize = 5, importance = TRUE) 
pred.rf = as.numeric(predict(rf, newdata = test.features))

decisionTree <- train(
  cost ~ .,
  data = train.df,
  method = 'rpart2',
  # preProcess = c("center", "scale")
)
pred.dt = predict(decisionTree, newdata = test.features)

grid1 <- expand.grid(.k = seq(2, 4, by = 1))
control <- trainControl(method = "cv")
# train the model
knn <- train(cost ~ ., data = train.df,
             method = "knn",
             trControl = control,
             tuneGrid = grid1)
pred.knn <- predict(knn, newdata = valid.df)

# Taking average of predicted values
valid.df$pred_avg<-(pred.lm+pred.rf+pred.dt+pred.knn)/4
print(valid.df$pred_avg)

#Mean Absolute Error
print(mean(abs(test.target - valid.df$pred_avg)))
# RMSE
print(sqrt(mean((test.target - valid.df$pred_avg)^2)))
# R2
print(cor(test.target, valid.df$pred_avg) ^ 2)

plot(test_y, valid.df$pred_avg, col = c("red", "blue"), type = "p",
     main = 'Averaging Real vs Predicted')
abline(0, 1, lwd = 2)

# Stacking ---------------------------------------------------------

#install.packages('tidyverse')
#install.packages('h2o')

library(tidyverse)
library(h2o)

# initialize the h2o
h2o.init()
# create the train and test h2o data frames

train_df_h2o<-as.h2o(train.df)
test_df_h2o<-as.h2o(valid.df)

# Identify predictors and response
y <- "cost"
x <- setdiff(names(train_df_h2o), y)

# Number of CV folds (to generate level-one data for stacking)
nfolds <- 5
# 1. Generate a 3-model ensemble (GBM + RF + Logistic)
# Train & Cross-validate a GBM
my_gbm <- h2o.gbm(x = x,
                  y = y,
                  training_frame = train_df_h2o,
                  nfolds = nfolds,
                  keep_cross_validation_predictions = TRUE,
                  seed = 5)

# Train & Cross-validate a RF
my_rf <- h2o.randomForest(x = x,
                          y = y,
                          training_frame = train_df_h2o,
                          nfolds = nfolds,
                          keep_cross_validation_predictions = TRUE,
                          seed = 5)

# Train & Cross-validate a LR
my_deeplearning <- h2o.deeplearning(x = x,
                 y = y,
                 training_frame = train_df_h2o,
                 nfolds = nfolds,
                 keep_cross_validation_predictions = TRUE,
                 seed = 5)

# Train a stacked random forest ensemble using the GBM, RF and LR above
ensemble <- h2o.stackedEnsemble(x = x,
                                y = y,
                                metalearner_algorithm="drf",
                                training_frame = train_df_h2o,
                                base_models = list(my_gbm, my_rf, my_deeplearning))

print(ensemble)

# Eval ensemble performance on a test set
perf <- h2o.performance(ensemble, newdata = test_df_h2o)
print(perf)

# Compare to base learner performance on the test set
perf_gbm_test <- h2o.performance(my_gbm, newdata = test_df_h2o)
print(perf_gbm_test)
perf_rf_test <- h2o.performance(my_rf, newdata = test_df_h2o)
print(perf_rf_test)
perf_dl_test <- h2o.performance(my_deeplearning, newdata = test_df_h2o)
print(perf_dl_test)

# r2 for training data
r2_basic <- h2o.r2(my_gbm)
r2_basic

# retrieve the r2 value for the validation data:
r2_basic_valid <- h2o.r2(test_y, perf_gbm_test) # this is not right
r2_basic_valid

#baselearner_best_auc_test <- max(h2o.auc(perf_gbm_test), h2o.auc(perf_rf_test), h2o.auc(perf_dl_test))
#ensemble_auc_test <- h2o.auc(perf)
#print(sprintf("Best Base-learner Test AUC:  %s", baselearner_best_auc_test))
#print(sprintf("Ensemble Test AUC:  %s", ensemble_auc_test))





