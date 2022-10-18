#cleaning all the consoles
rm(list=ls())

#reading the dataset
#setwd("C:/Users/anura/OneDrive/Documents/Personal/Gsu/sem 1/Big data analytics/Project work/dataset")
setwd("C:/Users/Arunita/CIS8695_Cost Prediction")
rawData = read.csv('media prediction and its cost.csv')
print(dim(rawData))
#[1] 60428    40


#methods

encode_ordinal <- function(x, order = unique(x)) {
  x <- as.numeric(factor(x, levels = order, exclude = NULL))
  x
}

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
install.packages("reshape2")
library(reshape2)

# creating correlation matrix
corrMmat <- round(cor(rawData[numCols]),2)

# reduce the size of correlation matrix
meltedCorrMat <- melt(corrMmat)
head(meltedCorrMat)

# plotting the correlation heatmap
library(ggplot2)
ggplot(data = meltedCorrMat, aes(x=Var1, y=Var2,
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


#PCA

# pcs <- prcomp(na.omit(rawData[,-c(106)])) 
# summary(pcs)
# # pcaTrainData = pcs$x[,1:2]
# pcaTrainData = data.frame(rawData$cost,pcs$x[,1:10])

#splitting the dataset
set.seed(2)
colnames(rawData) <- make.names(colnames(rawData))
train.index <- sample(c(1:dim(rawData)[1]), dim(rawData)[1]*0.7)
train.df <- rawData[train.index, ]
valid.df <- rawData[-train.index, ]

library(forecast)
LinearRegression <- function(train.df,valid.df,key) {
  model <- lm(cost ~ .,train.df)
  options(scipen = 999)
  print(summary(model))
  pred <- predict(model,valid.df)
  print(accuracy(pred,valid.df$cost))
  # print(pred)
  # predicted[key] <<- c(pred)
}

LinearRegression(train.df,valid.df,'Linear Regression on Ordinal Encoded Data')


library(randomForest)
RandomForestRegression <- function(train.df,valid.df,key) {
  model <- randomForest(as.factor(cost) ~ ., data =  train.df, ntree = 50, 
                        mtry = 4, nodesize = 5, importance = TRUE) 
  # varImpPlot(rf, type = 1)
  # print(pred)
  
  test.features = subset(valid.df, select=-c(cost))
  test.target = subset(valid.df, select=cost)[,1]
  predictions = as.numeric(predict(model, newdata = test.features))
  #Mean Absolute Error
  print(mean(abs(test.target - predictions)))
  # RMSE
  print(sqrt(mean(abs(test.target - predictions))))
  # R2
  print(cor(test.target, predictions) ^ 2)
}

RandomForestRegression(train.df,valid.df,'Random Forest Regression on Ordinal Encoded Data')

library(rpart)
library(caret)
DecisionTreeRegressor <- function(train.df,valid.df,key) {
  decisionTree <- train(
    cost ~ .,
    data = train.df,
    method = 'rpart2',
    # preProcess = c("center", "scale")
  )
  print(decisionTree)
  
  test.features = subset(valid.df, select=-c(cost))
  test.target = subset(valid.df, select=cost)[,1]
  
  predictions = predict(decisionTree, newdata = test.features)
  #Mean Absolute Error
  print(mean(abs(test.target - predictions)))
  # RMSE
  print(sqrt(mean(abs(test.target - predictions))))
  # R2
  print(cor(test.target, predictions) ^ 2)
}

DecisionTreeRegressor(train.df,valid.df,'Decision Tree Regressor onOrdinal Encoded Data')

library(class)
KNNReg <- function(train.df,valid.df,key) {
  
  #Loop from k=2-6
  grid1 <- expand.grid(.k = seq(2, 6, by = 1))
  
  #cross validation by 10 fold(default)
  control <- trainControl(method = "cv")
  # train the model
  knn <- train(cost ~ ., data = train.df,
               method = "knn",
               trControl = control,
               tuneGrid = grid1)
  
  test.features = subset(valid.df, select=-c(cost))
  test.target = subset(valid.df, select=cost)[,1]
  
  # RMSE,R-squared,MAE    
  print(knn)
  
  #Prediction
  predictions <- predict(knn, newdata = valid.df)
  #Mean Absolute Error
  print(sapply(abs(test.target - predictions),mean))
  # RMSE
  #print(sqrt(sapply(((test.target - predictions)^2),mean)))
  print(sqrt(mean((test.target - predictions)^2)))
  # R2
  print(cor(test.target, predictions) ^ 2)
  
  plot(test.target, predictions, col = c("red", "blue"), 
       main = 'Real vs Predicted')
  abline(0, 1, lwd = 2)
  
}

KNNReg(train.df,valid.df,'KNN on Ordinal Encoded Data')
# ---------------------------------------------------------------------
model <- randomForest(as.factor(cost) ~ ., data =  train.df, ntree = 1, 
                      mtry = 4, nodesize = 5, importance = TRUE) 
# varImpPlot(rf, type = 1)
# print(pred)

test.features = subset(valid.df, select=-c(cost))
test.target = subset(valid.df, select=cost)[,1]
predictions = as.numeric(predict(model, newdata = test.features))
#Mean Absolute Error
print(mean(abs(test.target - predictions)))
# RMSE
print(sqrt(mean(abs(test.target - predictions))))
# R2
print(cor(test.target, predictions) ^ 2)
#

##-----------------------------------------------------------------------
## Ridge regression
library(glmnet)
library(forecast)
RidgeRegressionUnscaled <- function(train.df,valid.df,key) {
  train_raw.df.mod <- subset (train.df, select = -19)
  x <- as.matrix(train_raw.df.mod)
  y <- train.df$cost
  RidgeMod <- glmnet(x, y, alpha=0, nlambda=100, lambda.min.ratio=0.0001)
  CvRidgeMod <- cv.glmnet(x, y, alpha=0, nlambda=100, lambda.min.ratio=0.0001)
  valid_raw.df.mod <- subset (valid.df, select = -19)
  test.features <- as.matrix(valid_raw.df.mod)
  test.target <- valid.df[, "cost"]
  best.lambda <- CvRidgeMod$lambda.min
  predictions <- predict(CvRidgeMod, s = best.lambda, newx = test.features)
  print(accuracy(as.vector(predictions), test.target))
  #Mean Absolute Error
  print(mean(abs(test.target - predictions)))
  # RMSE
  print(sqrt(mean(abs(test.target - predictions))))
  # R2
  print(cor(test.target, predictions) ^ 2)
}

RidgeRegressionUnscaled(train.df, valid.df, 'Ridge Regression')

##----------------------------------------------------------------------------
## Lasso regression

LassoRegressionUnscaled <- function(train.df,valid.df,key) {
  train_raw.df.mod <- subset (train.df, select = -19)
  x <- as.matrix(train_raw.df.mod)
  y <- train.df$cost
  LassoMod <- glmnet(x, y, alpha=1, nlambda=100, lambda.min.ratio=0.0001)
  CvLassoMod <- cv.glmnet(x, y, alpha=1, nlambda=100, lambda.min.ratio=0.0001)
  best.lambda <- CvLassoMod$lambda.min
  coef(CvLassoMod, s = "lambda.min")
  valid_raw.df.mod <- subset (valid.df, select = -19)
  test.features <- as.matrix(valid_raw.df.mod)
  test.target <- valid.df[, "cost"]
  predictions <- predict(CvLassoMod, s = best.lambda, newx = test.features)
  print(accuracy(as.vector(predictions), test.target))
  #eval_results(valid.df[,"cost"], Lasso.pred, valid.df)
  
  #Mean Absolute Error
  print(mean(abs(test.target - predictions)))
  # RMSE
  print(sqrt(mean(abs(test.target - predictions))))
  # R2
  print(cor(test.target, predictions) ^ 2)
}

LassoRegressionUnscaled(train.df, valid.df, 'Lasso Regression')

#Reference
#1. https://www.geeksforgeeks.org/how-to-create-correlation-heatmap-in-r/
# https://www.r-bloggers.com/2020/02/a-guide-to-encoding-categorical-features-using-r/
#https://koalatea.io/r-decision-tree-regression/