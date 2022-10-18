#cleaning all the consoles
rm(list =ls())
#reading the dataset
setwd("C:/Users/anura/OneDrive/Documents/Personal/Gsu/sem 1/Big data analytics/Project work/dataset")
rawData = read.csv('media prediction and its cost.csv')
print(dim(rawData))
#[1] 60428    40

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
# install.packages("reshape2")
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

rawData = within(rawData, rm('salad_bar', 'gross_weight', 'avg_cars_at.home.approx..1', 'meat_sqft','store_sales.in.millions.', 'store_cost.in.millions.', 'store_sqft'))

# We also drop columns which have no correlation with with cost
rawData = within(rawData, rm('recyclable_package','low_fat'))

# We also drop the columns which have next to no difference in between them and will just create noise in the data

rawData = within(rawData, rm('food_category', 'food_department', 'food_family', 'gender', 'marital_status',
                             'brand_name', 'avg..yearly_income', 'education', 'member_card', 'houseowner', 
                             'sales_country', 'occupation'))

library(mltools)
library(data.table)

rawData$store_type <- as.factor(rawData$store_type)
rawData$store_city <- as.factor(rawData$store_city)
rawData$store_state <- as.factor(rawData$store_state)
rawData$media_type <- as.factor(rawData$media_type)
rawData$promotion_name <- as.factor(rawData$promotion_name)
oneHotData <- one_hot(as.data.table(rawData))

# rawData = within(rawData, rm('store_type', 'store_city','store_state','media_type','promotion_name'))
# rawData = within(rawData, rm('unit_sales.in.millions.'))
# 
# total <-cbind(rawData,newData)

colnames(oneHotData)
colnames(oneHotData) <- make.names(colnames(oneHotData))
set.seed(2)
train.index <- sample(c(1:dim(oneHotData)[1]), dim(oneHotData)[1]*0.6)
train.df <- oneHotData[train.index, ]
valid.df <- oneHotData[-train.index, ]

# an output dictionary to cotain results of all models
# predicted = c(
#   'Linear Regression'='',
#   'Decision Tree Regressor'='',
#   'Random Forest Regressor'=''
# )


predicted <- c()
library(forecast)
LinearRegression <- function(train.df,valid.df,key) {
  model <- lm(cost ~ .,train.df)
  options(scipen = 999)
  print(summary(model))
  test.features = subset(valid.df, select=-c(cost))
  test.target = subset(valid.df, select=cost)[,1]
  predictions <- predict(model,test.features)
  #Mean Absolute Error
  print(sapply(abs(test.target - predictions),mean))
  # RMSE
  print(sqrt(sapply(((test.target - predictions)^2),mean)))
  # R2
  print(cor(test.target, predictions) ^ 2)
  plot(test.target$cost, predictions, col = "red", 
       main = 'Real vs Predicted')
  abline(0, 1, lwd = 2)
}

LinearRegression(train.df,valid.df,'Linear Regression on One hot Encoded Data')


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
  print(sapply(abs(test.target - predictions),mean))
  # RMSE
  print(sqrt(sapply(((test.target - predictions)^2),mean)))
  # R2
  print(cor(test.target, predictions) ^ 2)
  plot(test.target$cost, predictions, col = "red", 
       main = 'Real vs Predicted')
  abline(0, 1, lwd = 2)
}

RandomForestRegression(train.df,valid.df,'Random Forest Regression on One hot Encoded Data')

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
  print(sapply(abs(test.target - predictions),mean))
  # RMSE
  print(sqrt(sapply(((test.target - predictions)^2),mean)))
  # R2
  print(cor(test.target, predictions) ^ 2)
  
  plot(test.target$cost, predictions, col = "red", 
       main = 'Real vs Predicted')
  abline(0, 1, lwd = 2)
}

DecisionTreeRegressor(train.df,valid.df,'Decision Tree Regressor on One hot Encoded Data')

library(class)
KNNReg <- function(train.df,valid.df,key) {
  
  #Loop from k=2-10
  grid1 <- expand.grid(.k = seq(2, 4, by = 1))
  
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
  print(sqrt(mean((test.target - predictions.rf)^2)))
  # R2
  print(cor(test.target, predictions) ^ 2)
  
}

KNNReg(train.df,valid.df,'KNN on One hot Encoded Data')

