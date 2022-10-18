#cleaning all the consoles
rm(list=ls())

#reading the dataset
setwd("C:/Users/Nikhil/Desktop/BIG DATA ANALYTICS/Project/")
rawData = read.csv('media prediction and its cost.csv')

encode_ordinal <- function(x, order = unique(x)) {
  x <- as.numeric(factor(x, levels = order, exclude = NULL))
  x
}

lapply(rawData,function(x) { length(which(is.na(x)))})
#There are no null values in the dataset

#finding numerical and categorical data
str(rawData)
numCols = unlist(lapply(rawData, is.numeric))         # Identify numeric columns
catCols = unlist(lapply(rawData, is.character))

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

library(tidyverse)
library(neuralnet)

# install.packages("GGally")
# install.packages("tidyverse")
library(GGally)
library(dplyr)
#Normalize the data
scale01 <- function(x){
  (x - min(x)) / (max(x) - min(x))
}

rawData <- rawData %>%
  mutate_all(scale01)


set.seed(12345)
rawData_Train <- sample_frac(tbl = rawData, replace = FALSE, size = 0.80)
rawData_Test <- anti_join(rawData, rawData_Train)

set.seed(12321)
rawData_NN1 <- neuralnet(cost ~ ., data = rawData_Train,
                         hidden = 1 ,
                         linear.output = TRUE)


plot(rawData_NN1, rep = 'best')

prediction <- compute(rawData_NN1, rawData_Test)

prediction

