# Decision Tree Analysis of Customer Intention Data
# by: Joe Domaleski
# for more information, please see the blog post on https://blog.marketingdatascience.ai

# Install necessary packages (if not already installed)
# install.packages("rpart")
# install.packages("rpart.plot")
# install.packages("caret")
# install.packages("dplyr")

# Load libraries
library(rpart)
library(rpart.plot)
library(caret)
library(dplyr)

# 1. Load the dataset (Ensure the CSV file is in your working directory)
data <- read.csv("online_shoppers_intention.csv")

# 2. Convert 'Revenue' (target variable) to a factor for classification
data$Revenue <- as.factor(data$Revenue)

# Convert categorical features to factors
data$Month <- as.factor(data$Month)
data$VisitorType <- as.factor(data$VisitorType)
data$Weekend <- as.factor(data$Weekend)

# 3. Split the data into training (70%) and testing (30%) sets
set.seed(123)  # Ensure reproducibility
trainIndex <- createDataPartition(data$Revenue, p = 0.7, list = FALSE)
trainData <- data[trainIndex, ]
testData <- data[-trainIndex, ]

# 4. Train a decision tree model using relevant features
model <- rpart(Revenue ~ Administrative + Administrative_Duration + 
                 Informational + Informational_Duration + ProductRelated + 
                 ProductRelated_Duration + BounceRates + ExitRates + PageValues + 
                 SpecialDay + Month + OperatingSystems + Browser + Region + 
                 TrafficType + VisitorType + Weekend, 
               data = trainData, method = "class")

# 5. Visualize the Decision Tree
rpart.plot(model, main = "Decision Tree for Online Shopping Purchase Prediction", type = 3, extra = 101)

# 6. Make predictions on the test set
predictions <- predict(model, testData, type = "class")

# 7. Evaluate model accuracy using a confusion matrix
confMatrix <- confusionMatrix(predictions, testData$Revenue)
print(confMatrix)

# 8. Display feature importance
print(model$variable.importance)
