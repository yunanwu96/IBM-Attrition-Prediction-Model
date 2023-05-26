# EDA

# Clean console environment
rm(list = ls())

# Import library
library(janitor)
library(csv)
library(dplyr)
library(plyr)
library(ggplot2)
library(corrplot)
library(RColorBrewer)
library(car)
library(tidyverse)
library(leaps)
library(gvlma)
library(caTools)
library(caret)
library(stats)
library(ISLR)
library(gridExtra)
library(pROC)
library(InformationValue)
library(nortest)
library(ggstatsplot)
library(statsExpressions)
library(insight)
library(parameters)
library(bayestestR)
library(datawizard)
library(zeallot)
library(correlation)
library(paletteer)
library(rematch2)
library(patchwork)
library(performance)
library(prismatic)
library(imputeR)
library(gplots)
library(RColorBrewer)
library(Hmisc)
library(latticeExtra)
library(png)
library(jpeg)
library(interp)
library(deldir)
library(htmlTable)
library(psych)
library(mnormt)
library(kableExtra)
library(svglite)
library(lares)
library(h2o)
library(randomForest)
library(ranger)
library(e1071)
library(glmnet)
library(Metrics)

# Import dataset
df <- read.csv("C:/Users/Yunan/Desktop/NEU/Aly6040/project/attrition_clean.csv")

# Check the structure of dataset
summary(df)
dim(df)
str(df)

######################################################################
# Research Question 1: Predict Monthly Income
# Split the data into a train and test set
set.seed(123)

trainIndex <- createDataPartition(df$MonthlyIncome, p = 0.7, list = FALSE, times = 1)
train <- df[trainIndex,]
test <- df[-trainIndex,]

train$Attrition <- replace(train$Attrition, train$Attrition  == 0,"No")
train$Attrition <- replace(train$Attrition, train$Attrition  == 1,"Yes")
test$Attrition <- replace(test$Attrition, test$Attrition  == 0,"No")
test$Attrition <- replace(test$Attrition, test$Attrition  == 1,"Yes")
test$Attrition <- as.factor(test$Attrition)
train$Attrition <- as.factor(train$Attrition)

train_x <- model.matrix(MonthlyIncome ~., train)[, -33]
test_x <- model.matrix(MonthlyIncome ~., test)[, -33]

train_y <- train$MonthlyIncome
test_y <- test$MonthlyIncome

# Build the random forest model
start.time <- Sys.time()

rf <- randomForest(MonthlyIncome ~.,data = train, importance=TRUE) 

end.time <- Sys.time()
time.taken <- end.time - start.time
time.taken

print(rf)

# Check the prediction accuracy
MonthlyIncome_predict <- predict(rf, train)

plot(train$MonthlyIncome, MonthlyIncome_predict, main = 'Train set',ylim = c(0,25000),
     xlab = 'Price', ylab = 'Predict')
abline(a=0,b=1,col='blue')

# Evaluate the model
MonthlyIncome_predict <- predict(rf, test)

plot(test$MonthlyIncome, MonthlyIncome_predict, main = 'Test Set',ylim = c(0,25000),
     xlab = 'Price', ylab = 'Predict')
abline(a=0,b=1,col='blue')

# Train set predictions
preds.train<- predict(rf, new = train)
train.rmse <- rmse(train$MonthlyIncome, preds.train)

train.rmse

# R squared
sst <- sum(train_y^2)
sse <- sum((preds.train - train_y)^2)
rsq <- 1 - sse / sst
rsq

# Test set predictions
preds.test<- predict(rf, new = test)
test.rmse <- rmse(test$MonthlyIncome, preds.test)

test.rmse

# R squared
sst <- sum(test_y^2)
sse <- sum((preds.test - test_y)^2)
rsq <- 1 - sse / sst
rsq

# Importance evaluation
# summary(rf)
importance_rf <- as.data.frame(rf$importance) 
importance_rf <- importance_rf%>% arrange(desc(importance_rf$IncNodePurity))
head(importance_rf)

importance_rf <- importance_rf[order(importance_rf$IncNodePurity, decreasing = TRUE), ]


importance_rf.select <- importance_rf[1:15, ]
importance_rf.select


# Plot variable importance order
varImpPlot(rf, n.var = min(30, nrow(rf$importance)), main = 'Variable importance order')

############################################################################
# Use k-fold cross validation with k = 5 folds to evaluate performance.
# Plot the fitted line plot
set.seed(123)
train.cv <- replicate(1, rfcv(train[-ncol(train)], train$MonthlyIncome, cv.fold = 5, step = 1.5), simplify = FALSE)
train.cv

train.cv <- data.frame(sapply(train.cv, '[[', 'error.cv'))
train.cv$otus <- rownames(train.cv)
train.cv <- reshape2::melt(train.cv, id = 'otus')
train.cv$otus <- as.numeric(as.character(train.cv$otus))

train.cv.mean <- aggregate(train.cv$value, by = list(train.cv$otus), FUN = mean)
head(train.cv.mean, 10)

ggplot(train.cv.mean, aes(Group.1, x)) +
  geom_line() +
  theme(panel.grid = element_blank(), panel.background = element_rect(color = 'black', 
                                                                      fill = 'transparent')) + labs(title = '',x = 'Number of variables', 
                                                                                                    y = 'Cross-validation error')
############################################################################
# Select features and/or tune model parameters to achieve the optimal performance. 
# Show (or plot) model performance under different feature selection and/or parameter tuning settings.
start.time <- Sys.time()

rf1 <- randomForest(MonthlyIncome ~ JobLevel + TotalWorkingYears +
                    + JobRoleManager + JobRoleResearch.Director + YearsAtCompany + Age,
                    data = train, importance=TRUE) 

end.time <- Sys.time()
time.taken <- end.time - start.time
time.taken

print(rf1)

# Check the prediction accuracy
MonthlyIncome_predict <- predict(rf1, train)

plot(train$MonthlyIncome, MonthlyIncome_predict, main = 'Train set',ylim = c(0,25000),
     xlab = 'MonthlyIncome', ylab = 'Predict')
abline(a=0,b=1,col='blue')

# Evaluate the model
MonthlyIncome_predict <- predict(rf1, test)

plot(test$MonthlyIncome, MonthlyIncome_predict, main = 'Test Set',ylim = c(0,25000),
     xlab = 'MonthlyIncome', ylab = 'Predict')
abline(a=0,b=1,col='blue')

# Train set predictions
preds.train<- predict(rf1, new = train)
train.rmse <- rmse(train$MonthlyIncome, preds.train)

train.rmse

# R squared
sst <- sum(train_y^2)
sse <- sum((preds.train - train_y)^2)
rsq <- 1 - sse / sst
rsq

# Test set predictions
preds.test<- predict(rf1, new = test)
test.rmse <- rmse(test$MonthlyIncome, preds.test)

test.rmse

# R squared
sst <- sum(test_y^2)
sse <- sum((preds.test - test_y)^2)
rsq <- 1 - sse / sst
rsq

######################################################################
# Research Question 2: Predict Attrition
#######################################################################
# Method 1: SVM
set.seed(123)

trainIndex <- createDataPartition(df$Attrition, p = 0.7, list = FALSE, times = 1)
train <- df[trainIndex,]
test <- df[-trainIndex,]

train$Attrition <- replace(train$Attrition, train$Attrition  == 0,"No")
train$Attrition <- replace(train$Attrition, train$Attrition  == 1,"Yes")
test$Attrition <- replace(test$Attrition, test$Attrition  == 0,"No")
test$Attrition <- replace(test$Attrition, test$Attrition  == 1,"Yes")
test$Attrition <- as.numeric(test$Attrition)
train$Attrition <- as.numeric(train$Attrition)

train_x <- model.matrix(Attrition ~., train)[, -1]
test_x <- model.matrix(Attrition ~., test)[, -1]

train_y <- train$Attrition
test_y <- test$Attrition


tuned <- tune(svm,factor(Attrition)~.,data = train)
svm.model <- svm(train$Attrition~., data=train
                 ,type="C-classification", gamma=tuned$best.model$gamma
                 ,cost=tuned$best.model$cost
                 ,kernel="radial")
summary(svm.model)
svm.prd <- predict(svm.model,newdata=test)
confusionMatrix(svm.prd,factor(test$Attrition))

svm.plot <-plot.roc(as.numeric(test$Attrition), 
                     as.numeric(svm.prd),lwd=2, type="b", print.auc=TRUE,col ="blue")


###############################################################################
# Method 2: Logistic Regression
# Fit a logistic regression model
model1 <- glm(Attrition ~., data = train, family = binomial(link = 'logit'))
summary(model1)

model2 <- glm(Attrition ~ BusinessTravel+MaritalStatus+OverTime+Age+DistanceFromHome+
                EnvironmentSatisfaction +JobInvolvement+JobSatisfaction+NumCompaniesWorked+
                WorkLifeBalance+RelationshipSatisfaction +YearsInCurrentRole,
              data = train, family = binomial(link = 'logit'))
summary(model2)

# Train set predictions
# Make predictions on the test data using lambda.min
probabilities.train <- predict(model2, newdata = train, type = 'response')
predicted.classes.min <- as.factor(ifelse(probabilities.train >= 0.5, 'Yes', 'No'))

# Model accuracy
confusionMatrix(predicted.classes.min, train$Attrition, positive = 'Yes')

# Test set predictions
probabilities.test <- predict(model2, newdata = test, type = 'response')
predicted.classes.min2 <- as.factor(ifelse(probabilities.test >= 0.5, 'Yes', 'No'))

# Model accuracy
confusionMatrix(data = predicted.classes.min2, reference = test$Attrition, positive = 'Yes')


# Plot the ROC curve
roc1 <- roc(test$Attrition, probabilities.test)
plot(roc1, col = 'blue', ylab = 'Sensitivity-TP Rate', xlab = 'Specificity - FP Rate' )

auc(roc1)
