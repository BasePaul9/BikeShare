library(vroom)
library(patchwork)
library(skimr)
library(DataExplorer)
library(GGally)
library(tidymodels)
library(poissonreg)
library(glmnet)
library(lubridate)

sample <- vroom("BikeShare/sampleSubmission.csv")
train <- vroom("BikeShare/train.csv")
test <- vroom("BikeShare/test.csv")

train$season <- as.factor(train$season)
test$season <- as.factor(test$season)
train$weather <- as.factor(train$weather)
test$weather <- as.factor(test$weather)
train$workingday <- as.factor(train$workingday)
test$workingday <- as.factor(test$workingday)
train$holiday <- as.factor(train$holiday)
test$holiday <- as.factor(test$holiday)

train$month <- month(train$datetime)
test$month <- month(test$datetime)
train$hour <- hour(train$datetime)
test$hour <- hour(test$datetime)

train <- train |> relocate(count, .after = hour)
train <- train |> relocate(casual, .after = count)
train <- train |> relocate(registered, .after = casual)

for (i in 1:nrow(train)) {
if(train$weather[i] == 4){
  train$weather[i] <- 3
}
}
for (i in 1:nrow(test)) {
  if(test$weather[i] == 4){
    test$weather[i] <- 3
  }
}

traindf <- as.data.frame(train)
train_x <- as.matrix(traindf[, 1:11])
train_y <- traindf[, 12]

train_lasso <- cv.glmnet(x = train_x, 
                           y = train_y, 
                           type.measure = "mse", 
                           alpha = 1) 
coef(train_lasso, s = "lambda.1se")

pois_model <- poisson_reg() %>% #Type of model
  set_engine("glm") %>% # GLM = generalized linear model
  set_mode("regression") %>%
  fit(formula = count ~ temp + holiday + humidity + month + hour, data = train)

pois_preds <- predict(pois_model, new_data = test)

pois_preds

# Format the Predictions for Submission to Kaggle
pois_kaggle_submission <- bike_predictions %>%
  bind_cols(., test) %>% #Bind predictions with test data
  select(datetime, .pred) %>% #Just keep datetime and prediction variables
  rename(count=.pred) %>% #rename pred to count (for submission to Kaggle)
  mutate(count=pmax(0, count)) %>% #pointwise max of (0, prediction)
  mutate(datetime=as.character(format(datetime))) #needed for right format to Kaggle

## Write out the file
vroom_write(x=pois_kaggle_submission, file="BikeShare/PoissonPreds.csv", delim=",")
