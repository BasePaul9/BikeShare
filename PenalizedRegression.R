library(tidyverse)
library(tidymodels)
library(poissonreg)
library(glmnet)
library(vroom)

sample <- vroom("BikeShare/sampleSubmission.csv")
train <- vroom("BikeShare/train.csv") %>%
  select(-casual, -registered) %>%
  mutate(count = log(count))
test <- vroom("BikeShare/test.csv")

penalized_recipe <- recipe(count ~ ., data = train) %>%
  step_mutate(weather = ifelse(weather == 4,3,weather)) %>%
  step_mutate(weather = factor(weather)) %>%
  step_mutate(holiday = factor(holiday)) %>%
  step_date(datetime, features = c("dow", "month")) %>%
  step_mutate(datetime_dow = factor(datetime_dow)) %>%
  step_mutate(datetime_month = factor(datetime_month)) %>%
  step_time(datetime, features = c("hour")) %>%
  step_mutate(datetime_hour = factor(datetime_hour)) %>%
  step_interact(terms = ~ datetime_dow:datetime_hour) %>%
  step_rm("datetime") %>%
  step_rm("season") %>%
  step_rm("workingday") %>%
  step_rm("atemp") %>%
  step_dummy(all_nominal_predictors()) %>%
  step_normalize(all_numeric_predictors())

## Penalized regression model
preg_model <- linear_reg(penalty = 0.5, mixture = 0.1) %>% #Set model and tuning
  set_engine("glmnet") # Function to fit in R

# Pre-Penalization Score = 0.52032
# lambda = 1, v = 0, score = 0.65742
# lambda = 1, v = 1, score = 1.41486
# lambda = 1, v = 0.5, score = 1.41486
# lambda = 1, v = 0.1, score = 0.90716
# lambda = 10, v = 0, score = 1.13620
# lambda = 2, v = 0, score = 0.76793
# lambda = 1.5, v = 0, score = 0.71529
# lambda = 0.5, v = 0, score = 0.59618
# lambda = 0.1, v = 0, score = 0.53581
# lambda = 0.01, v = 0, score = 0.52694
# lambda = 0.001, v = 0, score = 0.52694
# lambda = 0.01, v = 1, score = 0.55066
# lambda = 0.01, v = 0.5, score = 0.52907
# lambda = 0, v = 0, score = 0.52694git
# lambda = 10, v = 1, score = 1.41486

preg_wf <- workflow() %>%
  add_recipe(penalized_recipe) %>%
  add_model(preg_model) %>%
  fit(data=train)
  
pen_preds <- predict(preg_wf, new_data = test)

# Format the Predictions for Submission to Kaggle
kag_sub <- pen_preds %>%
  bind_cols(., test) %>% #Bind predictions with test data
  select(datetime, .pred) %>% #Just keep datetime and prediction variables
  rename(count=.pred) %>% #rename pred to count (for submission to Kaggle)
  mutate(count = exp(count)) %>%  
  mutate(count=pmax(0, count)) %>% #pointwise max of (0, prediction)
  mutate(datetime=as.character(format(datetime))) #needed for right format to Kaggle

## Write out the file
vroom_write(x=kag_sub, file="BikeShare/PenRegModel.csv", delim=",")

