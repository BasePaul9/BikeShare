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

cv_recipe <- recipe(count ~ ., data = train) %>%
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
reg_model <- linear_reg(penalty=tune(),
                         mixture=tune()) %>% #Set model and tuning
  set_engine("glmnet") # Function to fit in R

## Set Workflow
reg_wf <- workflow() %>%
  add_recipe(cv_recipe) %>%
  add_model(reg_model)

## Grid of values to tune over
tuning_grid <- grid_regular(penalty(),
                            mixture(),
                            levels = ) ## L^2 total tuning possibilities

## Split data for CV
folds <- vfold_cv(train, v = , repeats = )

CV_results <- reg_wf %>%
  tune_grid(resamples=folds,
            grid=tuning_grid,
            metrics=metric_set(rmse, mae, rsq)) #Or leave metrics NULL

## Plot Results (example)
collect_metrics(CV_results) %>% # Gathers metrics into DF
  filter(.metric=="rmse") %>%
  ggplot(data=., aes(x=penalty, y=mean, color=factor(mixture))) +
  geom_line()

## Find Best Tuning Parameters
bestTune <- CV_results %>%
  select_best()

## Finalize the Workflow & fit it
final_wf <-
  reg_wf %>%
  finalize_workflow(bestTune) %>%
  fit(data=train)

## Predict
preds <- final_wf %>%
  predict(new_data = test)

# Format the Predictions for Submission to Kaggle
kag_sub <- preds %>%
  bind_cols(., test) %>% #Bind predictions with test data
  select(datetime, .pred) %>% #Just keep datetime and prediction variables
  rename(count=.pred) %>% #rename pred to count (for submission to Kaggle)
  mutate(count = exp(count)) %>%  
  mutate(count=pmax(0, count)) %>% #pointwise max of (0, prediction)
  mutate(datetime=as.character(format(datetime))) #needed for right format to Kaggle

## Write out the file
vroom_write(x=kag_sub, file="BikeShare/CVRegModel.csv", delim=",")