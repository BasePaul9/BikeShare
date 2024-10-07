library(tidyverse)
library(tidymodels)
library(poissonreg)
library(glmnet)
library(vroom)
library(rpart)
library(ranger)

sample <- vroom("BikeShare/sampleSubmission.csv")
train <- vroom("BikeShare/train.csv") %>%
  select(-casual, -registered) %>%
  mutate(count = log(count))
test <- vroom("BikeShare/test.csv")

rf_mod <- rand_forest(mtry = tune(),
                      min_n = tune(),
                      trees = 500) %>%
  set_engine("ranger") %>%
  set_mode("regression")

tree_recipe <- recipe(count ~ ., data = train) %>%
  step_mutate(weather = ifelse(weather == 4,3,weather)) %>%
  step_mutate(weather = factor(weather)) %>%
  step_mutate(holiday = factor(holiday)) %>%
  step_date(datetime, features = c("dow", "month")) %>%
  step_mutate(datetime_dow = factor(datetime_dow)) %>%
  step_mutate(datetime_month = factor(datetime_month)) %>%
  step_time(datetime, features = c("hour")) %>%
  step_mutate(datetime_hour = factor(datetime_hour)) %>%
  step_interact(terms = ~ datetime_dow:datetime_hour)

recipe <- prep(tree_recipe)

## Set Workflow
forest_wf <- workflow() %>%
  add_recipe(tree_recipe) %>%
  add_model(rf_mod)

## Grid of values to tune over
tuning_grid <- grid_regular(mtry(range = (c(1,(ncol(bake(recipe,train))-1)))),
                            min_n(),
                            levels = 5) ## L^2 total tuning possibilities

## Split data for CV
folds <- vfold_cv(train, v = 5, repeats = 1)

CV_results <- forest_wf %>%
  tune_grid(resamples=folds,
            grid=tuning_grid,
            metrics=metric_set(rmse, mae, rsq))

collect_metrics(CV_results) %>% # Gathers metrics into DF
  filter(.metric=="rmse") %>%
  ggplot(data=., aes(x = mtry, y=mean, color=min_n)) +
  geom_line()

bestTune <- CV_results %>%
  select_best()

## Finalize the Workflow & fit it
final_wf <-
  forest_wf %>%
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
vroom_write(x=kag_sub, file="BikeShare/RandomForestModel.csv", delim=",")
