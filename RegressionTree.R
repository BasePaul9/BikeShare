library(tidyverse)
library(tidymodels)
library(poissonreg)
library(glmnet)
library(vroom)
library(rpart)

sample <- vroom("BikeShare/sampleSubmission.csv")
train <- vroom("BikeShare/train.csv") %>%
  select(-casual, -registered) %>%
  mutate(count = log(count))
test <- vroom("BikeShare/test.csv")

tree_mod <- decision_tree(tree_depth = tune(),
                          cost_complexity = tune(),
                          min_n = tune()) %>%
  set_engine("rpart") %>%
  set_mode("regression")

tree_recipe <- recipe(count ~ ., data = train) %>%
  step_mutate(weather = ifelse(weather == 4,3,weather)) %>%
  step_mutate(weather = factor(weather)) %>%
  step_mutate(holiday = factor(holiday)) %>%
  step_date(datetime, features = c("dow", "month")) %>%
  step_mutate(datetime_dow = factor(datetime_dow)) %>%
  step_mutate(datetime_month = factor(datetime_month)) %>%
  step_time(datetime, features = c("hour")) %>%
  step_mutate(datetime_hour = factor(datetime_hour))

## Set Workflow
tree_wf <- workflow() %>%
  add_recipe(tree_recipe) %>%
  add_model(tree_mod)

## Grid of values to tune over
tuning_grid <- grid_regular(cost_complexity(),
                            tree_depth(),
                            min_n(),
                            levels = 9) ## L^2 total tuning possibilities

## Split data for CV
folds <- vfold_cv(train, v = 5, repeats = 1)

CV_results <- tree_wf %>%
  tune_grid(resamples=folds,
            grid=tuning_grid,
            metrics=metric_set(rmse, mae, rsq))

collect_metrics(CV_results) %>% # Gathers metrics into DF
  filter(.metric=="rmse") %>%
  ggplot(data=., aes(x = cost_complexity, y=mean, color=factor(tree_depth))) +
  geom_line()

bestTune <- CV_results %>%
  select_best()

## Finalize the Workflow & fit it
final_wf <-
  tree_wf %>%
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
vroom_write(x=kag_sub, file="BikeShare/RegTreeModel.csv", delim=",")
