library(tidyverse)
library(tidymodels)
library(poissonreg)
library(glmnet)
library(vroom)
library(rpart)
library(ranger)
library(dbarts)

#sample <- vroom("BikeShare/sampleSubmission.csv")
train <- vroom("BikeShare/train.csv") %>%
  select(-casual, -registered) %>%
  mutate(count = log(count))
test <- vroom("BikeShare/test.csv")

bart_recipe <- recipe(count ~ ., data = train) %>%
  step_mutate(weather = ifelse(weather == 4,3,weather)) %>%
  step_mutate(weather = factor(weather)) %>%
  step_mutate(holiday = factor(holiday)) %>%
  step_date(datetime, features = c("dow", "month")) %>%
  step_mutate(datetime_dow = factor(datetime_dow)) %>%
  step_mutate(datetime_month = factor(datetime_month)) %>%
  #step_mutate(datetime_year = factor(datetime_year)) %>%
  step_time(datetime, features = c("hour")) %>%
  step_mutate(datetime_hour = factor(datetime_hour)) %>%
  #step_interact(terms = ~ datetime_dow:datetime_year) %>%
  step_interact(terms = ~ datetime_dow:datetime_hour)

bart_model <- parsnip::bart(
  mode = "regression",
  engine = "dbarts",
  trees = 1000,
  prior_terminal_node_coef = .95,
  prior_terminal_node_expo = 2,
  prior_outcome_range = 2
)

bart_wf <- workflow() %>%
  add_recipe(bart_recipe) %>%
  add_model(bart_model) %>%
  fit(data = train)


# ## Grid of values to tune over
# tuning_grid <- grid_regular(prior_terminal_node_coef(),
#                             prior_terminal_node_expo(),
#                             prior_outcome_range(),
#                             levels = 5) ## L^2 total tuning possibilities
# 
# ## Split data for CV
# folds <- vfold_cv(train, v = 5, repeats = 1)
# 
# CV_results <- bart_wf %>%
#   tune_grid(resamples=folds,
#             grid=tuning_grid)
# 
# ## Plot Results (example)
# collect_metrics(CV_results) %>% # Gathers metrics into DF
#   filter(.metric=="rmse") %>%
#   ggplot(data=., aes(x=penalty, y=mean, color=factor(mixture))) +
#   geom_line()
# 
# ## Find Best Tuning Parameters
# bestTune <- CV_results %>%
#   select_best()
# 
# final_wf <- bart_wf %>%
#   finalize_workflow(bestTune) %>%
#   fit(data=train)
#
# bart_preds <- predict(final_wf, new_data = test)

bart_preds <- predict(bart_wf, new_data = test)

# Format the Predictions for Submission to Kaggle
kag_sub <- bart_preds %>%
  bind_cols(., test) %>% #Bind predictions with test data
  select(datetime, .pred) %>% #Just keep datetime and prediction variables
  rename(count=.pred) %>% #rename pred to count (for submission to Kaggle)
  mutate(count = exp(count)) %>%
  mutate(count=pmax(0, count)) %>% #pointwise max of (0, prediction)
  mutate(datetime=as.character(format(datetime))) #needed for right format to Kaggle## Write out the file

vroom_write(x=kag_sub, file="BikeShare/BARTModel.csv", delim=",")




