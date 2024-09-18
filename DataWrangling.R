library(tidyverse)
library(vroom)
library(tidymodels)
library(corrplot)


sample <- vroom("BikeShare/sampleSubmission.csv")
train <- vroom("BikeShare/train.csv") %>%
  select(-casual, -registered) %>%
  mutate(count = log(count))
test <- vroom("BikeShare/test.csv")

bike_recipe <- recipe(count ~ ., data = train) %>%
  step_mutate(weather = ifelse(weather == 4,3,weather)) %>%
  step_mutate(weather = factor(weather)) %>%
  step_mutate(season = factor(season)) %>%
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
  step_rm("atemp")

#bike_recipe <- prep(bike_recipe)
#bake(bike_recipe, new_data = test)

lm_bike <- linear_reg() %>%
  set_engine("lm") %>%
  set_mode("regression")

bike_workflow <- workflow() %>%
  add_recipe(bike_recipe) %>%
  add_model(lm_bike) %>%
  fit(data = train)

lin_preds <- predict(bike_workflow, new_data = test)
  
# Format the Predictions for Submission to Kaggle
kag_sub <- lin_preds %>%
  bind_cols(., test) %>% #Bind predictions with test data
  select(datetime, .pred) %>% #Just keep datetime and prediction variables
  rename(count=.pred) %>% #rename pred to count (for submission to Kaggle)
  mutate(count=pmax(0, count)) %>% #pointwise max of (0, prediction)
  mutate(count = exp(count)) %>%
  mutate(datetime=as.character(format(datetime))) #needed for right format to Kaggle

## Write out the file
vroom_write(x=kag_sub, file="BikeShare/NewLinPreds.csv", delim=",")
