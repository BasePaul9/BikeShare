library(tidyverse)
library(tidymodels)
library(poissonreg)
library(glmnet)
library(vroom)
library(rpart)
library(ranger)
library(stacks)

sample <- vroom("BikeShare/sampleSubmission.csv")
train <- vroom("BikeShare/train.csv") %>%
  select(-casual, -registered) %>%
  mutate(count = log(count))
test <- vroom("BikeShare/test.csv")

stacked_recipe <- recipe(count ~ ., data = train) %>%
  step_mutate(weather = ifelse(weather == 4,3,weather)) %>%
  step_date(datetime, features = c("dow", "month")) %>%
  step_mutate(datetime_dow = factor(datetime_dow)) %>%
  step_mutate(datetime_month = factor(datetime_month)) %>%
  step_time(datetime, features = c("hour")) %>%
  step_interact(terms = ~ datetime_dow:datetime_hour) %>%
  step_rm("datetime") %>%
  step_dummy(all_nominal_predictors(), keep_original_cols = FALSE)
recipe <- prep(stacked_recipe)

folds <- vfold_cv(train, v = 5, repeats = 1)

untunedModel <- control_stack_grid() #If tuning over a grid
tunedModel <- control_stack_resamples() #If not tuning a model

preg_model <- linear_reg(penalty=tune(),
                         mixture=tune()) %>% #Set model and tuning
              set_engine("glmnet") # Function to fit in R

## Set Workflow
preg_wf <- workflow() %>%
  add_recipe(stacked_recipe) %>%
  add_model(preg_model)

## Grid of values to tune over
tuning_grid <- grid_regular(penalty(),
                                 mixture(),
                                 levels = 5) ## L^2 total tuning possibilities

# Run the CV
preg_model <- preg_wf %>%
  tune_grid(resamples=folds,
          grid=tuning_grid,
          metrics=metric_set(rmse, mae, rsq),
          control = untunedModel) # including the control grid in the tuning ensures you can
                                  # call on it later in the stacked model

## Create other resampling objects with different ML algorithms to include in a stacked model, for ex9
# linear regression

lin_reg <- linear_reg() %>% 
  set_engine("lm")
lin_reg_wf <-
  workflow() %>%
  add_model(lin_reg) %>%
  add_recipe(stacked_recipe)
lin_reg_model <-
  fit_resamples(
    lin_reg_wf,
    resamples = folds,
    metrics = metric_set(rmse, mae, rsq),
    control = tunedModel)

#Random Forests

rf_mod <- rand_forest(mtry = 10,
                      min_n = 15,
                      trees = 1000) %>%
  set_engine("ranger") %>%
  set_mode("regression")

## Set Workflow
forest_wf <- workflow() %>%
  add_recipe(stacked_recipe) %>%
  add_model(rf_mod)

## Grid of values to tune over
tuning_grid <- grid_regular(mtry(range = (c(1,(ncol(bake(recipe,train))-1)))),
                            min_n(),
                            levels = 5) ## L^2 total tuning possibilities

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

rand_forest_model <-
  fit_resamples(
    final_wf,
    resamples = folds,
    metrics = metric_set(rmse, mae, rsq),
    control = tunedModel)

#Bart addition

## Stacked Model
my_stack <- stacks() %>%
  add_candidates(preg_model) %>%
  add_candidates(lin_reg_model) %>%
  add_candidates(rand_forest_model) %>%
  add_candidates(bart_model)

# Fit the stacked model
stack_mod <- my_stack %>%
blend_predictions() %>% # LASSO penalized regression meta-learner
  fit_members() ## Fit the members to the dataset

## If you want to build your own metalearner you'll have to do so manually13
## using14
stackData <- as_tibble(my_stack)

## Use the stacked data to get a prediction17
stack_preds <- stack_mod %>% predict(new_data = test)

kag_sub <- stack_preds %>%
  bind_cols(., test) %>% #Bind predictions with test data
  select(datetime, .pred) %>% #Just keep datetime and prediction variables
  rename(count=.pred) %>% #rename pred to count (for submission to Kaggle)
  mutate(count = exp(count)) %>%  
  mutate(count=pmax(0, count)) %>% #pointwise max of (0, prediction)
  mutate(datetime=as.character(format(datetime))) #needed for right format to Kaggle

## Write out the file
vroom_write(x=kag_sub, file="BikeShare/StackedModel.csv", delim=",")
