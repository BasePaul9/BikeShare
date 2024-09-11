library(tidymodels)
library(vroom)
library(patchwork)
library(skimr)
library(DataExplorer)
library(GGally)

train <- vroom("BikeShare/train.csv")
test <- vroom("BikeShare/test.csv")

## Setup and Fit the Linear Regression Model
my_linear_model <- linear_reg() %>% #Type of model
  set_engine("lm") %>% # Engine = What R function to use
  set_mode("regression") %>% # Regression just means quantitative response
  fit(formula = log(count) ~ season + workingday + weather + temp, data=train)

## Generate Predictions Using Linear Model
bike_predictions <- predict(my_linear_model,
                            new_data=test) # Use fit to predict
bike_predictions ## Look at the output

# Format the Predictions for Submission to Kaggle
kaggle_submission <- bike_predictions %>%
  bind_cols(., test) %>% #Bind predictions with test data
  select(datetime, .pred) %>% #Just keep datetime and prediction variables
  rename(count=.pred) %>% #rename pred to count (for submission to Kaggle)
  mutate(count=pmax(0, count)) %>% #pointwise max of (0, prediction)
  mutate(datetime=as.character(format(datetime))) #needed for right format to Kaggle

## Write out the file
vroom_write(x=kaggle_submission, file="BikeShare/LinearPreds.csv", delim=",")
