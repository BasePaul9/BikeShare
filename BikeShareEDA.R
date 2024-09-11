library(vroom)
library(patchwork)
library(skimr)
library(DataExplorer)
library(GGally)

sample <- vroom("BikeShare/sampleSubmission.csv")

glimpse(sample)
skim(sample)


train <- vroom("BikeShare/train.csv")

glimpse(train)
skim(train)
plot_intro(train)
plot_correlation(train)
plot_bar(train)
plot_histogram(train)
plot_missing(train)
ggpairs(train)

weather_bar <- ggplot(train, mapping = aes(x = weather)) +
  geom_bar() +
  ggtitle("Histogram of Weather Observations")
weather_bar

temp_trend <- ggplot(train, mapping = aes(x = temp,y = count)) +
  geom_point(position = "jitter") +
  geom_smooth(se = FALSE) + 
  ggtitle("Bike use at Different Temperatures")
temp_trend

test <- vroom("BikeShare/test.csv")

glimpse(test)
skim(test)
plot_intro(test)
plot_correlation(test)
plot_bar(test)
plot_histogram(test)
plot_missing(test)
ggpairs(test)

day_temp <- ggplot(test, mapping = aes(x = datetime, y = temp)) +
  geom_point() +
  ggtitle("Temperature by Day & Time")
day_temp  

reg_seas <- ggplot(train, mapping = aes(x = season, y = registered)) +
  geom_point(position = "jitter") +
  ggtitle("Service Registration by Season")

(weather_bar + temp_trend) / (day_temp + reg_seas) 

ggsave("BikeShare/EDA_plots.jpg")  

ggplot(train, mapping = aes(x = atemp,y = count)) +
  geom_point(position = "jitter") +
  geom_smooth(se = FALSE)


