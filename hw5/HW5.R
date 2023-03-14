pacman::p_load(tidyverse, magrittr, aod, MASS, forcats)

raw_data <- read_csv("mushrooms.csv")
raw_data %<>% mutate_if(is.character, as.factor)
levels(raw_data$class) <- c("e" = 0, "p" = 1)
data <- raw_data

# FEATURE PRUNING ####
col_remove_threshold <- 0.8
col_to_remove <- c('stalk-root')
for (i in colnames(data)) {
  max_prop <- table(data[i]) %>% prop.table %>% max
  if (max_prop >= col_remove_threshold) {
    col_to_remove %<>% c(i)
  }
}
col_to_remove

data %<>% mutate(across(where(is.factor), ~ fct_lump_prop(.x,0.10)))

focus_columns = c(
  'class',
  'cap-shape',
  'cap-color',
  'bruises',
  'odor',
  'gill-size',
  'gill-color',
  'stalk-shape',
  'stalk-color-above-ring',
  'stalk-color-below-ring',
  'population',
  'habitat'
)

data <- data[, focus_columns]

# BASELINE MODELS ####
set.seed(7406)
flag <- sample(1:nrow(data), replace=FALSE, size=round(0.25 * nrow(data)))
train <- data[-flag,]
test <- data[flag,]

## LOG REG ####
empty_logit <- glm(class ~ 1, data = train, family="binomial")
full_logit <- glm(class ~ ., data = train, family="binomial")
logit_model <- step(empty_logit, direction = "both",
                    scope = list(upper=full_logit, lower=empty_logit),
                    trace=FALSE)

pred <- predict(logit_model, newdata = test, type="response") %>% round
confusionMatrix(as.factor(pred), test$class)

## KNN ####
trnCtrl <- trainControl(method="repeatedcv", number = 5, repeats = 3)
knn_model <- train(class ~ ., data=train, method="knn",
                   trControl = trnCtrl, tuneLength = 10)
knn_pred <- predict(knn_model, newdata=test)
confusionMatrix(knn_pred, test$class) 
