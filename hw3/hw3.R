pacman::p_load(tidyverse, magrittr, caret, GGally, 
               rstatix, MASS, e1071, forcats, reshape)

#### INGESTION ####

full_data <- read_csv("Auto.csv", show_col_types = FALSE)
full_data %<>% mutate(mpg01 = mpg >= median(mpg), .after=mpg) %>% subset(select=-mpg)
full_data$mpg01 <- factor(full_data$mpg01, levels=c(FALSE, TRUE), labels=c(0, 1))
factor_cols <- c("cylinders", "origin")
cont_cols <- c("displacement", "horsepower", "weight", "acceleration", "year")
full_data %<>% mutate_at(factor_cols, factor)
full_data$cylinders %<>% fct_collapse("<6"=c("3","4","5"))
#full_data$origin %<>% fct_collapse("2+3"=c("2","3"))

#### EXPLORATORY ANALYSIS ####

ggpairs(full_data, columns=c("mpg01", factor_cols),
        mapping=ggplot2::aes(color=mpg01),
        upper=list(discrete="colbar"),
        lower="blank", proportions="auto", legend = 1) +
  theme(legend.position="bottom") + labs(fill="MPG better than Median?")
ggpairs(full_data, columns=c("mpg01", cont_cols),
        mapping=ggplot2::aes(color=mpg01),
        upper=list(continuous="points"),
        lower="blank", legend=1) +
  theme(legend.position="bottom") + labs(fill="MPG better than Median?")

subset(full_data, select=cont_cols) %>% cor %>% round(4) %>% melt %>% 
  ggplot() + geom_tile(aes(x=X1,y=X2,fill=value)) +
  scale_fill_gradient2(low = "blue", high = "red", mid = "white",
                       limit = c(-1,1), name="Correlation") +
  xlab("") + ylab("")

#### VARIABLE SELECTION ####
summary(aov(displacement~mpg01, data=full_data))
summary(aov(weight~mpg01, data=full_data))
summary(aov(horsepower~mpg01, data=full_data))
summary(aov(acceleration~mpg01, data=full_data))
summary(aov(year~mpg01, data=full_data))

data <- full_data

set.seed(1234)
folds_accel <- createMultiFolds(data$mpg01, k = 5, times = 3)
seeds_accel <- sample.int(10000, length(folds)+1)
folds_dwh <- createMultiFolds(data$mpg01, k = 5, times = 3)
seeds_dwh <- sample.int(10000, length(folds)+1)
accel_trControl <- trainControl(method = "repeatedcv", seeds=seeds_accel, index=folds_accel)
dwh_trControl <- trainControl(method = "repeatedcv", seeds=seeds_dwh, index=folds_dwh)

accel_full <- train(mpg01 ~ ., data=data, 
                  method = "glm", family="binomial",
                  trControl=accel_trControl)
accel_rdx <- train(mpg01 ~ . - acceleration, data=data, 
                  method = "glm", family="binomial",
                  trControl=accel_trControl)
accel_redux <- data.frame(Model = c("Full", "W/o Acceleration"),
                          Accuracy = c(accel_full$results$Accuracy,
                                       accel_rdx$results$Accuracy),
                          AIC = c(AIC(accel_full$finalModel),
                                  AIC(accel_rdx$finalModel))
                          )
accel_redux
data %<>% subset(select=-acceleration)

dwh_full <- train(mpg01 ~ ., data=data, 
                     method = "glm", family="binomial",
                     trControl=dwh_trControl)
dwh_disp <- train(mpg01 ~ . - weight - horsepower, data=data, 
                     method = "glm", family="binomial",
                     trControl=dwh_trControl)
dwh_w <- train(mpg01 ~ . - displacement - horsepower, data=data, 
                     method = "glm", family="binomial",
                     trControl=dwh_trControl)
dwh_hp <- train(mpg01 ~ . - weight - displacement, data=data, 
                     method = "glm", family="binomial",
                     trControl=dwh_trControl)
dwh_redux <- data.frame(Model = c("Full", "Only Displacement", "Only Weight", "Only HP"),
                        Accuracy = c(dwh_full$results$Accuracy, 
                                     dwh_disp$results$Accuracy, 
                                     dwh_w$results$Accuracy, 
                                     dwh_hp$results$Accuracy),
                        AIC = c(AIC(dwh_full$finalModel), 
                                AIC(dwh_disp$finalModel), 
                                AIC(dwh_w$finalModel), 
                                AIC(dwh_hp$finalModel))
                        )

dwh_redux

data %<>% subset(select=-c(displacement, horsepower))

trials=100
model_names <- c("LogRegr", "LDA", "QDA", "NaiveBayes", "KNN")
cv_raw <- data.frame(matrix(nrow=0, ncol=length(model_names)))
set.seed(7406)
for (i in seq(1,trials)) {
  flag <- sample(1:nrow(data), replace=FALSE, size=round(0.2 * nrow(data)))
  train <- data[-flag,]
  test <- data[flag,]
  
#### LOGISTIC REGRESSION ####
  regr_model <- glm(mpg01 ~ ., family="binomial", data=train)
  regr_pred <- predict(regr_model, newdata = test, type = "response") %>% round
  regr_acc <- sum(regr_pred == test$mpg01) / nrow(test)
  table(regr_pred, test$mpg01)
  
#### LDA ####
  lda_model <- lda(mpg01 ~ ., data=train)
  lda_pred <- predict(lda_model, newdata=test)
  lda_acc <- sum(lda_pred$class == test$mpg01) / nrow(test)
  table(lda_pred$class, test$mpg01)
  
#### QDA ####
  qda_model <- qda(mpg01 ~ ., data=train)
  qda_pred <- predict(qda_model, newdata=test)
  qda_acc <- sum(qda_pred$class == test$mpg01) / nrow(test)
  table(qda_pred$class, test$mpg01)
  
#### NAIVE BAYES ####
  nb_model <- naiveBayes(mpg01 ~ ., data=train)
  nb_pred <- predict(nb_model, newdata=test)
  nb_acc <- sum(nb_pred == test$mpg01) / nrow(test)
  table(nb_pred, test$mpg01)
  
#### KNN ####
  trnCtrl <- trainControl(method="repeatedcv", number = 5, repeats = 3)
  knn_model <- train(mpg01 ~ ., data=train, method="knn",
                     trControl = trnCtrl, tuneLength = 10)
  knn_pred <- predict(knn_model, newdata=test)
  knn_acc <- sum(knn_pred == test$mpg01) / nrow(test)
  table(knn_pred, test$mpg01)
  
  trial_res <- c(regr_acc, lda_acc, qda_acc, nb_acc, knn_acc)
  cv_raw %<>% rbind(trial_res)
}

cv_results <- data.frame(Mean=sapply(cv_raw,mean), Median=sapply(cv_raw,median), 
                         Variance=sapply(cv_raw, var), row.names = model_names)
cv_results %>% round(4)
