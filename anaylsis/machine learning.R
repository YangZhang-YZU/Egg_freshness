library(tidyverse)
library(randomForest)
library(caret)
library(e1071)
library(dplyr)

df <- read.table(text = readClipboard(), header = TRUE)

data_ml <- df %>% select(H_albumen, theta, Species)

train_control <- trainControl(method = "LOOCV")
set.seed(123) 
model_lm <- train(H_albumen ~ theta + Species, 
                  data = data_ml, 
                  method = "lm", 
                  trControl = train_control)
set.seed(123)
model_svm <- train(H_albumen ~ theta + Species, 
                   data = data_ml, 
                   method = "svmRadial", 
                   trControl = train_control)
set.seed(123)
model_rf <- train(H_albumen ~ theta + Species, 
                  data = data_ml, 
                  method = "rf", 
                  ntree = 500, 
                  trControl = train_control)

data_ml <- df %>% select(H_albumen, xi_eff, Species)

train_control <- trainControl(method = "LOOCV")

set.seed(142)
model_lm <- train(H_albumen ~ xi_eff + Species, data = data_ml, method = "lm", trControl = train_control)

set.seed(142)
model_svm <- train(H_albumen ~ xi_eff + Species, data = data_ml, method = "svmRadial", trControl = train_control)

set.seed(142)
model_rf <- train(H_albumen ~ xi_eff + Species, data = data_ml, method = "rf", ntree = 500, trControl = train_control)


results_summary <- data.frame(
  Model = c("Linear Regression", "SVM", "Random Forest"),
  R_squared = c(model_lm$results$Rsquared, max(model_svm$results$Rsquared), max(model_rf$results$Rsquared)),
  MAE = c(model_lm$results$MAE, min(model_svm$results$MAE), min(model_rf$results$MAE)),
  RMSE = c(model_lm$results$RMSE, min(model_svm$results$RMSE), min(model_rf$results$RMSE))
)
print(results_summary)

true_r2 <- max(model_rf$results$Rsquared)
print(paste("R²:", round(true_r2, 4)))

n_permutations <- 2000
permuted_r2 <- numeric(n_permutations) 
set.seed(20260425) 

for(i in 1:n_permutations){
  
  shuffled_H <- sample(data_ml$H_albumen)
  
  temp_rf <- randomForest(x = data_ml[, c("xi_eff", "Species")], 
                          y = shuffled_H, 
                          ntree = 200) 
  
  permuted_r2[i] <- tail(temp_rf$rsq, 1)
  
  if(i %% 100 == 0) {
    cat(sprintf("已完成打乱重训: %d / %d 次\n", i, n_permutations))
  }
}

p_value <- (sum(permuted_r2 >= true_r2) + 1) / (n_permutations + 1)


### CR vs. HU
train_class <- df %>% filter(Species == "Chicken") %>% drop_na(Fresh, CR)
test_class  <- df %>% filter(Species != "Chicken") %>% drop_na(Fresh, CR)

train_class$Fresh <- as.factor(train_class$Fresh)
test_class$Fresh <- as.factor(test_class$Fresh)

set.seed(2026)
rf_class <- randomForest(Fresh ~ CR, data = train_class, ntree = 500)
pred_class <- predict(rf_class, newdata = test_class)

confusionMatrix(pred_class, test_class$Fresh)

### RTP vs. Albumen height
df_mixed <- df %>% drop_na(H_albumen, RTP, Species)
df_mixed$Species <- as.factor(df_mixed$Species)

train_control <- trainControl(method = "LOOCV", number = 10)

mixed_model <- train(
  H_albumen ~ RTP + Species, 
  data = df_mixed, 
  method = "rf", 
  ntree = 500, 
  trControl = train_control,
  importance = TRUE  
)

rf_imp <- varImp(mixed_model, scale = TRUE)
print(rf_imp)
