## Load data

library(rio)
don <- import("https://github.com/ktmccabe/teachingdata/blob/main/donation.dta?raw=true")


## outcome
table(don$donation)
don$donation <- as.factor(don$donation)



## build a model (choose "features" you think will be good for prediction)

## you will want to remove missing data first
donsub <- don[, c("donation", "NetWorth", "Edsum")]
donsub <- na.omit(donsub)

fit <- glm(donation ~ NetWorth + Edsum, 
           family = binomial(link = "logit"), data = donsub)


## generate probability of donation for each observation
donsub$prob <- predict(fit, type = "response")

## set a prediction threshold
donsub$pred <- ifelse(donsub$prob > 0.05, 1, 0)

## accuracy- proportion where prediction matches reality
mean(donsub$pred == donsub$donation)

## confusion matrix
table(truth = donsub$donation, predicted = donsub$pred)

## where did we miss
table(actual = donsub$donation, pred = donsub$pred)
truepos <- table(actual = donsub$donation, pred = donsub$pred)[2,2]
falsepos <- table(actual = donsub$donation, pred = donsub$pred)[1,2]
trueneg <- table(actual = donsub$donation, pred = donsub$pred)[1,1]
falseneg <- table(actual = donsub$donation, pred = donsub$pred)[2,1]

## precision
precision <- truepos/(truepos + falsepos)

## specificity
specificity <- trueneg / (trueneg + falsepos)

## false positive rate
falsepos <- falsepos/(trueneg+ falsepos)

## recall aka sensitivity
recall <- truepos/(truepos + falseneg)

## f-score, combination of precision/recall
F1 <- (2 * precision * recall) / (precision + recall)


# install.packages("pROC")
library(pROC)
ROC <- roc(response = donsub$donation,
                  predictor = donsub$pred)
plot(ROC, print.thres = "best")
auc(ROC)


## Continous option
fit <- lm(total_donation ~ NetWorth + Edsum, data = don)

## Root mean squared error
rmse <- sqrt(sum(residuals(fit)^2)/fit$df.residual)
rmse


## tidymodels

library(tidyverse)
library(tidymodels)
library(textrecipes)
library(themis)
library(yardstick)
library(glmnet)
library(ranger)

## Split into train vs. test
set.seed(123)
df_split <- donsub %>% initial_split(prop = 0.8)
df_train <- training(df_split)
df_test <- testing(df_split)


## recipe for pre-processing variables
df_rec <-   recipe(donation ~ NetWorth + Edsum, data = df_train) %>%
  
  
  ## Downsample based on outcome
  themis::step_downsample(donation) 



## Specify model: Three examples

## A penalized logistic regression model
logit_mod <- logistic_reg() %>%
  set_mode("classification") %>%
  set_engine("glm")

## A penalized logistic model with penalty tuning
logit_tune <- logistic_reg(penalty=tune(), mixture = 1) %>%
  set_mode("classification") %>%
  set_engine("glmnet")
lambda_grid <- grid_regular(penalty(), levels = 30)

## Random Forest
rf_spec <- 
  rand_forest() %>% 
  set_engine("ranger") %>% 
  set_mode("classification")



## workflow penalized logit
df_wf <- workflow() %>%
  add_recipe(df_rec) %>%
  add_model(logit_mod)

## workflow for the tuning specification
df_wft <- workflow() %>%
  add_recipe(df_rec) %>%
  add_model(logit_tune)

## workflow for random forest
df_wfrf <- workflow() %>%
  add_recipe(df_rec) %>%
  add_model(rf_spec)



## cross-validation
set.seed(123)
df_folds <- vfold_cv(df_train, v=5)


## Fitting model on training data

## Basic model fit to training data as one sample, no cross-validation
df_basic <- df_wf %>%
  fit(data = df_train)

## Cross-validation on penalized logistic model
my_metrics <- yardstick::metric_set(accuracy, ppv, npv)

df_cv <- fit_resamples(
  df_wf,
  df_folds,
  control = control_resamples(save_pred = TRUE),
  metrics = my_metrics
)

## Tuning the penalized logistic model
df_rs <- tune_grid(
  df_wft,
  df_folds,
  grid=lambda_grid, # note the use of the grid here
  control = control_resamples(save_pred = TRUE),
  metrics = my_metrics
)

## Random forest with cross-validation
df_rf <- fit_resamples(
  df_wfrf,
  df_folds,
  control = control_resamples(save_pred = TRUE),
  metrics = my_metrics
)


# Add prediction columns to training data
results <- df_train  %>% select(donation) %>% 
  bind_cols(df_basic %>% 
              predict(new_data = df_train))


results %>% 
  conf_mat(truth = donation, estimate = .pred_class)


eval_metrics <- metric_set(accuracy, ppv, npv)
eval_metrics(data = results, 
             truth = donation, 
             estimate = .pred_class)


## cross-validated logistic
resultscv <- collect_metrics(df_cv)
resultscv


## cross-validated random forest
resultsrf <- collect_metrics(df_rf)
resultsrf



resultstuning <- collect_metrics(df_rs)

chosen_acc <- df_rs %>% select_best("accuracy")
chosen_acc



## Saving the random forest model
final_wfrf <- fit(df_wfrf, df_train)


## Saving the tuned workflow-- note we include are chosen lambda
final_wft <- finalize_workflow(df_wft, chosen_acc)  %>%
  fit(data=df_train)


## saving decision tree
saveRDS(final_wfrf, file = "donationclassifierRF.RDS")

## saving tuned workflow
saveRDS(final_wft, file = "donationclassifierT.RDS")


testres <- df_test %>% select(donation) %>%
  bind_cols(predict(final_wft, new_data=df_test))

testres_cf <- testres %>% 
  conf_mat(truth = donation, estimate = .pred_class)

test_metrics <- metric_set(accuracy, ppv, npv)
test_metrics(data = testres, truth = donation, estimate = .pred_class)



newdf <-  data.frame(NetWorth = c(1, 3,3),
                     Edsum = c(2, 5, 6))


## cross-validated tree model
donationclassRF <- readRDS("donationclassifierRF.RDS")
predict(donationclassRF, newdf, type = "class")


## Tune penalized logistic
donationclassT <- readRDS("donationclassifierT.RDS")
predict(donationclassT, newdf, type = "class")

