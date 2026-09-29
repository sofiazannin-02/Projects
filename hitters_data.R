# DATA LOADING 
library(ISLR2)
library(ggplot2)
library(corrplot)
library(glmnet)

data('Hitters')
dim(Hitters)
str(Hitters)

# number of NAs per variable
colSums(is.na(Hitters))

hitters = na.omit(Hitters)
dim(hitters)

# EDA
# distribution of y
summary(hitters$Salary)

ggplot(hitters, aes(x = Salary)) +
  geom_histogram(bins = 30, color = "white") +
  labs(
    title = "Distribution of players' salaries",
    x = "Salary (thousands of dollars)",
    y = "Number of players"
  ) +
  theme_minimal()
# distribution is asymmetric

# logaritmic distribution since Salary has only positive values
ggplot(hitters, aes(x = log(Salary))) +
  geom_histogram(bins = 30, color = "white") +
  labs(
    title = "Distribution of log-transformed salaries",
    x = "log(Salary)",
    y = "Number of players"
  ) +
  theme_minimal()
# Salary is now more symmetric


# numerical variables 
numeric_vars = names(hitters[sapply(hitters, is.numeric)])

summary(hitters[numeric_vars])

# scatterplot for each variable
numeric_predictors = setdiff(numeric_vars, "Salary")

for (var in numeric_predictors) {
  
  p = ggplot(hitters, aes(x = .data[[var]], y = log(Salary))) +
    geom_point(alpha = 0.6) +
    labs(
      title = paste("Salary vs", var),
      x = var,
      y = "log(Salary) (thousands of dollars)"
    ) +
    theme_minimal()
  
  print(p)
}


# categorical vars
hitters$League = factor(hitters$League)
hitters$Division = factor(hitters$Division)
hitters$NewLeague = factor(hitters$NewLeague)

categorical_vars = c('League', 'Division', 'NewLeague')

for (var in categorical_vars) {
  
  p = ggplot(hitters, aes(x = .data[[var]], y = log(Salary))) +
    geom_boxplot() +
    labs(
      title = paste("Salary by", var),
      x = var,
      y = "log(Salary) (thousands of dollars)"
    ) +
    theme_minimal()
  
  print(p)
}


# correlation between numeric vars
cor_matrix = cor(
  hitters[, numeric_vars],
  use = "complete.obs"
)

corrplot(
  cor_matrix,
  method = "color",
  type = "upper",
  tl.cex = 0.7,
  addCoef.col = "black",
  number.cex = 0.5
)

# interesting observations about this, many variables are highly correlated and 
# can lead to multicollinearity 

# MODELS

# Response
y <- hitters$Salary

# Predictors
x <- model.matrix(Salary ~ ., data = hitters)[, -1]


# ============================================================
# 1. Best subset selection using BIC
# ============================================================

regfit <- regsubsets(
  Salary ~ .,
  data = hitters,
  nvmax = 19
)

regfit_summary <- summary(regfit)

# Number of predictors selected by BIC
best_size <- which.min(regfit_summary$bic)

best_size

# Coefficients of the selected model
coef(regfit, best_size)

# Names of selected predictors
selected_vars <- names(coef(regfit, best_size))[-1]

selected_vars


# ============================================================
# 2. Create 10 common folds
# ============================================================

set.seed(123)

K <- 10

foldid <- sample(
  rep(1:K, length.out = nrow(hitters))
)


# ============================================================
# 3. Cross-validation for best subset selection
#    BIC selection is repeated within each training fold
# ============================================================

mse_regsubsets <- numeric(K)

for (k in 1:K) {
  
  train <- hitters[foldid != k, ]
  test  <- hitters[foldid == k, ]
  
  # Select variables using BIC on training data only
  regfit_train <- regsubsets(
    Salary ~ .,
    data = train,
    nvmax = 19
  )
  
  regfit_train_summary <- summary(regfit_train)
  
  best_size_train <- which.min(regfit_train_summary$bic)
  
  # Extract selected coefficients
  coef_train <- coef(regfit_train, best_size_train)
  
  selected_vars_train <- names(coef_train)[-1]
  
  # Convert dummy-variable names back to original factor names
  selected_vars_train[selected_vars_train == "LeagueN"] <- "League"
  selected_vars_train[selected_vars_train == "DivisionW"] <- "Division"
  selected_vars_train[selected_vars_train == "NewLeagueN"] <- "NewLeague"
  
  # Remove duplicates if multiple factor levels were selected
  selected_vars_train <- unique(selected_vars_train)
  
  # Build formula
  formula_train <- as.formula(
    paste(
      "Salary ~",
      paste(selected_vars_train, collapse = " + ")
    )
  )
  
  # Fit linear model using selected variables
  model_train <- lm(
    formula_train,
    data = train
  )
  
  # Predict held-out observations
  predictions <- predict(
    model_train,
    newdata = test
  )
  
  # Calculate test MSE
  mse_regsubsets[k] <- mean(
    (test$Salary - predictions)^2
  )
}

mse.regsubsets <- mean(mse_regsubsets)

rmse.regsubsets <- sqrt(mse.regsubsets)


# ============================================================
# 4. Ridge regression
# ============================================================

set.seed(123)

cv_ridge <- cv.glmnet(
  x,
  y,
  alpha = 0,
  foldid = foldid
)

mse.ridge.min <- cv_ridge$cvm[
  cv_ridge$lambda == cv_ridge$lambda.min
]

mse.ridge.1se <- cv_ridge$cvm[
  cv_ridge$lambda == cv_ridge$lambda.1se
]


# ============================================================
# 5. Lasso regression
# ============================================================

set.seed(123)

cv_lasso <- cv.glmnet(
  x,
  y,
  alpha = 1,
  foldid = foldid
)

mse.lasso.min <- cv_lasso$cvm[
  cv_lasso$lambda == cv_lasso$lambda.min
]

mse.lasso.1se <- cv_lasso$cvm[
  cv_lasso$lambda == cv_lasso$lambda.1se
]


# ============================================================
# 6. Elastic Net regression
# ============================================================

set.seed(123)

cv_enet <- cv.glmnet(
  x,
  y,
  alpha = 0.5,
  foldid = foldid
)

mse.enet.min <- cv_enet$cvm[
  cv_enet$lambda == cv_enet$lambda.min
]

mse.enet.1se <- cv_enet$cvm[
  cv_enet$lambda == cv_enet$lambda.1se
]


# ============================================================
# 7. Comparison table
# ============================================================

comparison_table <- data.frame(
  Model = c(
    "Best subset (BIC)",
    "Ridge",
    "Lasso",
    "Elastic Net (alpha = 0.5)"
  ),
  
  MSE_min = c(
    mse.regsubsets,
    mse.ridge.min,
    mse.lasso.min,
    mse.enet.min
  ),
  
  MSE_1se = c(
    NA,
    mse.ridge.1se,
    mse.lasso.1se,
    mse.enet.1se
  )
)

comparison_table$RMSE_min <- sqrt(
  comparison_table$MSE_min
)

comparison_table$RMSE_1se <- sqrt(
  comparison_table$MSE_1se
)

comparison_table

# Among the considered models, Lasso achieved the lowest cross-validated RMSE at 
# lambda.min (334.67), closely followed by Elastic Net (335.36) and Ridge (336.34). 
# The differences among the penalized regression methods were small. The BIC-selected 
# best subset showed a higher RMSE (358.81), suggesting a trade-off between model 
# parsimony and predictive performance. These results should be interpreted as 
# exploratory because hyperparameter selection and performance estimation were not 
# fully separated.


# ============================================================
# 8. Number of non-zero coefficients
# ============================================================

# Best subset: number of selected predictors
n_regsubsets <- best_size

# Ridge: coefficients at lambda.min
coef_ridge <- coef(
  cv_ridge,
  s = "lambda.min"
)

n_ridge <- sum(coef_ridge[-1] != 0)


# Lasso: coefficients at lambda.min
coef_lasso <- coef(
  cv_lasso,
  s = "lambda.min"
)

n_lasso <- sum(coef_lasso[-1] != 0)


# Elastic Net: coefficients at lambda.min
coef_enet <- coef(
  cv_enet,
  s = "lambda.min"
)

n_enet <- sum(coef_enet[-1] != 0)


number_predictors <- data.frame(
  Model = c(
    "Best subset (BIC)",
    "Ridge",
    "Lasso",
    "Elastic Net"
  ),
  
  Nonzero_coefficients = c(
    n_regsubsets,
    n_ridge,
    n_lasso,
    n_enet
  )
)

number_predictors


# ============================================================
# 9. Display selected coefficients
# ============================================================

# Best subset
coef(regfit, best_size)

# Ridge
coef(cv_ridge, s = "lambda.min")

# Lasso
coef(cv_lasso, s = "lambda.min")

# Elastic Net
coef(cv_enet, s = "lambda.min")



# GRAFICO PREDETTI-X
# ============================================================
# OBSERVED VS PREDICTED + ANALISI DEGLI ERRORI
# ============================================================

library(ggplot2)

# ------------------------------------------------------------
# 1. Predizioni
# ------------------------------------------------------------

pred_ridge <- as.numeric(
  predict(cv_ridge, newx = x, s = "lambda.min")
)

pred_lasso <- as.numeric(
  predict(cv_lasso, newx = x, s = "lambda.min")
)

pred_enet <- as.numeric(
  predict(cv_enet, newx = x, s = "lambda.min")
)

# ------------------------------------------------------------
# 2. Dataset con osservati, predetti e residui
# ------------------------------------------------------------

predictions <- data.frame(
  Player = rownames(hitters),
  Salary = y,
  Ridge = pred_ridge,
  Lasso = pred_lasso,
  ElasticNet = pred_enet
)

predictions$Residual_Ridge <- predictions$Salary - predictions$Ridge
predictions$Residual_Lasso <- predictions$Salary - predictions$Lasso
predictions$Residual_ENet <- predictions$Salary - predictions$ElasticNet

# ------------------------------------------------------------
# 3. Observed vs Predicted
# ------------------------------------------------------------

ggplot(predictions, aes(x = Salary, y = Ridge)) +
  geom_point(alpha = 0.6) +
  geom_abline(intercept = 0, slope = 1, linetype = "dashed") +
  labs(
    title = "Observed vs Predicted Salary - Ridge",
    x = "Observed Salary",
    y = "Predicted Salary"
  ) +
  theme_minimal()

ggplot(predictions, aes(x = Salary, y = Lasso)) +
  geom_point(alpha = 0.6) +
  geom_abline(intercept = 0, slope = 1, linetype = "dashed") +
  labs(
    title = "Observed vs Predicted Salary - Lasso",
    x = "Observed Salary",
    y = "Predicted Salary"
  ) +
  theme_minimal()

ggplot(predictions, aes(x = Salary, y = ElasticNet)) +
  geom_point(alpha = 0.6) +
  geom_abline(intercept = 0, slope = 1, linetype = "dashed") +
  labs(
    title = "Observed vs Predicted Salary - Elastic Net",
    x = "Observed Salary",
    y = "Predicted Salary"
  ) +
  theme_minimal()

# ------------------------------------------------------------
# 4. Residuals vs Predicted
# ------------------------------------------------------------

ggplot(predictions, aes(x = Ridge, y = Residual_Ridge)) +
  geom_point(alpha = 0.6) +
  geom_hline(yintercept = 0, linetype = "dashed") +
  labs(
    title = "Residuals vs Predicted - Ridge",
    x = "Predicted Salary",
    y = "Residual"
  ) +
  theme_minimal()

ggplot(predictions, aes(x = Lasso, y = Residual_Lasso)) +
  geom_point(alpha = 0.6) +
  geom_hline(yintercept = 0, linetype = "dashed") +
  labs(
    title = "Residuals vs Predicted - Lasso",
    x = "Predicted Salary",
    y = "Residual"
  ) +
  theme_minimal()

ggplot(predictions, aes(x = ElasticNet, y = Residual_ENet)) +
  geom_point(alpha = 0.6) +
  geom_hline(yintercept = 0, linetype = "dashed") +
  labs(
    title = "Residuals vs Predicted - Elastic Net",
    x = "Predicted Salary",
    y = "Residual"
  ) +
  theme_minimal()

# ------------------------------------------------------------
# 5. Residuals vs Observed Salary
# ------------------------------------------------------------

ggplot(predictions, aes(x = Salary, y = Residual_Lasso)) +
  geom_point(alpha = 0.6) +
  geom_hline(yintercept = 0, linetype = "dashed") +
  labs(
    title = "Prediction Errors vs Observed Salary - Lasso",
    x = "Observed Salary",
    y = "Prediction Error"
  ) +
  theme_minimal()

# ------------------------------------------------------------
# 6. 10 osservazioni con maggiore errore assoluto - Lasso
# ------------------------------------------------------------

predictions$AbsError_Lasso <- abs(predictions$Residual_Lasso)

largest_errors <- predictions[
  order(-predictions$AbsError_Lasso),
]

head(
  largest_errors[, c(
    "Player",
    "Salary",
    "Lasso",
    "Residual_Lasso",
    "AbsError_Lasso"
  )],
  10
)
