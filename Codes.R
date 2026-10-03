library(ggcorrplot)
library(car)
library(psych)
library(glmnet)
library(pROC)

d <- read.csv("data.csv")
dd <- d
attach(d)

par(mfrow = c(1, 2))

boxplot(Radius, col = "tomato", main = "Boxplot of Radius")
hist(Radius, col = "pink")

boxplot(Texture, col = "tomato", main = "Boxplot of Texture")
hist(Texture, col = "pink")

boxplot(Perimeter, col = "tomato", main = "Boxplot of Perimeter")
hist(Perimeter, col = "pink")

boxplot(Area, col = "tomato", main = "Boxplot of Area")
hist(Area, col = "pink")

boxplot(Smoothness, col = "tomato", main = "Boxplot of Smoothness")
hist(Smoothness, col = "pink")

boxplot(Compactness, col = "tomato", main = "Boxplot of Compactness")
hist(Compactness, col = "pink")

boxplot(Concave_Points, col = "tomato", main = "Boxplot of Concave points")
hist(Concave_Points, col = "pink")

boxplot(Concavity, col = "tomato", main = "Boxplot of Concavity")
hist(Concavity, col = "pink")

boxplot(Symmetry, col = "tomato", main = "Boxplot of Symmetry")
hist(Symmetry, col = "pink")

boxplot(Fractal_Dimension, col = "tomato", main = "Boxplot of Fractal dimension")
hist(Fractal_Dimension, col = "pink")

cor_matrix <- cor(dd, method = "pearson", use = "complete.obs")
ggcorrplot(cor_matrix,
           method = "circle",
           type = "lower",
           lab = TRUE,
           title = "Spearman Rank Correlation Heatmap",
           colors = c("blue", "white", "red"),
           ggtheme = theme_minimal())

vTert <- quantile(d$Concave_Points, c(0:3/3))
d$tert <- with(d,
               cut(Concave_Points,
                   vTert,
                   include.lowest = T,
                   labels = c("Low", "Medium", "High")))

par(mfrow = c(1, 2))

boxplot(Radius ~ d$tert, data = d,
        main = "Radius vs Concave points group",
        xlab = "Concave Points Group",
        ylab = "Radius",
        col = c("tomato", "skyblue", "steelblue"))
boxplot(Texture ~ d$tert, data = d,
        main = "Texture vs Concave points group",
        xlab = "Concave Points Group",
        ylab = "Texture",
        col = c("tomato", "skyblue", "steelblue"))

boxplot(Perimeter ~ d$tert, data = d,
        main = "Perimeter vs Concave points group",
        xlab = "Concave Points Group",
        ylab = "Perimeter",
        col = c("tomato", "skyblue", "steelblue"))
boxplot(Area ~ d$tert, data = d,
        main = "Area vs Concave points group",
        xlab = "Concave Points Group",
        ylab = "Area",
        col = c("tomato", "skyblue", "steelblue"))

boxplot(Smoothness ~ d$tert, data = d,
        main = "Smoothness vs Concave points group",
        xlab = "Concave Points Group",
        ylab = "Smoothness",
        col = c("tomato", "skyblue", "steelblue"))
boxplot(Compactness ~ d$tert, data = d,
        main = "Compactness vs Concave points group",
        xlab = "Concave Points Group",
        ylab = "Compactness",
        col = c("tomato", "skyblue", "steelblue"))

boxplot(Concavity ~ d$tert, data = d,
        main = "Concavity vs Concave points group",
        xlab = "Concave Points Group",
        ylab = "Concavity",
        col = c("tomato", "skyblue", "steelblue"))
boxplot(Symmetry ~ d$tert, data = d,
        main = "Symmetry vs Concave points group",
        xlab = "Concave Points Group",
        ylab = "Symmetry",
        col = c("tomato", "skyblue", "steelblue"))

par(mfrow = c(1, 1))
boxplot(Fractal_Dimension ~ d$tert, data = d,
        main = "Fractal dimension vs Concave points groups",
        xlab = "Concave Points Group",
        ylab = "Fractal dimension",
        col = c("tomato", "skyblue", "steelblue"))

d$tert <- NULL

model2 <- glm(Diagnosis ~ ., data = d, family = binomial)
vif(model2)

sample_index <- sample(1:nrow(d), size = 0.8 * nrow(d))
train_data <- d[sample_index, ]
test_data  <- d[-sample_index, ]

x_train <- as.matrix(subset(train_data, select = -Diagnosis))
y_train <- train_data$Diagnosis

x_test <- as.matrix(subset(test_data, select = -Diagnosis))
y_test <- test_data$Diagnosis

pca_model <- prcomp(x_train, center = TRUE, scale. = TRUE)
explained_var <- cumsum(pca_model$sdev^2) / sum(pca_model$sdev^2)
num_components <- which(explained_var >= 0.95)[1]
explained_var
num_components

pca_train <- as.data.frame(pca_model$x[, 1:num_components])
pca_test  <- as.data.frame(predict(pca_model, newdata = x_test)[, 1:num_components])
logit_model <- glm(y_train ~ ., data = pca_train, family = "binomial")

test_probs <- predict(logit_model, newdata = pca_test, type = "response")
test_pred  <- ifelse(test_probs > 0.5, 1, 0)
conf_mat   <- table(Predicted = test_pred, Actual = y_test)
accuracy   <- mean(test_pred == y_test)
cat("Confusion Matrix:\n")
print(conf_mat)
cat("Accuracy:", round(accuracy, 4), "\n")

roc_obj <- roc(y_test, test_probs)
cat("AUC:", auc(roc_obj), "\n")
plot(roc_obj, main = "ROC Curve - PCA + Logistic Regression",
     col = "blue", lwd = 2, print.auc = TRUE)

scree_values  <- pca_model$sdev^2
scree_percent <- scree_values / sum(scree_values) * 100
plot(scree_percent, type = "b",
     xlab = "Principal Component",
     ylab = "Percentage of Variance Explained",
     main = "Scree Plot",
     pch = 19)

train_scaled <- scale(train_data)
means <- attr(train_scaled, "scaled:center")
sds   <- attr(train_scaled, "scaled:scale")
test_scaled <- scale(test_data, center = means, scale = sds)

fa_model <- fa(train_scaled, nfactors = 2, rotate = "varimax", fm = "ml")
L <- as.matrix(fa_model$loadings)
train_scores <- train_scaled %*% L
test_scores  <- test_scaled %*% L
train_df <- data.frame(train_scores, y = y_train)
test_df  <- data.frame(test_scores,  y = y_test)

model <- glm(y ~ ., data = train_df, family = binomial)
probs <- predict(model, newdata = test_df, type = "response")
preds <- ifelse(probs > 0.5, 1, 0)
acc   <- mean(preds == test_df$y)
print(paste("Accuracy:", round(acc, 3)))
print(table(Predicted = preds, Actual = test_df$y))

roc_obj <- roc(test_df$y, probs)
plot(roc_obj, main = "FA+ROC Curve", col = "red", print.auc = TRUE)
auc(roc_obj)

cv_model    <- cv.glmnet(x_train, y_train, alpha = 1, family = "binomial")
best_lambda <- cv_model$lambda.min
lasso_model <- glmnet(x_train, y_train, alpha = 1, lambda = best_lambda, family = "binomial")

test_probs <- predict(lasso_model, s = best_lambda, newx = x_test, type = "response")
test_pred  <- ifelse(test_probs > 0.5, 1, 0)
conf_mat   <- table(Predicted = test_pred, Actual = y_test)
print(conf_mat)
accuracy <- mean(test_pred == y_test)
cat("Accuracy:", accuracy, "\n")

roc_obj <- roc(y_test, as.numeric(test_probs))
cat("AUC:", auc(roc_obj), "\n")
plot(roc_obj, main = "ROC Curve - Lasso Logistic Regression",
     lwd = 2, print.auc = TRUE)

coef(lasso_model)