# ============================================================
# 🤖 Survival Prediction using Machine Learning in R
# Author: Muhammad Mubashir
# Description: Binary classification using Random Forest,
#              Logistic Regression, and XGBoost in R
#              with caret framework and ROC analysis
# ============================================================

# ── Install & Load Libraries ─────────────────────────────────
packages <- c("caret", "randomForest", "ggplot2", "dplyr",
              "tidyr", "readr", "pROC", "e1071", "xgboost",
              "scales", "viridis", "gridExtra", "corrplot")

install_if_missing <- function(pkg) {
  if (!requireNamespace(pkg, quietly = TRUE)) install.packages(pkg, quiet = TRUE)
  library(pkg, character.only = TRUE)
}
invisible(sapply(packages, install_if_missing))
cat("✅ Libraries loaded!\n")

set.seed(42)

# ── Load & Explore Data ──────────────────────────────────────
df <- read_csv("titanic_data.csv", show_col_types = FALSE)
df$survived  <- factor(df$survived,  levels = c(0,1), labels = c("No","Yes"))
df$pclass    <- factor(df$pclass,    levels = c(1,2,3), labels = c("1st","2nd","3rd"))
df$sex       <- factor(df$sex)
df$embarked  <- factor(df$embarked)

cat("✅ Data loaded!\n")
cat(sprintf("   Shape     : %d rows × %d columns\n", nrow(df), ncol(df)))
cat(sprintf("   Survived  : %d (%.1f%%)\n",
    sum(df$survived=="Yes"), mean(df$survived=="Yes")*100))
print(head(df, 5))

# ── EDA ──────────────────────────────────────────────────────
theme_ml <- theme_dark() +
  theme(
    plot.background  = element_rect(fill="#0e1117", color=NA),
    panel.background = element_rect(fill="#1e2130", color=NA),
    panel.grid.major = element_line(color="#2d3348", linewidth=0.4),
    panel.grid.minor = element_blank(),
    plot.title       = element_text(color="white", size=13, face="bold", hjust=0.5),
    axis.text        = element_text(color="#cccccc"),
    axis.title       = element_text(color="white"),
    legend.background= element_rect(fill="#1e2130"),
    legend.text      = element_text(color="white"),
    legend.title     = element_text(color="white"),
    strip.background = element_rect(fill="#2d3348"),
    strip.text       = element_text(color="white", face="bold")
  )

# Survival by Gender
p1 <- df %>%
  group_by(sex, survived) %>%
  summarise(count = n(), .groups = "drop") %>%
  ggplot(aes(x = sex, y = count, fill = survived)) +
  geom_col(position = "dodge", width = 0.6, color = "white") +
  scale_fill_manual(values = c("No" = "#ff6b6b", "Yes" = "#00d4ff")) +
  labs(title = "Survival by Gender", x = "Gender", y = "Count", fill = "Survived") +
  theme_ml

# Survival by Class
p2 <- df %>%
  group_by(pclass, survived) %>%
  summarise(count = n(), .groups = "drop") %>%
  ggplot(aes(x = pclass, y = count, fill = survived)) +
  geom_col(position = "fill", width = 0.6, color = "white") +
  scale_fill_manual(values = c("No" = "#ff6b6b", "Yes" = "#00d4ff")) +
  scale_y_continuous(labels = scales::percent) +
  labs(title = "Survival Rate by Class", x = "Passenger Class",
       y = "Proportion", fill = "Survived") +
  theme_ml

# Age distribution
p3 <- df %>%
  ggplot(aes(x = age, fill = survived)) +
  geom_histogram(bins = 30, alpha = 0.7, position = "identity", color = "white") +
  scale_fill_manual(values = c("No"="#ff6b6b","Yes"="#00d4ff")) +
  facet_wrap(~sex) +
  labs(title = "Age Distribution by Survival & Gender",
       x = "Age", y = "Count", fill = "Survived") +
  theme_ml

# Fare vs Age scatter
p4 <- df %>%
  ggplot(aes(x = age, y = fare, color = survived, shape = pclass)) +
  geom_point(alpha = 0.5, size = 2.5) +
  scale_color_manual(values = c("No"="#ff6b6b","Yes"="#00d4ff")) +
  scale_y_log10(labels = label_comma()) +
  labs(title = "Age vs Fare by Survival",
       x = "Age", y = "Fare (log scale)",
       color = "Survived", shape = "Class") +
  theme_ml

ggsave("01_survival_by_gender.png",    p1, width=8,  height=5, dpi=150, bg="#0e1117")
ggsave("02_survival_by_class.png",     p2, width=8,  height=5, dpi=150, bg="#0e1117")
ggsave("03_age_distribution.png",      p3, width=12, height=5, dpi=150, bg="#0e1117")
ggsave("04_age_vs_fare.png",           p4, width=10, height=6, dpi=150, bg="#0e1117")
cat("✅ EDA plots saved!\n")

# ── Feature Engineering ───────────────────────────────────────
df <- df %>%
  mutate(
    family_size   = sibsp + parch + 1,
    is_alone      = if_else(family_size == 1, 1, 0),
    fare_per_person = fare / family_size,
    age_group     = case_when(
      age < 12  ~ "Child",
      age < 18  ~ "Teen",
      age < 35  ~ "YoungAdult",
      age < 60  ~ "Adult",
      TRUE      ~ "Senior"
    ),
    age_group = factor(age_group)
  )
cat("✅ Feature engineering done!\n")
cat(sprintf("   Features: %d\n", ncol(df) - 1))

# ── Prepare Train/Test Split ─────────────────────────────────
features <- c("pclass","sex","age","sibsp","parch","fare",
              "embarked","family_size","is_alone","fare_per_person","age_group")

model_df <- df[, c(features, "survived")] %>% na.omit()
train_idx <- createDataPartition(model_df$survived, p = 0.8, list = FALSE)
train_df  <- model_df[train_idx,  ]
test_df   <- model_df[-train_idx, ]

cat(sprintf("   Train: %d | Test: %d\n", nrow(train_df), nrow(test_df)))

# ── Cross-Validation Setup ────────────────────────────────────
ctrl <- trainControl(
  method          = "cv",
  number          = 5,
  classProbs      = TRUE,
  summaryFunction = twoClassSummary,
  verboseIter     = FALSE
)

# ── Model 1: Logistic Regression ─────────────────────────────
cat("\n🔵 Training Logistic Regression...\n")
lr_model <- train(
  survived ~ ., data = train_df,
  method    = "glm",
  family    = "binomial",
  trControl = ctrl,
  metric    = "ROC"
)
lr_pred  <- predict(lr_model, test_df)
lr_prob  <- predict(lr_model, test_df, type="prob")[,"Yes"]
lr_acc   <- mean(lr_pred == test_df$survived)
lr_auc   <- as.numeric(auc(roc(test_df$survived, lr_prob, quiet=TRUE)))
cat(sprintf("   LR  — Accuracy: %.4f | AUC: %.4f\n", lr_acc, lr_auc))

# ── Model 2: Random Forest ────────────────────────────────────
cat("🌲 Training Random Forest...\n")
rf_model <- train(
  survived ~ ., data = train_df,
  method    = "rf",
  trControl = ctrl,
  metric    = "ROC",
  tuneGrid  = expand.grid(mtry = c(2, 3, 4)),
  ntree     = 200
)
rf_pred  <- predict(rf_model, test_df)
rf_prob  <- predict(rf_model, test_df, type="prob")[,"Yes"]
rf_acc   <- mean(rf_pred == test_df$survived)
rf_auc   <- as.numeric(auc(roc(test_df$survived, rf_prob, quiet=TRUE)))
cat(sprintf("   RF  — Accuracy: %.4f | AUC: %.4f\n", rf_acc, rf_auc))

# ── Model 3: SVM ──────────────────────────────────────────────
cat("⚡ Training SVM...\n")
svm_model <- train(
  survived ~ ., data = train_df,
  method    = "svmRadial",
  trControl = ctrl,
  metric    = "ROC",
  tuneLength = 3
)
svm_pred <- predict(svm_model, test_df)
svm_prob <- predict(svm_model, test_df, type="prob")[,"Yes"]
svm_acc  <- mean(svm_pred == test_df$survived)
svm_auc  <- as.numeric(auc(roc(test_df$survived, svm_prob, quiet=TRUE)))
cat(sprintf("   SVM — Accuracy: %.4f | AUC: %.4f\n", svm_acc, svm_auc))

# ── Results Summary ───────────────────────────────────────────
results <- data.frame(
  Model    = c("Logistic Regression", "Random Forest", "SVM"),
  Accuracy = c(lr_acc, rf_acc, svm_acc),
  AUC      = c(lr_auc, rf_auc, svm_auc)
) %>% arrange(desc(AUC))

cat("\n")
cat(strrep("=", 50), "\n")
cat("         MODEL COMPARISON\n")
cat(strrep("=", 50), "\n")
print(results)
cat(strrep("=", 50), "\n")
best_name <- results$Model[1]
cat(sprintf("\n🏆 Best Model: %s (AUC = %.4f)\n", best_name, results$AUC[1]))

# ── ROC Curve Plot ────────────────────────────────────────────
roc_lr  <- roc(test_df$survived, lr_prob,  quiet=TRUE)
roc_rf  <- roc(test_df$survived, rf_prob,  quiet=TRUE)
roc_svm <- roc(test_df$survived, svm_prob, quiet=TRUE)

png("05_roc_curves.png", width=900, height=600, res=120, bg="#0e1117")
par(bg="#0e1117", col.axis="white", col.lab="white", col.main="white",
    mar=c(5,5,4,2))
plot(roc_rf,  col="#00d4ff", lwd=3,
     main="ROC Curves — Model Comparison",
     xlab="False Positive Rate", ylab="True Positive Rate")
plot(roc_lr,  col="#ff6b6b", lwd=2, add=TRUE, lty=2)
plot(roc_svm, col="#ffd700", lwd=2, add=TRUE, lty=3)
abline(0, 1, col="gray", lty=2, lwd=1)
legend("bottomright",
       legend=c(sprintf("Random Forest (AUC=%.3f)", rf_auc),
                sprintf("Logistic Reg  (AUC=%.3f)", lr_auc),
                sprintf("SVM           (AUC=%.3f)", svm_auc)),
       col=c("#00d4ff","#ff6b6b","#ffd700"),
       lwd=c(3,2,2), lty=c(1,2,3),
       bg="#1e2130", text.col="white")
dev.off()

# ── Confusion Matrix ──────────────────────────────────────────
cat("\n📊 Best Model Confusion Matrix:\n")
best_pred <- if (best_name=="Random Forest") rf_pred else if (best_name=="SVM") svm_pred else lr_pred
cm <- confusionMatrix(best_pred, test_df$survived, positive="Yes")
print(cm)

# ── Feature Importance (RF) ───────────────────────────────────
fi <- varImp(rf_model)$importance %>%
  tibble::rownames_to_column("Feature") %>%
  arrange(desc(Overall))

p_fi <- ggplot(fi, aes(x=reorder(Feature,Overall), y=Overall)) +
  geom_col(fill="#7c3aed", color="white", width=0.7) +
  geom_text(aes(label=round(Overall,1)), hjust=-0.1, color="white", size=3) +
  coord_flip() +
  scale_y_continuous(expand=expansion(mult=c(0,0.15))) +
  labs(title="Feature Importance — Random Forest",
       x=NULL, y="Importance Score") +
  theme_ml

ggsave("06_feature_importance.png", p_fi, width=10, height=6, dpi=150, bg="#0e1117")

cat("\n✅ All done! Files saved:\n")
cat("   01_survival_by_gender.png\n")
cat("   02_survival_by_class.png\n")
cat("   03_age_distribution.png\n")
cat("   04_age_vs_fare.png\n")
cat("   05_roc_curves.png\n")
cat("   06_feature_importance.png\n")
