# load libraries
library(tidyverse)
library(grf)  # causal forests
library(quantregForest)  # quantile regression forests
library(glmnet)
library(randomForest)
library(caret)
library(lubridate)
library(ggplot2)


# ====== READ DATA ======

##  socio
socio_csv_path <- "/Users/finbarrhodes/Documents/Github/ST498_CAPSTONE/FLASH/toShare/socioEcodata.csv"
df_socio <- read_csv(socio_csv_path)

## old flash (for reference)
old_flash <- read_csv("/Users/finbarrhodes/Documents/GitHub/ST498_CAPSTONE/Notebooks/ANON_ID_w_socio_and_clusters.csv")

## new setup (added upper 2.5%)
all_methods_flash <- read_csv("/Users/finbarrhodes/Documents/GitHub/ST498_CAPSTONE/Notebooks/comp_socio_df.csv")
flash <- all_methods_flash[,1:15] # just gets socio features; cluster column still to add
flash['Group'] <- all_methods_flash[,32]
peak_kwh_data <- read_csv("/Users/finbarrhodes/Documents/GitHub/ST498_CAPSTONE/Notebooks/peak_kwh.csv")

# ====== CHOOSE CLUSTERING SETUP ======
cluster_column <- all_methods_flash |> select("Cluster_Comp06_k07")
flash['Cluster'] <- cluster_column


# ====== ADD TARGETS ======

## Calculate Pre and Post averages for each household
household_peak_summary <- peak_kwh_data %>%
  group_by(ANON_ID, Post) %>%
  summarise(
    avg_peak_kwh = mean(peak_kwh, na.rm = TRUE),
    .groups = "drop"
  ) %>%
  # Pivot to get Pre and Post in separate columns
  pivot_wider(
    names_from = Post,
    values_from = avg_peak_kwh,
    names_prefix = "Period_"
  ) %>%
  rename(
    Pre = Period_0,
    Post = Period_1
  ) %>%
  # Calculate differences
  mutate(
    Delta = Post - Pre,  # Absolute difference (Post - Pre)
    Relative_Delta = (Post - Pre) / Pre  # Relative difference
  )


"# Check distribution by treatment group
flash %>%
  group_by(Group) %>%
  summarise(
    n = n(),
    mean_delta = mean(Delta, na.rm = TRUE),
    sd_delta = sd(Delta, na.rm = TRUE),
    mean_rel_delta = mean(Relative_Delta, na.rm = TRUE),
    sd_rel_delta = sd(Relative_Delta, na.rm = TRUE)
  ) %>%
  print()"


# ====== CHOOSE TARGET ======

# establishing target as Delta here
flash <- flash %>%
  left_join(
    household_peak_summary %>% 
      select("ANON_ID", Target = Delta),
    by = "ANON_ID"
  )

flash <- na.omit(flash)

# Create IDs
df_socio <- df_socio |>
  mutate(ID = row_number())

# Create ID dictionary
id_dict <- df_socio |>
  select(HASH_KEY, ID) |>
  deframe()

# Map ANON_ID using id_dict
flash <- flash |> mutate(ANON_ID = id_dict[ANON_ID],
                         Cluster = as.factor(as.integer(Cluster)))

# Train-test split
set.seed(1)
train_idx <- createDataPartition(flash$Target, p = 0.6, list = FALSE)
flash_train <- flash[train_idx, ]
flash_test <- flash[-train_idx, ]

# Define features
features <- c('INCOME_CATEGORY','AGE_GROUP', 'CHILDREN_AT_HOME' )

# Prepare data for analysis
# Y -> Outcome Variable: Peak-hour consumption change
Y <- flash$Target
Y_train <- flash_train$Target
Y_test <- flash_test$Target

# T -> Treatment Variable: Control vs. Intervention Group
T <- flash$Group
T_train <- flash_train$Group
T_test <- flash_test$Group

# X -> Heterogeneity Features: Cluster labels as dummies + socio features
# One-hot encode clusters
cluster_dummies <- model.matrix(~ Cluster - 1, data = flash)
cluster_dummies <- as.data.frame(cluster_dummies)

# One-hot encode socio features
socio_dummies <- model.matrix(~ . - 1, data = flash[features])
socio_dummies <- as.data.frame(socio_dummies)

# Combine features
X <- cbind(cluster_dummies, socio_dummies)
X <- as.matrix(X)

# Split X for train/test
X_train <- X[train_idx, ]
X_test <- X[-train_idx, ]

# ====== QUANTILE REGRESSION FOREST ======
# Approach: Fit separate models for treatment/control
# Fitting separate QRFs for treatment and control groups allows us to estimate the full conditional distribution for each group

# Control group QRF
control_idx <- T_train == 0
qrf_control <- quantregForest(
  x = X_train[control_idx, ],
  y = Y_train[control_idx],
  ntree = 5000,
  nodesize = 5,
  mtry = floor(sqrt(ncol(X_train)))
)

# Treatment group QRF
treatment_idx <- T_train == 1
qrf_treatment <- quantregForest(
  x = X_train[treatment_idx, ],
  y = Y_train[treatment_idx],
  ntree = 5000,
  nodesize = 5,
  mtry = floor(sqrt(ncol(X_train)))
)

# Define quantiles of interest
deciles <- seq(0.1, 0.9, by = 0.1)
customs <- c(0.2, 0.4, 0.6, 0.8)
quantiles <- deciles

# Function to calculate quantile treatment effects
calculate_qte <- function(X_pred, qrf_treat, qrf_control, tau) {
  # Predict quantiles for treatment and control
  pred_treat <- predict(qrf_treat, X_pred, what = tau)
  pred_control <- predict(qrf_control, X_pred, what = tau)
  
  # Quantile treatment effect
  qte <- pred_treat - pred_control
  return(qte)
}



# Calculate QTEs for different quantiles across all data
qte_results <- list()
for (tau in quantiles) {
  qte_results[[as.character(tau)]] <- calculate_qte(X, qrf_treatment, qrf_control, tau)
}



# Analyze QTEs by cluster
cluster_qte_summary <- data.frame()

for (cluster_id in unique(flash$Cluster)) {
  cluster_mask <- flash$Cluster == cluster_id
  
  for (tau in quantiles) {
    qte_cluster <- qte_results[[as.character(tau)]][cluster_mask]
    
    cluster_qte_summary <- rbind(cluster_qte_summary, data.frame(
      cluster = paste0("Cluster ", cluster_id),
      quantile = tau,
      mean_qte = mean(qte_cluster),
      sd_qte = sd(qte_cluster),
      n = sum(cluster_mask)
    ))
  }
}

# =============================================
# OVERALL QTE PLOT (Not split by cluster)
# =============================================

# Calculate overall QTEs
overall_qte <- data.frame()

for (tau in quantiles) {
  # Predict for all observations
  qte_all <- calculate_qte(X_test, qrf_treatment, qrf_control, tau)
  
  overall_qte <- rbind(overall_qte, data.frame(
    quantile = tau,
    mean_qte = mean(qte_all),
    sd_qte = sd(qte_all),
    n = length(qte_all)
  ))
}

# Plot overall QTE
p_overall <- ggplot(overall_qte, aes(x = factor(quantile), y = mean_qte)) +
  geom_line(aes(group = 1), size = 1.5, color = "darkblue") +
  geom_point(size = 3, color = "darkblue") +
  geom_ribbon(aes(ymin = mean_qte - 1.96*sd_qte/sqrt(n), 
                  ymax = mean_qte + 1.96*sd_qte/sqrt(n),
                  group = 1),
              alpha = 0.2, fill = "blue") +
  geom_hline(yintercept = 0, linetype = "dashed", alpha = 0.5) +
  ylim(-.05, .05) + 
  labs(
    title = "Overall Quantile Treatment Effects (Target: Absolute Change)",
    subtitle = paste(nrow(flash), "households"),
    x = "Quantile",
    y = "Treatment Effect (kWh)"
  ) +
  theme_minimal() +
  theme(
    plot.title = element_text(size = 14, face = "bold"),
    axis.text = element_text(size = 10)
  )

print(p_overall)


# Visualize QTE distributions by cluster
p_qte <- ggplot(cluster_qte_summary, aes(x = factor(quantile), y = mean_qte, 
                                         color = cluster, group = cluster)) +
  geom_line(size = 1.2) +
  geom_point(size = 3) +
  geom_hline(yintercept = 0, linetype = "dashed", alpha = 0.5) +
  ylim(-.05, .05) + 
  labs(
    title = "Quantile Treatment Effects by Cluster",
    x = "Quantile",
    y = "Treatment Effect (kWh)",
    color = "Cluster"
  ) +
  
  theme_minimal() +
  theme(
    plot.title = element_text(size = 14, face = "bold"),
    legend.position = "right"
  )

print(p_qte)

# ====== STANDARD CAUSAL FOREST / GENERALIZED RANDOM FOREST =======

# Fit Generalized Random Forest (GRF) Causal Forest
cf <- causal_forest(
  X = X,
  Y = Y,
  W = T,
  num.trees = 5000,
  min.node.size = 5,
  honesty = TRUE,
  seed = 1
)

# Get average treatment effects by cluster
ate_results <- data.frame()

for (cluster_id in unique(flash$Cluster)) {
  cluster_mask <- flash$Cluster == cluster_id
  cluster_X <- X[cluster_mask, ]
  
  # Predict treatment effects
  predictions <- predict(cf, cluster_X, estimate.variance = TRUE)
  cluster_effects <- predictions$predictions
  
  # Calculate confidence intervals (100 - alpha % CI)
  alpha <- 0.15
  z_score <- qnorm(1 - alpha/2)
  mean_effect <- mean(cluster_effects)
  mean_std_error <- sqrt(mean(predictions$variance.estimates))
  
  lower <- mean_effect - z_score * mean_std_error
  upper <- mean_effect + z_score * mean_std_error
  
  ate_results <- rbind(ate_results, data.frame(
    cluster = paste0("Cluster ", cluster_id),
    effect = mean_effect,
    lower = lower,
    upper = upper
  ))
}

# Sort results by effect size
ate_results <- ate_results |> arrange(effect)

# Visualize ATE comparison
p_ate <- ggplot(ate_results, aes(x = reorder(cluster, effect), y = effect)) +
  geom_bar(stat = "identity", fill = "orange", alpha = 0.8) +
  geom_errorbar(aes(ymin = lower, ymax = upper), width = 0.2) +
  geom_hline(yintercept = 0, color = "black", linetype = "solid", alpha = 0.3) +
  labs(
    title = "Average Treatment Effects by Cluster (with 85% CI)",
    y = "Treatment Effect (Relative Peak-hour Consumption Change)",
    x = ""
  ) +
  theme_minimal() +
  theme(
    plot.title = element_text(size = 14, face = "bold"),
    axis.text = element_text(size = 10),
    panel.grid.major.x = element_blank(),
    panel.grid.minor = element_blank()
  ) +
  coord_flip()


print(p_ate)


# =============================================
# HIGHLIGHT SPECIFIC CLUSTER
# =============================================

# Specify which cluster to highlight
highlight_cluster <- "Cluster 4"  # Change this to your cluster of interest

# Create a highlight variable
cluster_qte_summary <- cluster_qte_summary %>%
  mutate(
    highlighted = ifelse(cluster == highlight_cluster, "Highlighted", "Other"),
    alpha_value = ifelse(cluster == highlight_cluster, 1, 0.3)
  )

# Version 1: Using transparency
p_highlight_v1 <- ggplot(cluster_qte_summary, 
                         aes(x = factor(quantile), y = mean_qte, 
                             color = cluster, group = cluster)) +
  geom_line(aes(alpha = alpha_value), size = 1.2) +
  geom_point(aes(alpha = alpha_value), size = 2.5) +
  geom_hline(yintercept = 0, linetype = "dashed", alpha = 0.5) +
  scale_alpha_identity() +  # Use the alpha values as-is
  ylim(-.05,.05 ) + 
  labs(
    title = paste("Quantile Treatment Effects - Highlighting", highlight_cluster),
    x = "Quantile",
    y = "Treatment Effect (kWh)",
    color = "Cluster"
  ) +
  theme_minimal() +
  theme(plot.title = element_text(size = 14, face = "bold"))

print(p_highlight_v1)

# Version 2: Using size and custom colors
p_highlight_v2 <- ggplot(cluster_qte_summary, 
                         aes(x = factor(quantile), y = mean_qte, 
                             group = cluster)) +
  # Plot non-highlighted clusters first
  geom_line(data = filter(cluster_qte_summary, cluster != highlight_cluster),
            aes(color = cluster), size = 0.8, alpha = 0.4) +
  geom_point(data = filter(cluster_qte_summary, cluster != highlight_cluster),
             aes(color = cluster), size = 2, alpha = 0.4) +
  # Plot highlighted cluster on top
  geom_line(data = filter(cluster_qte_summary, cluster == highlight_cluster),
            color = "red", size = 2) +
  geom_point(data = filter(cluster_qte_summary, cluster == highlight_cluster),
             color = "red", size = 4) +
  # Add label for highlighted cluster
  geom_text(data = filter(cluster_qte_summary, 
                          cluster == highlight_cluster & quantile == 0.9),
            aes(label = cluster), 
            hjust = -0.1, color = "red", fontface = "bold") +
  geom_hline(yintercept = 0, linetype = "dashed", alpha = 0.5) +
  labs(
    title = paste("Quantile Treatment Effects - Highlighting", highlight_cluster),
    x = "Quantile",
    y = "Treatment Effect (kWh)"
  ) +
  theme_minimal() +
  theme(
    plot.title = element_text(size = 14, face = "bold"),
    legend.position = "none"  # Remove legend since we're highlighting
  )

print(p_highlight_v2)

# Version 3: Interactive highlighting with plotly
library(plotly)

p_interactive <- plot_ly(cluster_qte_summary, 
                         x = ~factor(quantile), 
                         y = ~mean_qte, 
                         color = ~cluster,
                         type = 'scatter', 
                         mode = 'lines+markers',
                         hovertemplate = paste(
                           '<b>%{fullData.name}</b><br>',
                           'Quantile: %{x}<br>',
                           'Effect: %{y:.3f} kWh<br>',
                           '<extra></extra>'
                         )) %>%
  layout(
    title = "Quantile Treatment Effects by Cluster (Interactive)",
    xaxis = list(title = "Quantile"),
    yaxis = list(title = "Treatment Effect (kWh)"),
    hovermode = 'closest'
  )

# This allows clicking on legend to show/hide clusters
p_interactive

# ====== HETEROGENEITY ANALYSIS WITH QRF =======

# Examine treatment effect heterogeneity at different quantiles
# For specific subgroups

# Income-based QTE analysis
income_qte_summary <- data.frame()

for (income_level in unique(flash$INCOME_CATEGORY)) {
  income_mask <- flash$INCOME_CATEGORY == income_level
  
  for (tau in quantiles) {
    qte_income <- qte_results[[as.character(tau)]][income_mask]
    
    income_qte_summary <- rbind(income_qte_summary, data.frame(
      income = income_level,
      quantile = tau,
      mean_qte = mean(qte_income),
      n = sum(income_mask)
    ))
  }
}

# Plot income-based QTE
p_income_qte <- ggplot(income_qte_summary |> 
                         filter(income %in% c("20,000-29,999", "50,000-74,999", "100,000+")),
                       aes(x = factor(quantile), y = mean_qte, 
                           color = income, group = income)) +
  geom_line(size = 1.2) +
  geom_point(size = 3) +
  geom_hline(yintercept = 0, linetype = "dashed", alpha = 0.5) +
  labs(
    title = "Quantile Treatment Effects by Income Level",
    x = "Quantile",
    y = "Treatment Effect (kWh)",
    color = "Income"
  ) +
  theme_minimal()

print(p_income_qte)

# ====== FEATURE IMPORTANCE FROM QRF =======

# Variable importance from QRF (average across both models)
importance_control <- importance(qrf_control)
importance_treatment <- importance(qrf_treatment)
avg_importance <- (importance_control + importance_treatment) / 2

qrf_importance_df <- data.frame(
  "Feature" = colnames(X),
  "QRF_Importance" = avg_importance) 

qrf_importance_df <- qrf_importance_df |>  arrange("Importance")
qrf_importance_df$IncNodePurity <- qrf_importance_df$IncNodePurity / sum(avg_importance)


print("Top 10 most important features from QRF:")
print(head(qrf_importance_df, 10))

# Compare with GRF importance
grf_importance <- variable_importance(cf)
grf_importance_df <- data.frame(
  "Feature" = colnames(X),
  "GRF_Importance" = grf_importance
)

# Merge importance measures
importance_comparison <- qrf_importance_df |>
  left_join(grf_importance_df, by = "Feature") |>
  rename("QRF_Importance" = "IncNodePurity")

print("Feature importance comparison (QRF vs GRF):")
print(importance_comparison)

# ====== ADDITIONAL ANALYSES =======

# Test for heterogeneous treatment effects using QRF
# Compare variance of QTEs across quantiles
qte_variance_by_cluster <- cluster_qte_summary |>
  group_by(cluster) |>
  summarise(
    qte_variance = var(mean_qte),
    qte_range = max(mean_qte) - min(mean_qte)
  ) |>
  arrange(desc(qte_variance))

print("Clusters with highest treatment effect heterogeneity:")
print(qte_variance_by_cluster)

# Save results
write.csv(cluster_qte_summary, "cluster_quantile_treatment_effects.csv", row.names = FALSE)
write.csv(ate_results, "cluster_average_treatment_effects.csv", row.names = FALSE)
write.csv(importance_comparison, "feature_importance_comparison.csv", row.names = FALSE)
