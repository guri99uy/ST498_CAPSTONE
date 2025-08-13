# load libraries
library(tidyverse)
library(grf)  # causal forests
library(quantregForest)  # quantile regression forests


# ====== READ DATA ======

##  socio
socio_csv_path <- "/Users/finbarrhodes/Documents/Github/ST498_CAPSTONE/FLASH/toShare/socioEcodata.csv"
df_socio <- read_csv(socio_csv_path)

## new setup 
all_methods_flash <- read_csv("/Users/finbarrhodes/Documents/GitHub/ST498_CAPSTONE/Notebooks/comp_socio_df_02.csv")
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

# check and omit NAs
flash <- na.omit(flash)


# old section for changing hash keys to integer IDs for readability; not totally necessary
"
# Create IDs
df_socio <- df_socio |>
  mutate(ID = row_number())

# Create ID dictionary
id_dict <- df_socio |>
  select(HASH_KEY, ID) |>
  deframe()

# Map ANON_ID using id_dict
flash <- flash |> mutate(ANON_ID = id_dict[ANON_ID])
"


# Ensure Cluster label is of type factor
flash <- flash |> mutate(Cluster = as.factor(as.integer(Cluster)))



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
X <- cluster_dummies
# cbind(cluster_dummies, socio_dummies)
X <- as.matrix(X)

# Split X for train/test
X_train <- X[train_idx, ]
X_test <- X[-train_idx, ]



# ====== CAUSAL FOREST / GENERALIZED RANDOM FOREST =======

# fit
cf <- causal_forest(
  X = X,
  Y = Y,
  W = T,
  num.trees = 5000,
  min.node.size = 5,
  honesty = TRUE,
  seed = 1
)

# ate by cluster
ate_results <- data.frame()
for (cluster_id in unique(flash$Cluster)) {
  cluster_mask <- flash$Cluster == cluster_id
  cluster_X <- X[cluster_mask, ]
  
  # predict treatment effects
  predictions <- predict(cf, cluster_X, estimate.variance = TRUE)
  cluster_effects <- predictions$predictions
  
  # calculate confidence intervals (100 - alpha % CI)
  alpha <- 0.25
  z_score <- qnorm(1 - alpha/2)
  mean_effect <- mean(cluster_effects)
  mean_std_error <- sqrt(mean(predictions$variance.estimates))
  
  lower <- mean_effect - z_score * mean_std_error
  upper <- mean_effect + z_score * mean_std_error
  
  ate_results <- rbind(ate_results, data.frame(
    cluster = paste0("Cluster ", cluster_id),
    effect = mean_effect,
    lower = lower,
    upper = upper,
    type = "Cluster"
  ))
}

# calculating overall ATE to compare with cluster values
all_predictions <- predict(cf, X, estimate.variance = TRUE)
overall_mean <- mean(all_predictions$predictions)
overall_std_error <- sqrt(mean(all_predictions$variance.estimates))
overall_lower <- overall_mean - z_score * overall_std_error
overall_upper <- overall_mean + z_score * overall_std_error

# add overall average to results
ate_results <- rbind(ate_results, data.frame(
  cluster = "Overall Average",
  effect = overall_mean,
  lower = overall_lower,
  upper = overall_upper,
  type = "Overall"
))

# sorting results to visualization
cluster_data <- ate_results |> arrange(effect)

# visualize ATE comparison
p_ate <- ggplot(cluster_data, aes(x = reorder(cluster, effect), y = effect, fill = type)) +
  geom_bar(stat = "identity", alpha = 0.8) +
  geom_errorbar(aes(ymin = lower, ymax = upper), width = 0.2) +
  geom_hline(yintercept = 0, color = "black", linetype = "solid", alpha = 0.3) +
  scale_fill_manual(values = c("Cluster" = "orange", "Overall" = "darkgrey")) +
  labs(
    y = "Treatment Effect (Peak-hour Consumption Change in kWh)",
    x = ""
  ) +
  theme_bw() +
  theme(
    plot.title = element_text(size = 14, face = "bold"),
    axis.text = element_text(size = 10),
    panel.grid.major.x = element_blank(),
    panel.grid.minor = element_blank(),
    legend.position = 'none'
  ) +
  coord_flip()

print(p_ate)





# ====== QUANTILE REGRESSION FOREST ======
# Approach: Fit separate models for treatment/control
# fitting separate QRFs for treatment and control groups allows us to estimate the full conditional distribution for each group

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



# Function to calculate quantile treatment effects (QTE): 
# QTE estimation is taken as the difference between treatment and control estimates at a given quantile.
calculate_qte <- function(X_pred, qrf_treat, qrf_control, tau) {
  
  # Get quantile values for treatment and control separately
  pred_treat <- predict(qrf_treat, X_pred, what = tau)
  pred_control <- predict(qrf_control, X_pred, what = tau)
  
  # Take the difference to get the QTE at a given tau
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


# OVERALL QTE PLOT (Not split by cluster)

# calculate overall QTEs
overall_qte <- data.frame()
for (tau in quantiles) {
  # predict for all observations
  qte_all <- calculate_qte(X_test, qrf_treatment, qrf_control, tau)
  
  overall_qte <- rbind(overall_qte, data.frame(
    quantile = tau,
    mean_qte = mean(qte_all),
    sd_qte = sd(qte_all),
    n = length(qte_all)
  ))
}

# overall qte plot
p_overall <- ggplot(overall_qte, aes(x = factor(quantile), y = mean_qte)) +
  geom_line(aes(group = 1), size = 1.5, color = "darkblue") +
  geom_point(size = 3, color = "darkblue") +
  geom_ribbon(aes(ymin = mean_qte - 1.96*sd_qte/sqrt(n), 
                  ymax = mean_qte + 1.96*sd_qte/sqrt(n),
                  group = 1),
              alpha = 0.2, fill = "blue") +
  geom_hline(yintercept = 0, linetype = "dashed", alpha = 0.5) +
  ylim(-.074, .08) + 
  labs(
    x = "Quantile",
    y = "Treatment Effect (kWh)"
  ) +
  theme_bw() +
  theme(
    plot.title = element_text(size = 14, face = "bold"),
    axis.text = element_text(size = 10)
  )

print(p_overall)


# separating qte by cluster
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

# highlighting cluster 4
highlight_cluster <- "Cluster 4"  # stay-at-home consumer

# creating a highlight variable
cluster_qte_summary <- cluster_qte_summary %>%
  mutate(
    highlighted = ifelse(cluster == highlight_cluster, "Highlighted", "Other"),
    alpha_value = ifelse(cluster == highlight_cluster, 1, 0.33)
  )

p_highlight <- ggplot(cluster_qte_summary, 
                         aes(x = factor(quantile), y = mean_qte, 
                             color = cluster, group = cluster)) +
  geom_line(aes(alpha = alpha_value), size = 1.2) +
  geom_point(aes(alpha = alpha_value), size = 2.5) +
  geom_hline(yintercept = 0, linetype = "dashed", alpha = 0.5) +
  scale_alpha_identity() +  # Use the alpha values as-is
  ylim(-.074,.08 ) + 
  labs(
    x = "Quantile",
    y = "Treatment Effect (kWh)",
    color = "Cluster"
  ) +
  theme_bw() +
  theme(plot.title = element_text(size = 14, face = "bold"))

print(p_highlight)





