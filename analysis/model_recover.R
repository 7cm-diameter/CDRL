library(tidyverse)
library(cmdstanr)
library(rstan)
# remotes::install_github("7cm-diameter/ggcolors")
library(ggcolors)

calc_WBIC <- function(vbfit) {
  read_stan_csv(vbfit$output_files()) %>%
    extract %>%
    (function(e) e$ll) %>%
    rowSums %>%
    mean %>%
    (function(x) -x)
}

call_pysim <- function(model, alpha, beta_, weight, filename) {
  pycmd <- paste("poetry run python", paste0("./cdrl/run_", model, ".py"))
  args <- paste("-a", alpha, "-b", beta_, "-w", weight, "-f", filename)
  cmd <- paste(pycmd, args)
  system(cmd)
  read.csv(paste("./data", filename, sep = "/"))
}

model_recover <- function(model, alpha, beta_, weight) {
  result <- call_pysim(model, alpha, beta_, weight, "sim_res.csv")

  fittable_data <- list(trial = nrow(result),
                        action = result$action + 1,
                        reward = result$reward)

  hvb <- heirarchical$variational(data = fittable_data,
                                  grad_samples = 5)
  qvb <- qlearning$variational(data = fittable_data,
                               grad_samples = 5)

  hwbic <- calc_WBIC(hvb)
  qwbic <- calc_WBIC(qvb)

  return(qwbic < hwbic)
}

heirarchical <- cmdstan_model("./analysis/heirarchical.stan")
qlearning <- cmdstan_model("./analysis/qlearning.stan")

model_comparison <- (function(n) {
  seq_len(n) %>%
    lapply(., function(i) {
      alpha <- runif(1, 0.05, 0.25)
      beta_ <- runif(1, 1., 4.)
      weight <- runif(1, 1., 4.)

      qrcv <- model_recover("qlearning", alpha, beta_, weight)
      hrcv <- !model_recover("heirarchical", alpha, beta_, weight)
      c(qrcv, hrcv)
  })
})(100)

recover_result <- model_comparison %>%
  unlist %>%
  (function(x) {
    n <- length(x) / 2
    q_corr <- x[seq_len(length(x)) %% 2 == 0] %>% sum
    h_corr <- x[seq_len(length(x)) %% 2 == 1] %>% sum
    q_fa <- n - q_corr
    h_fa <- n - h_corr
    models <- c("Q-Learning", "Heirarchical-Q-Learning")
    data.frame(true = rep(models, 2),
               selected = rep(models, each = 2),
               prob = c(q_corr, q_fa, h_fa, h_corr) / n)
})

write.csv(recover_result, "./data/model_recover.csv", row.names = F)
recover_result <- read.csv("./data/model_recover.csv")


fonts_config <- theme(axis.text = element_text(size = 15.),
                      axis.title = element_text(size = 20.),
                      strip.text = element_text(size = 20.),
                      legend.position = "none")

ggplot(data = recover_result, aes(x = true, y = selected)) +
  geom_tile(aes(fill = prob)) +
  geom_text(aes(label = round(prob, digits=2)), color = "black", size = 7.5) +
  scale_fill_gradient(low = "#faefed", high = "#d75170") +
  labs(y = "真のモデル", x = "WBICによって選択したモデル") +
  theme_bw() +
  fonts_config

ggsave("./model_recover.jpg", dpi = 300)
