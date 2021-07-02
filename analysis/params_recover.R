library(tidyverse)
library(cmdstanr)
library(rstan)
library(ggcolors)

call_pysim <- function(model, alpha, beta_, weight, filename) {
  pycmd <- paste("poetry run python", paste0("./cdrl/run_", model, ".py"))
  args <- paste("-a", alpha, "-b", beta_, "-w", weight, "-f", filename)
  cmd <- paste(pycmd, args)
  system(cmd)
  read.csv(paste("./data", filename, sep = "/"))
}

real_params <- expand.grid(alpha = seq(0.05, 0.25, 0.05),
                           beta = seq(1., 5., 1.),
                           weight = seq(1., 5., 1.))

heirarchical <- cmdstan_model("./analysis/heirarchical.stan")

params_comparison <- apply(real_params, 1, function(p) {
  real_alpha <- p["alpha"]
  real_beta <- p["beta"]
  real_weight <- p["weight"]

  result <- call_pysim("heirarchical",
                       real_alpha,
                       real_beta,
                       real_weight,
                       "sim_res.csv")

  fittable_data <- list(trial = nrow(result),
                        action = result$action + 1,
                        reward = result$reward)

  hvb <- heirarchical$variational(data = fittable_data,
                                  grad_samples = 5)

  est_alpha <- hvb$summary()[3, 2] %>% as.numeric
  est_beta <- hvb$summary()[4, 2] %>% as.numeric
  est_weight <- hvb$summary()[5, 2] %>% as.numeric

  data.frame(ra = real_alpha, rb = real_beta, rw = real_weight,
             ea = est_alpha, eb = est_beta, ew = est_weight)
}) %>%
  do.call(rbind, .)


write.csv(params_comparison, "./data/params_recover.csv", row.names = F)
params_comparison <- read.csv("./data/params_recover.csv")

fonts_config <- theme(axis.text = element_text(size = 15.),
                      axis.title = element_text(size = 20.),
                      strip.text = element_text(size = 20.),
                      legend.position = "none")


ggplot(data = params_comparison, aes(x = ra, y = ea)) +
  geom_boxplot(aes(color = "black", group = ra),
               fill = "transparent", size = 1) +
  geom_point(color = "white", size = 4) +
  geom_point(aes(color = "brightred"), size = 2) +
  geom_point(aes(y = ra), color = "white", size = 6) +
  geom_point(aes(y = ra, color = "red"), size = 4) +
  ylim(0, 1) +
  labs(x = "学習率(α)の真のパラメータ", y = "モデルによって推定されたパラメータ") +
  thanatos_light_color_with_name() +
  theme_bw() +
  fonts_config

ggsave("./param_recv_alpha.jpg", dpi = 300)

ggplot(data = params_comparison, aes(x = rb, y = eb)) +
  geom_boxplot(aes(color = "black", group = rb),
               fill = "transparent", size=1) +
  geom_point(color = "white", size = 4) +
  geom_point(aes(color = "brightred"), size = 2) +
  geom_point(aes(y = rb), color = "white", size = 6) +
  geom_point(aes(y = rb, color = "red"), size = 4) +
  ylim(0, 10) +
  labs(x = "逆温度(β)の真のパラメータ", y = "モデルによって推定されたパラメータ") +
  thanatos_light_color_with_name() +
  theme_bw() +
  fonts_config

ggsave("./param_recv_beta.jpg", dpi = 300)

ggplot(data = params_comparison, aes(x = rw, y = ew)) +
  geom_boxplot(aes(color = "black", group = rw),
               fill = "transparent", size=1) +
  geom_point(color = "white", size = 4) +
  geom_point(aes(color = "brightred"), size = 2) +
  geom_point(aes(y = rw), color = "white", size = 6) +
  geom_point(aes(y = rw, color = "red"), size = 4) +
  ylim(0, 10) +
  labs(x = "好奇心への重み(w)の真のパラメータ", y = "モデルによって推定されたパラメータ") +
  thanatos_light_color_with_name() +
  theme_bw() +
  fonts_config

ggsave("./param_recv_w.jpg", dpi = 300)
