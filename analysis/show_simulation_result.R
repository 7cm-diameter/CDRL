library(tidyverse)
library(gridExtra)
library(ggcolors)

result <- read.csv("./data/heirarchical_result.csv")
result$trial <- seq_len(nrow(result))

fonts_config <- theme(axis.text = element_text(size = 15.),
                      axis.title = element_text(size = 20.),
                      strip.text = element_text(size = 20.),
                      legend.position = "none")

LINE_SIZE <- 1.5
RED <- "red"
BLACK <- "black"

base_plot <- ggplot(data = result, aes(x = trial)) +
  geom_line(aes(y = lprob, color = BLACK),
            linetype = "dashed", size = 1) +
  geom_line(aes(y = rprob, color = RED),
            linetype = "dashed", size = 1) +
  thanatos_dark_color_with_name() +
  theme_bw() +
  fonts_config


q_plot <- base_plot +
  geom_line(aes(y = lq, color = BLACK), size = LINE_SIZE) +
  geom_line(aes(y = rq, color = RED), size = LINE_SIZE) +
  labs(y = "Q-value", x = NULL)

c_plot <- base_plot +
  geom_line(aes(y = lc, color = BLACK), size = 1.5) +
  geom_line(aes(y = rc, color = RED), size = 1.5) +
  labs(y = "Curiosity", x = NULL)

p_plot <- base_plot +
  geom_line(aes(y = lp, color = BLACK), size = 1.5) +
  geom_line(aes(y = rp, color = RED), size = 1.5) +
  labs(y = "Probability", x = "Trial")

plot_sim_result <- grid.arrange(q_plot, c_plot, p_plot)
ggsave("./sim_result.jpg", plot_sim_result, dpi = 300)
