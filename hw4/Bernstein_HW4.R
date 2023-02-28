pacman::p_load(tidyverse, ggplot2, mgcv, matrixStats, magrittr)

mexican_hat <- function(x) {
  y <- (1-(x^2))*exp(-0.5*(x^2))
  return(y)
}

smoothing_stats <- function(data, true) {
  m <- ncol(data)
  data$fm <- rowMeans(data)
  data$Bias <- data$fm - true
  data$Var <- rowVars(data, center = data$fm) * ((m-1)/m)
  data$MSE <- rowVars(data, center = true) * ((m-1)/m)
  return(data)
}

# Part 1 Values
# n <- 101
# i <- seq(from=1, to=n)
# x_i <- (2 * pi) * (-1 + (2 * ((i - 1) / (n - 1))))
# span <- 0.75
# bandwidth <- 0.2
# spar <- NULL
# title_prefix <- "Part 1: "

# Part 2 Values
set.seed(79)
x_i <- round(2*pi*sort(c(0.5, -1 + rbeta(50,2,2), rbeta(50,2,2))), 8)
span <- 0.3365
bandwidth <- 0.2
spar <- 0.7163
title_prefix <- "Part 2: "

fx_i <- mexican_hat(x_i)
trials <- 1000
loess_data <- data.frame(matrix(nrow=n, ncol=0))
kernel_data <- data.frame(matrix(nrow=n, ncol=0))
splines_data <- data.frame(matrix(nrow=n, ncol=0))

set.seed(7406)
for (i in 1:trials) {
  Y_i <- fx_i + rnorm(length(x_i), mean = 0, sd = 0.2)
  
  loess_model <- loess(Y_i ~ x_i, span = span)
  kernel_model <- ksmooth(x_i, Y_i, kernel = "normal", 
                          bandwidth = bandwidth, x.points = x_i)
  splines_model <- smooth.spline(Y_i ~ x_i, spar = spar)

  loess_data %<>% cbind(loess_model$fitted)
  kernel_data %<>% cbind(kernel_model$y)
  splines_data %<>% cbind(splines_model$y)
  
  # ggplot() + 
  #   geom_line(aes(x=x_i, y=Y_i), col='black') +
  #   geom_line(aes(x=x_i, y=loess_model$fitted), col='red') +
  #   geom_line(aes(x=x_i, y=kernel_model$y), col='blue') +
  #   geom_line(aes(x=x_i, y=splines_model$y), col='green')
}

cols <- paste0("D", 1:(ncol(loess_data)))
colnames(loess_data) <- cols
colnames(kernel_data) <- cols
colnames(splines_data) <- cols

loess_data %<>% smoothing_stats(fx_i)
kernel_data %<>% smoothing_stats(fx_i)
splines_data %<>% smoothing_stats(fx_i)

fm_data <- data.frame(x = x_i,
                      True = fx_i,
                      LOESS = loess_data$fm,
                      NW_Kernel = kernel_data$fm,
                      Splines = splines_data$fm)
bias_data <- data.frame(x = x_i,
                      LOESS = loess_data$Bias,
                      NW_Kernel = kernel_data$Bias,
                      Splines = splines_data$Bias)
var_data <- data.frame(x = x_i,
                        LOESS = loess_data$Var,
                        NW_Kernel = kernel_data$Var,
                        Splines = splines_data$Var)
mse_data <- data.frame(x = x_i,
                        LOESS = loess_data$MSE,
                        NW_Kernel = kernel_data$MSE,
                        Splines = splines_data$MSE)

fm_data %>% 
  pivot_longer(cols=-any_of("x"), names_to = "Smoother", values_to = "Value") %>%
  ggplot() + 
    geom_line(aes(x = x, y = Value, col = Smoother)) + 
    scale_color_manual(values=c("True" = "black", "LOESS" = "red",
                                "NW_Kernel" = "blue", "Splines" = "darkgreen")) +
    ggtitle(paste0(title_prefix,"Average Value of all Smoothers"))

bias_data %>% 
  pivot_longer(cols=-any_of("x"), names_to = "Smoother", values_to = "Value") %>%
  ggplot() + 
  geom_line(aes(x = x, y = Value, col = Smoother)) + 
  scale_color_manual(values=c("LOESS" = "red", "NW_Kernel" = "blue", 
                              "Splines" = "darkgreen")) +
  ylab("Bias") + ggtitle(paste0(title_prefix,"Average Bias of all Smoothers"))

var_data %>% 
  pivot_longer(cols=-any_of("x"), names_to = "Smoother", values_to = "Value") %>%
  ggplot() + 
  geom_line(aes(x = x, y = Value, col = Smoother)) + 
  scale_color_manual(values=c("LOESS" = "red", "NW_Kernel" = "blue", 
                              "Splines" = "darkgreen")) +
  ylab("Variance") + ggtitle(paste0(title_prefix,"Average Variance of all Smoothers"))

mse_data %>% 
  pivot_longer(cols=-any_of("x"), names_to = "Smoother", values_to = "Value") %>%
  ggplot() + 
  geom_line(aes(x = x, y = Value, col = Smoother)) + 
  scale_color_manual(values=c("LOESS" = "red", "NW_Kernel" = "blue", 
                              "Splines" = "darkgreen")) +
  ylab("MSE") + ggtitle(paste0(title_prefix,"Average MSE of all Smoothers"))
