mexican_hat <- function(x) {
  y <- (1-(x^2))*exp(-0.5*(x^2))
  return(y)
}


n <- 100
i <- seq(from=1, to=n)
x_i <- (2 * pi) * (-1 + (2 * ((i - 1) / (n - 1))))
fx_i <- mexican_hat(x_i)

sigma_i <- rnorm(x_i, mean = 0, sd = (0.2^2))

data <- data.frame(Y_i = fx_i+sigma_i, x_i = x_i)  
loess_model <- loess(Y_i ~ x_i, data=data, span = 0.75, normalize=FALSE)
kernel_model <- ksmooth(x_i, data$Y_i, "normal", bandwidth = 0.2)
splines_model <- gam(Y_i ~ s(x_i), data = data)
