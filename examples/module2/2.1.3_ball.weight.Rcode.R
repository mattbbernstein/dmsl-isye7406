# R code for the ball weight example

# Read Data
## The response Y is the observed weight
## The X1 variable refers to whether ball A is in the observed weight or not
## The X2 variable refer to whether ball B counts in the observed weight or not. 
## 
Y  <- c(3,4,6,6); 
X1 <- c(1,0,1,1); 
X2 <- c(0,1,1,1);

# Linear Regression Model # Here "-1" means "no intercept"
mod1 <- lm( Y ~ X1 + X2 -1) 

## Check model coefficients 
## 1.667  2.667 
## These correspond to our estimated weights of ball A and B, resp.  
mod1;                          

# 70% Confidence Interval on Ball A weight 
#        fit       lwr     upr
# 1 1.666667 0.7414832 2.59185
# This means that 70% CI is [0.7415, 2.5919] if we keep 4 digits in our answers
predict(mod1, data.frame(X1=1, X2=0), interval="confidence", level = 0.7)

# 70% prediction interval on the weight of A+B  
#        fit     lwr      upr
#  1 4.333333 2.87049 5.796177
# This implieds the desird prediction interval is [2.8705, 5.7962]
#      
predict(mod1, data.frame(X1=1, X2=1), interval="prediction", level = 0.7)



