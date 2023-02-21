##  Linear Method for Classication 
##
## Example A: The Vowel data set from the textbook ElemStatLearn, also see the homepage
##
## https://statweb.stanford.edu/~tibs/ElemStatLearn/datasets/vowel.info.txt
##
##  I. Data Preparation  
## We assume that you save the data "vowel.train.csv" and "vowel.test.csv" 
##            in your local folder "C:/temp"
##  Ia. read the data
vowel.train <- read.table("C:/temp/vowel.train.csv",  header= TRUE, sep = ",");
vowel.test <- read.table("C:/temp/vowel.test.csv",  header= TRUE, sep = ",");

## The response values are {1,2,...,11}, denoting the 11 vowel 
## i (heed), I (hid), E (head), A (had), a: (hard), Y (hud), 
## O (hod), C: (hoard), U (hood), u: (who'd), 3: (heard) ###
##
## Ib:  We change the response Y values from {1,2,...11} to the category variable
##    by using the function "as.factor" 
## (otherwise the software might treat the response "1,..,11" as real values)
vowel.train$y <- as.factor(vowel.train$y); 

## II: We run four linear classification methods:
##  1. LDA            2. QDA
##  3. Naive Bayes    4. (multinomial) logisitic regression
## The placeholder for the training errors and testing errors
##      of these four methods 
TrainErr <- NULL;
TestErr  <- NULL; 

### Method 1: LDA
library(MASS)
# fit1 <- lda( y ~ ., data= vowel.train, CV= TRUE)
mod1 <- lda(vowel.train[,2:11], vowel.train[,1]); 

## training error 
## we provide a detailed code here 
pred1 <- predict(mod1,vowel.train[,2:11])$class; 
TrainErr <- c(TrainErr, mean( pred1  != vowel.train$y)); 
TrainErr; 
## 0.3162879 for miss.class.train.error
## testing error 
pred1test <- predict(mod1,vowel.test[,2:11])$class; 
TestErr <- c(TestErr,mean(pred1test != vowel.test$y));  
TestErr;
## 0.5562771 for miss.class.test.error

## This looks a large testing error, but note that we have 11 classes, 
##   and a random guess has a miclassification error of 10/11 = 91%!!!
## You can also see the details of Testing Error
##     by the confusion table, which shows how the errors occur
table(pred1test,  vowel.test$y) 


## Method 2: QDA
mod2 <- qda(vowel.train[,2:11], vowel.train[,1])
## Training Error 
pred2 <- predict(mod2,vowel.train[,2:11])$class
TrainErr <- c(TrainErr, mean( pred2!= vowel.train$y))
TrainErr
## 0.01136364 for miss.class.train.error of QDA,
##  which is much smaller than LDA
##  Testing Error 
TestErr <- c(TestErr, mean( predict(mod2,vowel.test[,2:11])$class != vowel.test$y))
TestErr
## 0.5281385 for miss.class.test.error

## Method 3: Naive Bayes
##  This has been implemented in the R library "e1071"
##  You need to first install this library 
##
library(e1071)
mod3 <- naiveBayes( vowel.train[,2:11], vowel.train[,1])
## Training Error
pred3 <- predict(mod3, vowel.train[,2:11]);
TrainErr <- c(TrainErr, mean( pred3 != vowel.train$y))
TrainErr 
##  0.2765152 for miss.class.train.error of Naive Bayes
## Testing Error 
TestErr <- c(TestErr,  mean( predict(mod3,vowel.test[,2:11]) != vowel.test$y))
TestErr
##  0.5324675 for miss.class.test.error of Naive Bayes 

### Method 4: (multinomial) logisitic regression) 
## You need to install the library "nnet" first 
## 
library(nnet)
mod4 <- multinom( y ~., data=vowel.train) 
summary(mod4);
## Training Error  of (multinomial) logisitic regression
TrainErr <- c(TrainErr, mean( predict(mod4, vowel.train[,2:11])  != vowel.train$y))
TrainErr
##  0.2215909 for miss.class.train.error
## Testing Error of (multinomial) logisitic regression
TestErr <- c(TestErr, mean( predict(mod4,vowel.test[,2:11]) != vowel.test$y) )
TestErr
##  0.512987 for miss.class.test.error of (multinomial) logisitic regression
## 
## For the vowel dataset, among all above 4 classification methods,
##  Multinomial logistic regression has the smallest mis-classification rate
##########



### More R codes for (binary) logisitic regression
###
### Example B: CHD vs Age 
###
data0 <- read.table("C:/temp/chdage.csv", head=T, sep=",")
attach(data0)
plot(Age, CHD)

## Both R code lead to the same results, the default link for binomial is "logit"
glm1 <- glm(CHD ~ Age , family = binomial(link="logit"), data = data0);  
glm1 <- glm(CHD ~ Age , family = binomial, data = data0); 

summary(glm1)
plot(Age, CHD)
lines(Age, fitted.values(glm1), col="red");

## when X is also binary
flag <- I(Age >=50); 
glm2 <- glm(CHD ~ flag, family = binomial(link="logit"), data = data0);
summary(glm2)

## Besides Binomial (and Gaussian/Normal), the "glm" can also deals with
##   other disributions such as Gamms, Inverse.Gaussian, Poisson, and 
##   quasi-distirbutions.  


##### Example C: low birth wight data 
##
## %<http://www.umass.edu/statdata/statdata/data/lowbwt.txt>
## 
## read data in
data1 <- read.table("C:/temp/lowbwt.csv", head=T, sep=",")

# take a look at first 3 rows of dataset
data1[1:3,]
## or look at the first several rows 
head(data1)

# fit a logit model with LOW as the dep. var. and AGE, LWT, and SMOKE
# as the covariates
#
logit.out <- glm(LOW~AGE+LWT+SMOKE, family=binomial(link=logit),
                 data=data1)

# take a look at the logit results
summary(logit.out)


# extract just the coefficients from the logit output object
coefficients(logit.out)
# put the logit coefficients in a new object called beta.logit
beta.logit <- coefficients(logit.out)

# plot low on AGE adding some jitter (noise) to LOW
plot(data1$AGE, jitter(data1$LOW, .1) )

# Now we're going to plot the predicted probabilities of a 120 lb. woman
# who is a smoker giving birth to a low birthweight child at different
# ages. First we need to construct a new matrix of covariate values that
# corresponds to our hypothetical women.
X1 <- cbind(1, seq(from=14, to=45, by=1), 120, 1)

# multiply this matrix by out logit coefficients to get the value of the
# linear predictor.
Xb1 <- X1 %*% beta.logit

# Now use the logistic cdf to transform the linear predictor into
# probabilities
prob1 <- exp(Xb1)/(1+exp(Xb1))

# now plot these probabilities as a function of age on the pre-existing
# graph of low on age
lines(seq(from=14, to=45, by=1), prob1, col="red")

# Now we're going to plot the predicted probabilities of a 120 lb. woman
# who is NOT a smoker giving birth to a low birthweight child at different
# ages. First we need to construct a new matrix of covariate values that
# corresponds to our hypothetical women.
X0 <- cbind(1, seq(from=14, to=45, by=1), 120, 0)

# multiply this matrix by out logit coefficients to get the value of the
# linear predictor.
Xb0 <- X0 %*% beta.logit

# Now use the logistic cdf to transform the linear predictor into
# probabilities
prob0 <- exp(Xb0)/(1+exp(Xb0))

# now plot these probabilities as a function of age on the pre-existing
# graph of low on age
lines(seq(from=14, to=45, by=1), prob0, col="blue")

# Now plot Confidence Interval Bands for probablities
# For woman who is a smoker
V <- vcov(logit.out)
Var.logit1 <- diag(X1 %*% V %*% t(X1))
Xb1.upper <- Xb1 + 1.96*sqrt(Var.logit1)
Xb1.low <- Xb1 - 1.96*sqrt(Var.logit1)

# Now use the logistic cdf to transform these CI into probabilities
prob1.upper <- exp(Xb1.upper)/(1+exp(Xb1.upper))
prob1.low <- exp(Xb1.low)/(1+exp(Xb1.low))

lines(seq(from=14, to=45, by=1), prob1.upper, col="red", lty="dashed")
lines(seq(from=14, to=45, by=1), prob1.low, col="red", lty="dashed")

# Similarly, for woman who is NOT a smoker
Var.logit0 <- diag(X0 %*% V %*% t(X0))
Xb0.upper <- Xb0 + 1.96*sqrt(Var.logit0)
Xb0.low <- Xb0 - 1.96*sqrt(Var.logit0)
prob0.upper <- exp(Xb0.upper)/(1+exp(Xb0.upper))
prob0.low <- exp(Xb0.low)/(1+exp(Xb0.low))
lines(seq(from=14, to=45, by=1), prob0.upper, col="blue", lty="dashed")
lines(seq(from=14, to=45, by=1), prob0.low, col="blue", lty="dashed")

# create a 3-d plot
weight <- seq(from=80, to=250, length=100)
age <- seq(from=14, to=45, length=100)
logit.prob.fun <- function(weight, age){
  exp(1.368225 -0.038995*age -0.012139*weight + 0.670764) /
    (1 + exp(1.368225 -0.038995*age -0.012139*weight + 0.670764))
}

prob <- outer(weight, age, logit.prob.fun)

persp(age, weight, prob, theta=30, phi=30, expand=0.5, col="lightblue")



# Hypothesis Testing in Logistic Regression
# fit a logit model

logit1.out <- glm(LOW~AGE+ LWT+SMOKE+HT+UI, family=binomial, data=data1)
summary(logit1.out)


# fit another logit model including race

data1$AfrAm <- data1$RACE==2
data1$othrace <- data1$RACE==3
logit2.out <- glm(LOW~AGE+ LWT+AfrAm+othrace+SMOKE+HT+UI,
                  family=binomial, data=data1)

summary(logit2.out)

# OK, let's conduct a likelihood ratio test of model 1 vs. model 2
# Here the constrained model is model 1 and the unconstrained model is
# model 2. Since 2 constraints are applied, the test statistic under
# the null follows a chi-square distribution with 2 degrees of freedom

lr <- deviance(logit1.out)  - deviance(logit2.out)
lr
# The p-value
1 - pchisq(lr, 2)

# The p-value of 0.01994 indicates that there is reason to believe
# (at 5% level) that the constraints implied by model 1 do not hold


# We can also use a Wald test to decide whether the
# coefficients on AfrAm and othrace are zero in the second model

R <- matrix(c(0,0,0,0,0,0,1,0,0,1,0,0,0,0,0,0), 2, 8)
beta <- coef(logit2.out)
r <- 0
V <- vcov(logit2.out)
W <- t(R %*% beta - r) %*% solve(R %*% V %*% t(R)) %*% (R %*% beta - r)
W
# The p-value from Wald test
1 - pchisq(W, 2)

# We got the same conclusion as in Likelihood test
# Why is the Wald statistic "only" 7.42, while the likelihood ratio
# statistic is 7.83 and both have the same df?
# ---- likelihood ratio test is more powerful


# We could also look at BIC to pick models. The AIC() function in R 
# will return BIC values if the argument k is set to log(n)

nrow(data1)
bic1 <- AIC(logit1.out, k=log(189))
bic2 <- AIC(logit2.out, k=log(189))
bic2 - bic1

# This indicates moderate support for model 1 over model 2. Nonetheless,
# given that we have strong reason to believe that race should be in the
# model we may well want to stick with model 2.


# Now suppose we want to test whether the coefficients on smoking
# and # hypertension are equal to each other in the second model.
# How to conduct a Wald test?

R <- matrix(c(0,0,0,0,0,1,-1,0), 1, 8)
beta <- coef(logit2.out)
r <- 0
V <- vcov(logit2.out)
W <- t(R %*% beta - r) %*% solve(R %*% V %*% t(R)) %*% (R %*% beta - r)
1 - pchisq(W, 1)

# the p-value of 0.293 suggests that there is no reason to believe
# the null hypothesis (that the coefficients are equal) is not true.

