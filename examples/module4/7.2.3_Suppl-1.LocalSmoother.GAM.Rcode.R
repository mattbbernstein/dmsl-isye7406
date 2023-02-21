
### 1. Local Smoothing

## First, simulate a dataset from a given model 
set.seed(123);
x <- seq(0, 10, 0.1);
y <-sin(x) + rnorm(length(x)); 

plot(x,y, pch=19);

## Second, we use three local smoothing to fit the data
##
## A. LOWESS vs LOESS 
##   LOESS is more modern/complciated but slower
##   they are similar for one-dim responses 
##      with different default tuning para. values

## A(i) LOWESS (this implements the method of Cleveland's 1979 JASA paper) 
lines(lowess(x,y), col="red", lwd=3);

## A(ii) LOESS (generalization with multiple predictors)
fitlo2 <- loess(y ~ x)
lines(x, fitlo2$fit, col="black", lwd=3);

## A(iii) one important tuning parameter is "span" that controls of degree of smoothing
##    maybe choose "span" via cross-validation
fitlo2b <- loess(y ~ x, span=0.4)
lines(x, fitlo2b$fit, col="purple", lwd=3);


## B.   Nadaraya-Watson kernel smoothing
##  compare a smaller (2) vs. a larger (10) bandwidth 
##   A larger bandwith produces a smoother curve

plot(x,y, pch=19);
## B(i) Nadaraya-Watson kernel smoothing with bandwith = 2
##      here we use the Gaussian Kernel, i.e., "normal"
fitkern2 <- ksmooth(x, y, "normal", bandwidth = 2);
lines(fitkern2, col = "green", lwd=3);
## B(ii) Nadaraya-Watson kernel smoothing with bandwith = 8
fitkern8 <- ksmooth(x, y, "normal", bandwidth = 8);
lines(fitkern8, col = "black", lwd=3);

## C. spline Smoothing 
## compare the tuning parameter of spar
plot(x,y, pch=19);
fitspl <- smooth.spline(x, y, spar=0.2)
lines(fitspl, col = "blue", lwd=3)
fitsp9 <- smooth.spline(x, y, spar=0.9)
lines(fitsp9, col = "black", lwd=3)


## Some R Code to choose "spar" via Leave-One-Out Cross Validation 
## when the data set is (x,y) as in our example 
## This R code is not efficient, but hopefully it is easy to understand.  
##
spars <- seq(0, 2, by = 0.01)
RES <- NULL;
## For each "spar" parameter
for (i in 1:length(spars)){
  ## Create a varaible to save the leave-one-out prediction value
  ypred1 <- rep(0, length(y));
  for (j in 1: length(x)){
    ## using the remaining n-1 data to build a spline smoothing model
    tempmod <- smooth.spline(x[-j], y[-j], spar = spars[i]);
    ## the predicted Y value for leave one out 
    ypred1[j] <- predict(tempmod, x[j])$y; 
  }
  ## For a given "spar", return the leave-one-out CV value
  RES <- c(RES, sum((ypred1-y)^2));
}

plot(spars, RES, 'l', xlab = 'spar', ylab = 'Cross Validation Residual Sum of Squares' , main = 'CV RSS vs Spar')
## the "spar" that minimizes the leave one out CV
sparopt = spars[which.min(RES)];
sparopt
## 0.73 in my laptop 


## 2. The backfitting algorithm 
##  is used to solve the equations
##  2*X1 + X2 = 1 and X1+ 2*X2 = 2
##

x1 <- 0;
x2 <- 0;
data <- c(0, x1, x2);
for (i in 1:40) {
  x1new <- (1- x2)/2;
  x2new <- (2-x1)/2;
  x1 <- x1new;
  x2 <- x2new;
  data <- rbind(data, c(i, x1, x2));
}
colnames(data) <- c("round", "x1", "x2"); 
plot(data[,1], data[,2], "l", ylim=c(0,1.1), lwd=3, xlab="round", ylab="X values", main="Black (X1) and Blue (X2)");
lines(data[,1], data[,3], lwd=3, col="blue");
round(data[1:20,],4)


## Another version that can converge faster. 
x1 <- 0;
x2 <- 0;
data <- c(0, x1, x2);
for (i in 1:40) {
  x1 <- (1- x2)/2;
  x2 <- (2-x1)/2;
  data <- rbind(data, c(i, x1, x2));
}
colnames(data) <- c("round", "x1", "x2"); 
plot(data[,1], data[,2], "l", ylim=c(0,1.1), lwd=3, xlab="round", ylab="X values", main="Black (X1) and Blue (X2)");
lines(data[,1], data[,3], lwd=3, col="blue");
round(data[1:20,],4)



# 3. Generalized Additive Models (GAM) 
#http://archive.ics.uci.edu/ml/machine-learning-databases/spambase/
## YOu can write the data directly from online
spam <- read.table(file= "http://archive.ics.uci.edu/ml/machine-learning-databases/spambase/spambase.data", sep = ",")

### Or you can download the data file "spambase.csv" from Canvas to your local laptop 
#spam <- read.table(file= "C:/temp/spambase.csv", sep = ",")

dim(spam)
set.seed(123)
flag <- sort(sample(4601,1536, replace = FALSE))
spamtrain <- spam[-flag,]
spamtest <- spam[flag,]
dim(spamtest[spamtest$V58 ==1,])

### 3A. Logistic Regression model by glm fucntion ###
fit1 <- glm( V58  ~ ., family = binomial, data= spamtrain)

#### it shows the following warning message. What does it mean?
####   we can still do prediction, though more in-depth investigation is needed
#Warning message:
#glm.fit: fitted probabilities numerically 0 or 1 occurred 

#predict(fit1,spamtrain,type='response')


## Trainign error 
fit1;
y1hat <- ifelse(fit1$fitted.values<0.5,0,1)
y1    <- spamtrain$V58;
sum(y1hat != y1)/length(y1)
# 0.06868072 (train error for logistic regression)

p2hat <- predict(fit1,newdata=spamtest[,-58],type="response")
y2hat <- ifelse(p2hat <0.5,0,1)
y2    <- spamtest$V58;
sum(y2hat != y2)/length(y2)
## 0.06835938 (test error)


#Some significant variables
summary(fit1)
# V5: word_freq_our
# V6: word_freq_over
# V7: word_freq_remove
# V8: word_freq_internet
# V16: word_freq_free
# V17: word_freq_business
# V52: word_freq_!
# V53: word_freq_$
# V56: capital_run_length_longest
# V57: capital_run_length_total 


## 3B variable/model selection in logisitc regression 
## Use "step" function to choose a sub-model of logistic regression 
##   by AIC in a stepwise algorithm

## Thie might take a long while to get the result
fit1step <- step(fit1);
## see the final selected model by AIC
fit1step; 

### Also see which X variances are removed 
fit1step$anova
fit1stepa <- glm( V58 ~ . -V37 - V13-V34-V31-V55-V50-V32-V11-V40-V18-V14-V51-V3-V30,family = binomial, data= spamtrain)


## 3C: GAM model
## YOu need to first install the package, "gam" or "gmcv"
##use GAM, try one of the following libraries "gam" or "gmcv" ###
library(gam)
#another libary that include the "gam" function is "mgcv", but with slighlty different way, 
##    and you can google "mgcv' to learn more
##

fit2 <- gam( V58 ~ ., family = binomial(link = "logit"), data= spamtrain, trace=TRUE)
## example, test error
p2hata <-  predict(fit2, spamtest[,-58],type="response")
y2hata <- ifelse(p2hata <0.5,0,1)
sum(y2hata != y2)/length(y2)
## 0.06835938 (test error)

# or use spline
fit3 <- gam( V58 ~ . + s(V5) + s(V6) + s(V7) + s(V8) + s(V16) + s(V17) + s(V52) + s(V53) + s(V56) + s(V57), family = binomial, data= spamtrain, trace=TRUE)

## Training & Test Error 
y1hatb <- ifelse(fit3$fitted.values<0.5,0,1)
sum(y1hatb != y1)/length(y1)
#  0.05303195 (train error vs 0.06868072 for logistic regression)
## example, test error
p2hatb <-  predict(fit3, spamtest[,-58],type="response")
y2hatb <- ifelse(p2hatb <0.5,0,1)
sum(y2hatb != y2)/length(y2)
## 0.05013021 (test error)


##Unfortunately s(.)  will not work
## fit4 <- gam( V58 ~ s(.), family = binomial(link = "logit"), data= spamtrain)
##
### below is the R code suggested by Dr. Ethan Mark, a former ISyE PhD student 
###               when he took ISyE 7406
### For instance, if we want to fit the model
##   V58 ~ beta1*V1 + ...+ \beta10*V10+ s(V11)+...+s(V57)
##  where the first 10 terms are parametriclinear regression, and 
##        and the last 47 terms are nonparametric splines
##
parametric_terms<-c(1:10) #variable numbers we want to fit parametrically
## the corrresponding variable names for linear regression
xname1 <- paste0("V",parametric_terms,collapse="+");  

smooth_terms <-c(11:57)   #variable numbers we want to use splines for
## the corrresponding variable names for splines
xname2 <-  paste0("s(V",smooth_terms,")",collapse="+"); 

## the fitted formula 
form <- as.formula(paste0("V58~",xname1,"+",xname2,collapse=""))
## 
fit4  <- gam(formula=form,family = binomial(link = "logit"), data= spamtrain);
## testing error of this new model
p2hatd <-  predict(fit4, spamtest[,-58],type="response")
y2hatd <- ifelse(p2hatd <0.5,0,1)
sum(y2hatd != y2)/length(y2)
## 0.03841146 (test error)
