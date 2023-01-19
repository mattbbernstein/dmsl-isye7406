
## R code for KNN
##

## Assume you save the dataset mixtureexample.csv in the folder "C:/temp"
# Read the data
mixture.example <- read.table(file = "C:/temp/mixtureexample.csv", header= TRUE, sep = ",");

## Look at the six rows of data: (1,2,3) and (198, 199, 200)
mixture.example[c(1:3,198:200),]

x1  <- mixture.example[, 1];
x2  <- mixture.example[, 2];
y1  <- mixture.example[, 3];

# Plot Fig 2.1:
plot(x1, x2, col=ifelse(y1==1, "red", "blue"), xlab="", ylab="", pch=21, bg=ifelse(y1==1, "red", "blue"));


# Linear regression for classification
method1 <- lm( y1 ~ x1 + x2);
b0 <- coef(method1)[1]; 
b1 <- coef(method1)[2]; 
b2 <- coef(method1)[3];

## This gives a fitted model of 
##  Yhat = b0 + b1* X1 + b2*X2 
## Now we want to plot the three-dimensional data (Y, X1, X2)
##  into the two-dimensional space (X1, X2). 
## Recall that we classify the response to 0 or 1,
##  depending on whether Yhat >= 0.5 
##  This translate to whether b0 + b1 *X1 + b2* X2 >= 0.5.
##   which becomes the following boundary line in (x1, x2) plane 
##   x2 >= (0.5 - b0)/ b2 - (b1/ b2) X1
intercept1 <- (0.5 - b0) / b2;
slope1 <- - b1 / b2;

## plot the 
abline(intercept1, slope1, lwd=3);

#
# KNN
# generate xnew=(x1new, x2new) data set over the grid of (X1, X2) plane

c(min(x1), max(x1));
c(min(x2), max(x2));
px1 <-  seq(-2.6, 4.2, 0.1);
px2 <-  seq(-2, 2.9, 0.05);

## The data xnew1 was generated in blocks, and
##   each block has length of length(px1) with 
##   the same x2[i] values and different x1 values. 
##  In different blocks, we have different x2 values


xnew1 <- NULL;
for (i in 1:length(px2)) xnew1 <- rbind(xnew1, cbind(px1, px2[i]));

## check 
xnew1[1:10,] 


### Plot the original data set 
plot(x1, x2, col=ifelse(y1==1, "red", "blue"), xlab="", ylab="", pch=21, bg=ifelse(y1==1, "red", "blue"));

### KNN 
library(class)
## knn(train, test, yclass, k value, l = 0, prob = FALSE, use.all = TRUE)

method2 <- knn(cbind(x1, x2), xnew1, y1, k=15, prob=TRUE);
## prob1 returns the proportion of the votes for the winning class are returned
##   min(prob1) = 0.53333
prob1 <- attr(method2, "prob");
## we do not know "prob1" refers to which class, 
## and thus we define prob2 to return the proportion of the votes for one class 
prob2 <- ifelse(method2=="1", 1- prob1, prob1);
## create matrix form of the "xnew1" data.frame
## that will allow us to plot the contour 
prob3 <- matrix(prob2, length(px1), length(px2));

contour(px1, px2, prob3, level=0.5, labels="", xlab="", ylab="", main="15-NN", lwd=3);
points(x1, x2, col=ifelse(y1==1, "red", "blue"), pch=21, bg=ifelse(y1==1, "red", "blue"));


### A simpler way for Training/Testing error
###  below we define "xnew1" and "ytrue" from the training data set
###       but they can also be defined from the testing data
xnew1   <- mixture.example[, 1:2];
ytrue   <- mixture.example[, 3];

## predicted "Y" class for the dataset "xnew1"
ypred  <- knn(mixture.example[, 1:2], xnew1, mixture.example[, 3], k=15);

## compute the mis-classification error of the data set of "xnew1" and "ytrue."
mean( ypred != ytrue) 
##  0.155 for miss.class.train.error
