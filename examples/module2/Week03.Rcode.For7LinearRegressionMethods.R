####
#### R code for Week #3: Linear Regression 
## 
## First, save the dataset "prostate.csv" in your laptop, say, 
##         in the local folder "C:/temp". 
##
## Data Set
prostate <- read.table("C:/temp/prostate.csv", header= TRUE, sep = ",")

##This dataset is from the textbook ESL, where the authors 
##  have split the data into the training and testing subsets
##  here we use their split to produce similar results 

training <- subset( prostate, train == TRUE)[,1:9];
test     <- subset( prostate, train == FALSE)[,1:9];
## The true Y response for the testing subset
ytrue    <- test$lpsa; 



### Below we consider seven (7) linear regression related models
### (1) Full model;  (2) The best subset model; (3) Stepwise variable selection with AIC
### (4) Ridge Regression; (5) LASSO; 
##  (6) Principal Component Regression, and (7) Parital Least Squares (PLS) Regression 
##
###   For each of these 7 models or methods, we fit to the training subset, 
###    and then compute its training and testing errors. 
##
##     Let us prepare to save all training and testing errors
MSEtrain <- NULL;
MSEtest  <- NULL; 

###
### (1) Linear regression with all predictors (Full Model)
###     This fits a full linear regression model on the training data
model1 <- lm( lpsa ~ ., data = training); 

## Model 1: Training error
MSEmod1train <-   mean( (resid(model1) )^2);
MSEtrain <- c(MSEtrain, MSEmod1train);
# Model 1: testing error 
pred1a <- predict(model1, test[,1:8]);
MSEmod1test <-   mean((pred1a - ytrue)^2);
MSEmod1test;
MSEtest <- c(MSEtest, MSEmod1test); 
#[1] 0.521274

### (2) Linear regression with the best subset model 
###  YOu need to first install the package "leaps"
library(leaps);
prostate.leaps <- regsubsets(lpsa ~ ., data= training, nbest= 100, really.big= TRUE); 

## Record useful information from the output
prostate.models <- summary(prostate.leaps)$which;
prostate.models.size <- as.numeric(attr(prostate.models, "dimnames")[[1]]);
prostate.models.rss <- summary(prostate.leaps)$rss;

## 2A:  The following are to show the plots of all subset models 
##   and the best subset model for each subset size k 
plot(prostate.models.size, prostate.models.rss); 
## find the smallest RSS values for each subset size 
prostate.models.best.rss <- tapply(prostate.models.rss, prostate.models.size, min); 
## Also add the results for the only intercept model
prostate.model0 <- lm( lpsa ~ 1, data = training); 
prostate.models.best.rss <- c( sum(resid(prostate.model0)^2), prostate.models.best.rss); 
## plot all RSS for all subset models and highlight the smallest values 
plot( 0:8, prostate.models.best.rss, type = "b", col= "red", xlab="Subset Size k", ylab="Residual Sum-of-Square")
points(prostate.models.size, prostate.models.rss)

# 2B: What is the best subset with k=3
op2 <- which(prostate.models.size == 3); 
flag2 <- op2[which.min(prostate.models.rss[op2])]; 

## There are two ways to fit this best subset model with k=3. 

## 2B(i) First, we can manual look at the selected model and fit it.
##      It will not be able to be implemented in cross-validation 
prostate.models[flag2,]
model2a <- lm( lpsa ~ lcavol + lweight + svi, data = training);
summary(model2a);

## 2B(ii) Second, we can auto-find the best subset with k=3
##   this way will be useful when doing cross-validation 
mod2selectedmodel <- prostate.models[flag2,]; 
mod2Xname <- paste(names(mod2selectedmodel)[mod2selectedmodel][-1], collapse="+"); 
mod2form <- paste ("lpsa ~", mod2Xname);
## To auto-fit the best subset model with k=3 to the data
model2 <- lm( as.formula(mod2form), data= training); 
# Model 2: training error 
MSEmod2train <- mean(resid(model2)^2);
## save this training error to the overall training error vector 
MSEtrain <- c(MSEtrain, MSEmod2train);
MSEtrain;
## Model 2:  testing error 
pred2 <- predict(model2, test[,1:8]);
MSEmod2test <-   mean((pred2 - ytrue)^2);
MSEtest <- c(MSEtest, MSEmod2test);
MSEtest;
## Check the answer
##[1] 0.5212740 0.4005308

## As compared to the full model #1, the best subset model with K=3
##   has a larger training eror (0.521 vs 0.439),
##   but has a smaller testing error (0.400 vs 0.521). 


### (3) Linear regression with the stepwise variable selection 
###     that minimizes the AIC criterion 
##    This can done by using the "step()" function in R, 
##       but we need to build the full model first

model1 <- lm( lpsa ~ ., data = training); 
model3  <- step(model1); 

## If you want, you can see the coefficents of model3
round(coef(model3),3)
summary(model3)

## Model 3: training  and  testing errors 
MSEmod3train <- mean(resid(model3)^2);
pred3 <- predict(model3, test[,1:8]);
MSEmod3test <-   mean((pred3 - ytrue)^2);
MSEtrain <- c(MSEtrain, MSEmod3train);
MSEtrain; 
## [1] 0.4391998 0.5210112 0.4393627
MSEtest <- c(MSEtest, MSEmod3test);
## Check your answer 
MSEtest;
## [1] 0.5212740 0.4005308 0.5165135


### (4) Ridge regression (MASS: lm.ridge, mda: gen.ridge)
### We need to call the "MASS" library in R
### 
library(MASS);

## The following R code gives the ridge regression for all penality function lamdba
##  Note that you can change lambda value to other different range/stepwise 
prostate.ridge <- lm.ridge( lpsa ~ ., data = training, lambda= seq(0,100,0.001));

## 4A. Ridge Regression plot how the \beta coefficients change with \lambda values 
##   Two equivalent ways to plot
plot(prostate.ridge) 
### Or "matplot" to plot the columns of one matrix against the columns of another
matplot(prostate.ridge$lambda, t(prostate.ridge$coef), type="l", lty=1, 
        xlab=expression(lambda), ylab=expression(hat(beta)))

## 4B: We need to select the ridge regression model
##        with the optimal lambda value 
##     There are two ways to do so

## 4B(i) manually find the optimal lambda value
##    but this is infeasible for cross-validation 
select(prostate.ridge)
## 
#modified HKB estimator is 3.355691 
#modified L-W estimator is 3.050708 
# smallest value of GCV  at 4.92 
#
# The output suggests that a good choice is lambda = 4.92, 
abline(v=4.92)
# Compare the ceofficients of ridge regression with lambda= 4.92
##  versus the full linear regression model #1 (i.e., with lambda = 0)
prostate.ridge$coef[, which(prostate.ridge$lambda == 4.92)]
prostate.ridge$coef[, which(prostate.ridge$lambda == 0)]

## 4B(ii) Auto-find the "index" for the optimal lambda value for Ridge regression 
##        and auto-compute the corresponding testing and testing error 
indexopt <-  which.min(prostate.ridge$GCV);  

## If you want, the corresponding coefficients with respect to the optimal "index"
##  it is okay not to check it!
prostate.ridge$coef[,indexopt]
## However, this coefficeints are for the the scaled/normalized data 
##      instead of original raw data 
## We need to transfer to the original data 
## Y = X \beta + \epsilon, and find the estimated \beta value 
##        for this "optimal" Ridge Regression Model
## For the estimated \beta, we need to sparate \beta_0 (intercept) with other \beta's
ridge.coeffs = prostate.ridge$coef[,indexopt]/ prostate.ridge$scales;
intercept = -sum( ridge.coeffs  * colMeans(training[,1:8] )  )+ mean(training[,9]);
## If you want to see the coefficients estimated from the Ridge Regression
##   on the original data scale
c(intercept, ridge.coeffs);

## Model 4 (Ridge): training errors 
yhat4train <- as.matrix( training[,1:8]) %*% as.vector(ridge.coeffs) + intercept;
MSEmod4train <- mean((yhat4train - training$lpsa)^2); 
MSEtrain <- c(MSEtrain, MSEmod4train); 
MSEtrain
## [1]  0.4391998 0.5210112 0.4393627 0.4473617
## Model 4 (Ridge):  testing errors in the subset "test" 
pred4test <- as.matrix( test[,1:8]) %*% as.vector(ridge.coeffs) + intercept;
MSEmod4test <-  mean((pred4test - ytrue)^2); 
MSEtest <- c(MSEtest, MSEmod4test);
MSEtest;
## [1] 0.5212740 0.4005308 0.5165135 0.4943531


## Model (5): LASSO 
## IMPORTANT: You need to install the R package "lars" beforehand
##
library(lars)
prostate.lars <- lars( as.matrix(training[,1:8]), training[,9], type= "lasso", trace= TRUE);

## 5A: some useful plots for LASSO for all penalty parameters \lambda 
plot(prostate.lars)

## 5B: choose the optimal \lambda value that minimizes Mellon's Cp criterion 
Cp1  <- summary(prostate.lars)$Cp;
index1 <- which.min(Cp1);

## 5B(i) if you want to see the beta coefficient values (except the intercepts)
##   There are three equivalent ways
##    the first two are directly from the lars algorithm
coef(prostate.lars)[index1,]
prostate.lars$beta[index1,]
##   the third way is to get the coefficients via prediction function 
lasso.lambda <- prostate.lars$lambda[index1]
coef.lars1 <- predict(prostate.lars, s=lasso.lambda, type="coef", mode="lambda")
coef.lars1$coef
## Can you get the intercept value? 
##  \beta0 = mean(Y) - mean(X)*\beta of training data
##       for all linear models including LASSO
LASSOintercept = mean(training[,9]) -sum( coef.lars1$coef  * colMeans(training[,1:8] ));
c(LASSOintercept, coef.lars1$coef)

## Model 5:  training error for lasso
## 
pred5train  <- predict(prostate.lars, as.matrix(training[,1:8]), s=lasso.lambda, type="fit", mode="lambda");
yhat5train <- pred5train$fit; 
MSEmod5train <- mean((yhat5train - training$lpsa)^2); 
MSEtrain <- c(MSEtrain, MSEmod5train); 
MSEtrain
# [1] 0.4391998 0.5210112 0.4393627 0.4473617 0.4398267
##
## Model 5:  training error for lasso  
pred5test <- predict(prostate.lars, as.matrix(test[,1:8]), s=lasso.lambda, type="fit", mode="lambda");
yhat5test <- pred5test$fit; 
MSEmod5test <- mean( (yhat5test - test$lpsa)^2); 
MSEtest <- c(MSEtest, MSEmod5test); 
MSEtest;
## Check your answer:
## [1] 0.5212740 0.4005308 0.5165135 0.4943531 0.5074249


#### Model 6: Principal Component Regression (PCR) 
##
## We can either manually conduct PCR by ourselves 
##   or use R package such as "pls" to auto-run PCR for us
##
## For purpose of learning, let us first conduct the manual run of PCR
##  6A: Manual PCR: 
##  6A (i) some fun plots for PCA of training data
trainpca <- prcomp(training[,1:8]);  
##
## 6A(ii)  Examine the square root of eigenvalues
## Most variation in the predictors can be explained 
## in the first a few dimensions
trainpca$sdev
round(trainpca$sdev,2)
### 6A (iii) Eigenvectors are in oj$rotation
### the dim of vectors is 8
###
matplot(1:8, trainpca$rot[,1:3], type ="l", xlab="", ylab="")
matplot(1:8, trainpca$rot[,1:5], type ="l", xlab="", ylab="")
##
## 6A (iv) Choose a number beyond which all e. values are relatively small 
plot(trainpca$sdev,type="l", ylab="SD of PC", xlab="PC number")
##
## 6A (v) An an example, suppose we want to do Regression on the first 4 PCs
## Get Pcs from obj$x
modelpca <- lm(lpsa ~ trainpca$x[,1:4], data = training)
##
## 6A (vi) note that this is on the PC space (denote by Z), with model Y= Z\gamma + epsilon
## Since the PCs Z= X U for the original data, this yields to 
## Y= X (U\gamma) + epsilon,
## which is the form Y=X\beta + epsilon in the original data space 
##  with \beta = U \gamma. 
beta.pca <- trainpca$rot[,1:4] %*% modelpca$coef[-1]; 
##
## 6A (vii) as a comparion of \beta for PCA, OLS, Ridge and LASSO
##   without intercepts, all on the original data scale
cbind(beta.pca, coef(model1)[-1], ridge.coeffs, coef.lars1$coef)
##
### 6A(viii) Prediciton for PCA
### To do so, we need to first standardize the training or testing data, 
### For any new data X, we need to impose the center as in the training data
###  This requires us to subtract the column mean of training from the test data
xmean <- apply(training[,1:8], 2, mean); 
xtesttransform <- as.matrix(sweep(test[,1:8], 2, xmean)); 
##
## 6A (iX) New testing data X on the four PCs
xtestPC <-  xtesttransform %*% trainpca$rot[,1:4]; 
##
## 6A (X) the Predicted Y
ypred6 <- cbind(1, xtestPC) %*% modelpca$coef;  
## 
## In practice, one must choose the number of PC carefully.
##   Use validation dataset to choose it. Or Use cross-Validation 
##  This can be done use the R package, say "pls"
##  in the "pls", use the K-fold CV -- default; divide the data into K=10 parts 
##
## 6B: auto-run PCR
##
## You need to first install the R package "pls" below
##
library(pls)
## 6B(i): call the pcr function to run the linear regression on all possible # of PCs.
##
prostate.pca <- pcr(lpsa~., data=training, validation="CV");  
## 
## 6B(ii) You can have some plots to see the effects on the number of PCs 
validationplot(prostate.pca);
summary(prostate.pca); 
## The minimum occurs at 8 components
## so for this dataset, maybe we should use full data
##
### 6B(iii) How to auto-select # of components
##     automatically optimazation by PCR based on the cross-validation
##     It chooses the optimal # of components 
ncompopt <- which.min(prostate.pca$validation$adj);
## 
## 6B(iv) Training Error with the optimal choice of PCs
ypred6train <- predict(prostate.pca, ncomp = ncompopt, newdata = training[1:8]); 
MSEmod6train <- mean( (ypred6train - training$lpsa)^2); 
MSEtrain <- c(MSEtrain, MSEmod6train); 
MSEtrain;
## 6B(v) Testing Error with the optimal choice of PCs
ypred6test <- predict(prostate.pca, ncomp = ncompopt, newdata = test[1:8]); 
MSEmod6test <- mean( (ypred6test - test$lpsa)^2); 
MSEtest <- c(MSEtest, MSEmod6test); 
MSEtest;
## Check your answer:
## [1] 0.5212740 0.4005308 0.5165135 0.4943531 0.5074249 0.5212740
##
## Fo this specific example, the optimal # of PC
##         ncompopt = 8, which is the full dimension of the original data
##   and thus the PCR reduces to the full model!!!


### Model 7. Partial Least Squares (PLS) Regression 
###
###  The idea is the same as the PCR and can be done by "pls" package
###  You need to call the fuction "plsr"  if you the code standalone 
#  library(pls)
prostate.pls <- plsr(lpsa ~ ., data = training, validation="CV");

### 7(i) auto-select the optimal # of components of PLS 
## choose the optimal # of components  
mod7ncompopt <- which.min(prostate.pls$validation$adj);
## The opt # of components, it turns out to be 8 for this dataset,
##       and thus PLS also reduces to the full model!!!    
 
# 7(ii) Training Error with the optimal choice of "mod7ncompopt" 
# note that the prediction is from "prostate.pls" with "mod7ncompopt" 
ypred7train <- predict(prostate.pls, ncomp = mod7ncompopt, newdata = training[1:8]); 
MSEmod7train <- mean( (ypred7train - training$lpsa)^2); 
MSEtrain <- c(MSEtrain, MSEmod7train); 
MSEtrain;
## 7(iii) Testing Error with the optimal choice of "mod7ncompopt" 
ypred7test <- predict(prostate.pls, ncomp = mod7ncompopt, newdata = test[1:8]); 
MSEmod7test <- mean( (ypred7test - test$lpsa)^2); 
MSEtest <- c(MSEtest, MSEmod7test); 
MSEtest;

## Check your answers
MSEtrain 
## Training errors of these 7 models/methods
#[1] 0.4391998 0.5210112 0.4393627 0.4473617 0.4398267 0.4391998 0.4391998
MSEtest
## Testing errors of these 7 models/methods
#[1] 0.5212740 0.4005308 0.5165135 0.4943531 0.5074249 0.5212740 0.5212740
##
## For this specific dataset, PCR and PLS reduce to the full model!!!
