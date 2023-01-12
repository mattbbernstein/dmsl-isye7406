setwd('C:/temp')  ## You might want to set the working directory to another folder 

## Suppose we save the data "zip.train.csv" in your working directory 
## read the data
zipa <- read.table(file="zip.train.csv", sep = ",");

## Here we only classify between 2's and 3's
zip23  <- subset(zipa, zipa[,1]==2 | zipa[,1]==3);

## Look at the first 3 rows
head(zip23,3)

## Exploratory Data Analysis
dim(zip23);                ## 1389  257
sum(zip23[,1] == 2);       ## 731

## The following results are too big for our data
summary(zip23);        ## summary statistics 
round(cor(zip23),2);   ## correlation matrix

## To see the letter picture of the 5-th row 
rowindex = 5;   zip23[rowindex,1]; ### it is 3! 
Xval = t(matrix(data.matrix(zip23[,-1])[rowindex,], byrow=TRUE,16,16)[16:1,]);
image(Xval, col=gray(0:1),axes=FALSE); 
image(Xval, col=gray(0:32/32),axes=FALSE) 

## As a demo, split the data into disjoint training and testing subset, 
## so that we can tune parameters and compare different models
n = dim(zip23)[1];  ## total number of observations
n1 = round(n/10);         ## number of observations for testing subset
set.seed(7406);           ## set the seed of randomization
flag = sort(sample(1:n, n1)); 
zip23train = zip23[-flag, ];  # Training subset
zip23test  = zip23[flag, ];   # Testing subset


library(class);  ## This R package includes the "KNN" algorithm
## KNN with kk=1,2,.,50 neighbors
kk <- 1:50; 

##The cross-validation error for this specific round for different k
cverror <- NULL;
for (i in 1:length(kk)){
  xnew <- zip23test[,-1];
  ypred.test <- knn(zip23train[,-1], xnew, zip23train[,1], k = kk[i]);
  temptesterror <- mean(ypred.test  != zip23test[,1]);
  cverror <- c(cverror, temptesterror); 
}


## This shows that KNN with k=1,3,5 ## yield the smallest CV error 
##  for this specific split 
plot(kk, cverror)

B <- 100;   #number of rounds or loops in the cross-validation
CVALL <- NULL;
for (b in 1:B){ 
  # randomly split training and testing subset in each loop
  flag = sort(sample(1:n, n1));  
  zip23train = zip23[-flag, ];    
  zip23test  = zip23[flag, ];
  ## Inside each loop, repeat the previous analysis of KNN for each k
  ## with a change on  cverror <- cbind(cverror, temptesterror);
  cverror <- NULL;
  for (i in 1:length(kk)){ 
    xnew <- zip23test[,-1];
    ypred.test <- knn(zip23train[,-1], xnew, zip23train[,1], k = kk[i]);
    temptesterror <- mean(ypred.test  != zip23test[,1]);
    cverror <- cbind(cverror, temptesterror); 
  }
  CVALL <- rbind(CVALL, cverror);
}   
# it took a while to get the results, as we run KNN for 100*50= 5000 times

##
## The plot shows that k=5 yields 
##  the smallest CV error among 
##  the B=100 Monte Carlo CV runs##
##
plot(kk, apply(CVALL, 2, mean), ylab='CV Error')

## By Cross-Validation, for this dataset,
## KNN algorithm with K=5 neighbors 
## is recommended to classify 2’s & 3’s. 



