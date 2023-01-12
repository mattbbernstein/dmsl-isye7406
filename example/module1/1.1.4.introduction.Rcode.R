## The R codes in Lecture 1.1.4 Introduction to R

#data in R

x1 <- 1:10                         
x2 <-  seq(1,4,0.5)    
x2
## This is equivalent to
## 
x2 <- c(1.0,1.5,2.0,2.5,3.0,3.5,4.0)        

x3 <- rep(1,5)
x3

x4 <- gl(2,3,10)    #(generate factor levels)
## Check X4 values in R
x4

## Matrix in R
y <- matrix(c(1,2,3,4,5,6),2,3)
# This is 2 by 3 Matrix  filled in column by column
y

## Matrix operation using % %
w <- matrix(c(1,2), 2, 1)
## Thisis a 2X1 matrix or vector 
w
## here t() means the transpose of matrix 
t(w) %*% y 

##More matrix 
y[1,3]   # is the element on row 1 and column 3
y[1,]     # is the first row    	   
y[,2]     # is the second column

## Very useful matrix function 
apply(y,2,mean) #gives us the column means
apply(y,1,sd) #gives us the standard deviation of each row 
 

## Data Frame in R
# Generate 100 normal variables 
x <- rnorm(100)

# w will be used as 'weight' vector 
w <- 1 + x/2 

# Generate the observed Y from x and w
y <- x + w * rnorm(x)

# Make a data frame of three columns named x, y, w
dum <- data.frame(x,y,w)
dum

## Fit a simple ordinary linear regression of y and x

fm1 <- lm(y ~ x, data = dum)
summary(fm1)

## Some numerical results from the above fit  
residuals(fm1)
predict(fm1)
coef(fm1)

## plot different plot of linear regression
plot(fm1)
plot(fitted(fm1),resid(fm1),xlab="Fitted Values", ylab= "Residuals")
qqnorm(resid(fm1))
qqline(resid(fm1))
 

## We can also do a weighted least squares regression
fm2 <- lm(y ~ x, data = dum, weight = 1 / w^2)
summary(fm2)
## sometimes you need to use the "attach" function
## to make the local variables become global  
#attach(dum)

## Plot the data and two fits 
plot(x,y)
abline(fm1)
abline(fm2, col="red")

## Programming language 
### Avoid Loops in R 

## Example 1: check whether X value is 3 or not
x <- c(-2,3,5,6)

# 1A: a very slow R code
y <- numeric(length(x))
for (i in 1:length(x)) 
   if (x[i]==3) y[i] <- 0 else y[i] <- 1
y

# 1B: a more efficient R code 
y <- (x != 3)
y


## Example 2: Bonus: generate bootstrapping sample

## 2A:  a slow code
bootsamp <- function(x){ 	
  n <- length(x)
  sample <- x
  for ( i in 1:n) {
    u <- ceiling(runif(1,0,n))
    sample[i] <- x[u] 	
  } 	
  return(sample)
}

x <- c(0,3,5,7)              
bootsamp(x) 

## 2B: a more efficient R code (i.e., avoiding the loops)  

bootsampB <- function(x){ 	
  n <- length(x)
  sample <- x[ceiling(runif(n,0,n))]
  return(sample)
}
x <- c(0,3,5,7)              
bootsampB(x) 

