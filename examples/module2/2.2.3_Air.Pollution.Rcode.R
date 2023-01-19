## Example:  Linear Regression: Air Pollution and Mortality
## 
## We assume that you save the data "mortality1.txt" in your local folder "C:/temp")
## Read the data 
## 

data1 <- read.table("C:/temp/mortality1.txt",header=T)


## plot for possible collinrarity?

library(lattice)
splom(data1[,1:6], pscales = 0)


# The corrlation table (the last column is Y)
cor(data1[1:6]); 

## Different to read so many digits
## Only keep 2 digits in the correlation matrix 
round(cor(data1[1:6]),2)


# do Linear Regression on Prec, Educ, NW

## First-order model
fm1 <- lm(Mort ~ Prec + Educ + NW, data = data1)
summary(fm1)

## What if we want to add (three-way) interaction terms
##   among Prec, Educ, and NW?

fm1b <- lm(Mort ~ Prec * Educ * NW, data = data1)
summary(fm1b)

## How about the Second-order models
## There are two ways. 
## First, the most natural way
fm1c <- lm( Mort ~ Prec +I(Prec^2) + Educ + I(Educ^2) + NW + I(NW^2) + I(Prec*Educ) + I(Prec*NW) + I(Educ*NW), data = data1)
summary(fm1c)

### Prediction on different prec and other X values are mean values##

apply(data1[,1:5],2, mean)
#     Prec      Educ        NW       NOX       SO2  
# 37.36667  10.97333  11.87000  22.65000  53.76667 

xnew <- data.frame(Prec = seq(10,60,5), Educ=10.97333, NW=11.87, NOX=22.65, SO2=53.76667)
predict(fm1c, xnew, interval="prediction", level=0.95) 

## second, use the orthogonal plynomials 
fm1d <- lm(Mort ~ poly(Prec, Educ, NW, degree =2), data = data1)
summary(fm1d)

## same fitted model, but different coefficients
fitted(fm1d)[1:10]
fitted(fm1c)[1:10]

# predict(fm1d, xnew, interval="prediction", level=0.95)

## Unfortunately, sometimes it is difficult to use fm1d to do prediction
#   Error in poly(dots[[1L]], degree, raw = raw) : 
#  'degree' must be less than number of unique points



### Now go back the first-order model
fm1 <- lm(Mort ~ Prec + Educ + NW, data = data1)
summary(fm1)

# we can see that Prec is not sig. while the two other variables are
# Useful Plots for regression 

par(mfrow = c(2,2))
plot(fm1)

# From the plot, we can remove two outlier obs.
#    #28 (lan, i.e., Lancaster, PA)
#    #32 (mia, i.e., Miami, FL).
# From notem York (yrk), like Lancaster, PA is heavily 
#    populated by memembers of the Amish religion
#    so obs, for York, pA (#59) are also removed
# New data deleted three cities #28,32,59


# 
# Build the baseline model after removing outliers 
#
rdata1 <- data1[c(-28,-32,-59),]
rfm1 <- lm(Mort ~ Prec + Educ + NW, data = rdata1)
summary(rfm1)

# plots show the model fit the data well
par(mfrow = c(2,2))
plot(rfm1)


# Add the NOX variable to the baseline model 

rfm2 <- lm(Mort ~ Prec + Educ + NW + NOX, data = rdata1)
summary(rfm2)

## F-test for model evaluation to see whether NOX variable is significant or not
anova(rfm1,rfm2)

# The large p-value (0.1646) shows that mortality may not be associtaed with 
# NOX once the effects of the climate and socioeconomic variables
# are accounted for.



# Now Add the SO2 variable to the baseline model

rfm3 <- lm(Mort ~ Prec + Educ + NW + SO2, data = rdata1)
summary(rfm3)
anova(rfm1,rfm3)

# The small p-value indicates that there is strong evidence that 
# increased SO2 pollution increases mortality 
#   (since the coefficient for SO2 is positive). 


# Finally, add both NOX and SO2 to the baseline model

rfm4 <- lm(Mort ~ Prec + Educ + NW + NOX + SO2, data = rdata1)
summary(rfm4)
anova(rfm1, rfm4)

# The results confirm that while the coefficient for NOX is not significnat,
# the coefficient for SO2 is significant.

# You can also look at the plot of the new regression

par(mfrow = c(2,2))
plot(rfm4)

# We may want to remove the observations for New Orleans, LA (no, #37) 
#    or Los Angeles, CA (la, #29) as well.
# Thus, Continue to delete #29, #37 of data1 
#    or to delete #28, #35 from rdata1

rdata2 <- rdata1[c(-28,-35),]
rfm5 <- lm(Mort ~ Prec + Educ + NW + NOX + SO2, data = rdata2)
summary(rfm5)

par(mfrow = c(2,2))
plot(rfm5)

# we stll have similar conclusion.
# Therefore, we conclude that increased SO2 leads to increased mortality,
#      whereas NOX levels do not correlate positively with mortality, 
#      after the effects of the climate and socioeconomic variables
#      are accounted for.




