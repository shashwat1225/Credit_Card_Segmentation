#clear all data
rm(list=ls())
#Set Working Directory
setwd("C:/Users/atul/Desktop/Data Science/Project-2")
# Check version
packageVersion("callr") #'3.3.1'
# Build from source
install.packages("callr", type="source")
# Check version
packageVersion("callr") #'3.4.0'
install.packages("tidyverse", type="source")
x = c("Hmisc" , "mice" , "DMwR", "ggplot2", "factoextra" , "devtools" , "ggthemes" , "plotly" , "htmlwidgets" , "GGally" , "tidyr" , "ggplot2", "corrgram", "DMwR", "usdm", "caret", "randomForest", "e1071",
      "DataCombine", "doSNOW", "inTrees","tidyr", "tidyverse",  "DataExplorer", "rpart.plot", "rpart","MASS","xgboost","stats","dplyr")
#load Packages
lapply(x, require, character.only = TRUE)
update.packages ()
# this will update all your packages
install.packages ("tidyverse")
install.packages("mice")
install.packages("xgboost")
install.packages("DmWR")
install.packages("tidyr")
install.packages("DataExplorer")
install.packages("Hmisc")
install.packages("mice")
install.packages("ggthemes")
install.packages("plotly")
install.packages("GGally")
install.packages("doSNOW")
rm(x)
cc = read.csv("credit-card-data.csv", header = T, na.strings = c(" ", "", "NA"))

##################EXPLORATORY DATA ANALYSIS##################
#Structure of the database
str(cc)
summary(cc)
head(cc)
tail(cc)
names(cc)
dim(cc)
cc_copy = cc
#################MISSING VALUE ANALYSIS######################
#Finding all the missing values
mv = data.frame(apply(cc_copy,2,function(x){sum(is.na(x))}))

#As we can observe, there is only 1 missing value in CREDIT_LIMIT and 313 missing values in MINIMUM_PAYMENTS
#To check whether to keep these variables or not, we willl calculate the missing value percentage
plot_missing(cc)

#As seen in the graph, we see that MINIMUM_PAYMENTS has 3.5% missing values and CREDIT_LIMIT has 0.01% missing values, which is way below than 30% threshold to delete a variable. Hence, we will consider all the variables and impute the missing values
#Since both the variables are numeric, we will use either mean or kNN method to perform imputation. 
#Selecting a random value from the dataset to check accuracy of both methods
#Actual value[2401,15] = 1148.9028
#Mean Method
cc_copy$MINIMUM_PAYMENTS[is.na(cc_copy$MINIMUM_PAYMENTS)] = mean(cc_copy$MINIMUM_PAYMENTS, na.rm = T)
#cc_copy$CREDIT_LIMIT[is.na(cc_copy$CREDIT_LIMIT)] = mean(cc_copy$CREDIT_LIMIT, na.rm = T)

#Value by mean = 864.1736

#kNN Method
cc_copy = knnImputation(cc_copy, k = 6)
#Value(k=5):419.9769
#Value(k=6):2035.935
#Value(k=5):2090.969
#The closest value to the original was found with the mean method, hence we will use that.
cc$MINIMUM_PAYMENTS[is.na(cc$MINIMUM_PAYMENTS)] = mean(cc$MINIMUM_PAYMENTS, na.rm = T)
cc$CREDIT_LIMIT[is.na(cc$CREDIT_LIMIT)] = mean(cc$CREDIT_LIMIT, na.rm = T)

##########################OUTLIER ANALYSIS############################
#Since our problem statement requires us to create a segmentation(classification) model which is an unsupervised learning technique, we won't be performing outlier analysis

##########################FEATURE SELECTION##########################
#Will remove CUST_ID since, it is the only categorical variable and is not even going to be used in our clustering analysis
cc = cc[-1]
summary(cc)
cc_copy = cc

##########################CORRELATION ANALYSIS###########################
#PLotting the correlation plot
ggcorr(cc_copy, 
       label = T, 
       label_size = 3,
       label_round = 2,
       hjust = 1,
       size = 3, 
       color = "royalblue",
       layout.exp = 5,
       low = "dodgerblue", 
       mid = "gray95", 
       high = "red2",
       name = "Correlation")

############################FEATURE SCALING############################
#

normalize <- function(x) {
  return ((x - min(x)) / (max(x) - min(x)))
}
cc_copy <- normalize(cc_copy)

summary(cc_copy)

##############################VISUALIZATIONS###############################
#We have to see whether there is some interesting finding on the data distribution, especially on Balance, Purchase, Credit Limit, and Tenure.

#Customer Balance

plot1 =  ggplot(cc_copy, aes(x=BALANCE)) +
  geom_histogram(col = "cyan", fill = "dodgerblue", bins = 30) +
  labs(x = "BALANCE", y = "Frequency", title = "Histogram of Customer Balance") +
  theme_igray()
ggplotly(plot1)

#Customer Purchase

plot2 = ggplot(cc_copy, aes(x=PURCHASES)) +
  geom_histogram(col = "lawngreen", fill = "springgreen4", bins = 40) +
  labs(x = "Purchase", y = "Frequency", title = "Histogram of Customer Purchase") +
  theme_igray()
ggplotly(plot2)


#Credit Limit

plot3 = ggplot(cc_copy, aes(x=CREDIT_LIMIT)) +
  geom_histogram(col = "yellow2", fill = "orangered2", bins = 30) +
  labs(x = "Credit Limit", y = "Frequency", title = "Histogram of Credit Limit") +
  theme_igray()
ggplotly(plot3)


#Tenure

plot4 =  ggplot(cc_copy, aes(x=TENURE)) +
  geom_bar(col = "magenta", fill = "maroon3") +
  labs(x = "Tenure", y = "Frequency", title = "Bar Chart of Tenure") +
  theme_igray()
ggplotly(plot4)

#Cash Advancements

plot5 =  ggplot(cc_copy, aes(x=CASH_ADVANCE_FREQUENCY)) +
  geom_bar(col = "blue", fill = "green") +
  labs(x = "Cash Advances", y = "Frequency", title = "Bar Chart of Cash advances") +
  theme_igray()
ggplotly(plot5)

##########################CLUSTERING##########################

#K-means cluestering:

#finding no of clusters to build using elbow graph
set.seed(123)
# Compute and plot wss for k = 2 to k = 15.
k.max <- 15
data <- cc_copy
wss <- sapply(1:k.max, 
              function(k){kmeans(data, k, nstart=50,iter.max = 15 )$tot.withinss})
wss
plot(1:k.max, wss,
     type="b", pch = 19, frame = FALSE, 
     xlab="Number of clusters K",
     ylab="Total within-clusters sum of squares")

#using the elbow graph we can see that  no of clusters to be 4

kmeans_model = kmeans(data, 8, nstart = 50, iter.max = 15)
#we keep number of iter.max=15 toa ensure the algorithm converges and nstart=50 to ensure that atleat 50 random sets are choosen  
#As the final result of k-means clustering result is sensitive to the random starting assignments, we specify nstart = 25. 
#This means that R will try 25 different random starting assignments and then select the best results corresponding to the one with the lowest within cluster variation. 

#Summarize cluster output
kmeans_model

#CLuster analysis:

Cluster_data = cbind(cc, clusterID = kmeans_model$cluster)
Cluster_data = data.frame(Cluster_data)
head(Cluster_data)

install.packages("cluster")
library(cluster)
clusplot(cc, kmeans_model$cluster, color=TRUE, shade=TRUE,labels=2, lines=0)

###############################VISUALIZATIONS###############################
#plotting obtained clusters on existing data:

p1<-ggplot(Cluster_data, aes(x = clusterID, y = PURCHASES)) +
  geom_bar(fill = "#0073C2FF", stat = "identity") +
  geom_text(aes(label = PURCHASES), vjust = -0.3)
ggplotly(p1)
p2<-ggplot(Cluster_data, aes(x = clusterID, y = ONEOFF_PURCHASES)) +
  geom_bar(fill = "#0073C2FF", stat = "identity") +
  geom_text(aes(label = ONEOFF_PURCHASES), vjust = -0.3)
ggplotly(p2)
p3<-ggplot(Cluster_data, aes(x = clusterID, y = INSTALLMENTS_PURCHASES)) +
  geom_bar(fill = "#0073C2FF", stat = "identity") +
  geom_text(aes(label = INSTALLMENTS_PURCHASES), vjust = -0.3)
ggplotly(p3)
p4<-ggplot(Cluster_data, aes(x = clusterID, y = CREDIT_LIMIT)) +
  geom_bar(fill = "#0073C2FF", stat = "identity") +
  geom_text(aes(label = CREDIT_LIMIT), vjust = -0.3)
ggplotly(p4)
p5<-ggplot(Cluster_data, aes(x = clusterID, y = PAYMENTS)) +
  geom_bar(fill = "#0073C2FF", stat = "identity") +
  geom_text(aes(label = PAYMENTS), vjust = -0.3)
ggplotly(p5)
p6<-ggplot(Cluster_data, aes(x = clusterID, y = PRC_FULL_PAYMENT)) +
  geom_bar(fill = "#0073C2FF", stat = "identity") +
  geom_text(aes(label = PRC_FULL_PAYMENT), vjust = -0.3)
ggplotly(p6)

gridExtra::grid.arrange(p1,p2,p3,p4,p5,p6, ncol=3)


#From the barplots we can conclude:

#Out of the 8 clusters, clusters like 1,3,8 are of high purchase and they tend to buy both one-off purhase and installment purchases high but they are good at insallment payments but not one off payments. SO its better offer them good plans for installment payments clusters like 1,3,6,7 are having high credit limit and we can see that people who have high purchases also has high payments and full payments indirectly showing that high credit limit comes with good payment history on the other hand from clusters like 2,4,5,8 we can see that people with low credit limit are likely to go for one-off purchase over installments and its common sense to choose one off purchases when you have low credit limit as installments need mostly need high credit limit for bigger purchases
