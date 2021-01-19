#Apriori

# data preprocessing
library(arules)
#dataset = read.csv("Market_Basket_Optimisation.csv", header = FALSE)
dataset = read.transactions("Market_Basket_Optimisation.csv", sep = ",",rm.duplicates = TRUE)
#>distribution of transactions with duplicates:
#  1 
#  5
# 5 transactions including 1 duplicate
summary(dataset)
itemFrequencyPlot(dataset,topN = 100)

#support = 3*7/7500=0.0028 = 0.003
#support = 4*7/7500=0.0037 = 0.004
#training Apriori on the dataset
rules = apriori(data = dataset, parameter = list(support = 0.004,confidence = 0.2 ))
#confidence is 0.8 by default

inspect(sort(rules, by = "lift")[1:10])
#this code makes sense but burned my brain 