#Eclat

library(arules)
dataset = read.transactions("Market_Basket_Optimisation.csv", sep = ",",rm.duplicates = TRUE)
summary(dataset)

#Training Eclat on the dataset
sprt = 4*7/7500
rules = eclat(data = dataset, parameter = list(support = sprt, minlen = 2))

#sort by decreasing support
inspect(sort(rules, by = "support")[1:10])