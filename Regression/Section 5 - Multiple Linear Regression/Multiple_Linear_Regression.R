dataset = read.csv("50_Startups.csv")
#dataset = dataset[,2:3]


dataset$State = factor(dataset$State,levels = c("New York","California","Florida"),labels = c(1,2,3))

#Splitting the dataset into the Training set and Test set
library("caTools")
set.seed(123)
split = sample.split(dataset$Profit, SplitRatio = 0.8)
training_set = subset(dataset,split==TRUE)
test_set= subset(dataset,split==FALSE)

# Feature Scaling
# training_set[,2:3] = scale(training_set[,2:3])
# test_set[,2:3] = scale(test_set[,2:3])

regressor = lm(formula = Profit ~ .,
               data= training_set)

# 
# regressor_opt = lm(formula = Profit ~ R.D.Spend,
#                    data= training_set)

y_pred = predict(regressor,newdata = test_set)

#backwards elimination
regressor_opt = lm(formula = Profit ~ R.D.Spend + Marketing.Spend,
               data = dataset)
