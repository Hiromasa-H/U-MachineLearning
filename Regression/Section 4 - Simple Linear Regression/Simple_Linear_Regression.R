
dataset = read.csv("Salary_Data.csv")
#dataset = dataset[,2:3]

#Splitting the dataset into the Training set and Test set
library("caTools")
set.seed(123)
split = sample.split(dataset$Salary, SplitRatio = 0.8)
training_set = subset(dataset,split==TRUE)
test_set= subset(dataset,split==FALSE)

# Feature Scaling will be taken care of via the library

#Simple Linear Regression 
regressor  = lm(formula = Salary ~ YearsExperience,
                data = training_set)

#Predicting the Test set results
y_pred = predict(regressor,newdata=test_set)

#plotting the stuff.
library(ggplot2)
ggplot()+
  geom_point(aes(x=training_set$YearsExperience,y=training_set$Salary),
             color="red")+
  geom_smooth(aes(x=training_set$YearsExperience,y=predict(regressor,newdata=training_set)),color="blue")+
  ggtitle("Salary vs Experience (training set)")+
  xlab("years of experience") + ylab("salary")

ggplot()+
  geom_point(aes(x=test_set$YearsExperience,y=test_set$Salary),
             color="red")+
  geom_smooth(aes(x=training_set$YearsExperience,y=predict(regressor,newdata=training_set)),color="blue")+
  ggtitle("Salary vs Experience (testing set)")+
  xlab("years of experience") + ylab("salary")

