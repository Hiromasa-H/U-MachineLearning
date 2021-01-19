dataset = read.csv("Position_Salaries.csv")
dataset = dataset[,2:3]

lin_reg = lm(formula = Salary ~., data=dataset)

dataset$Level2 = dataset$Level ** 2;
dataset$Level3 = dataset$Level ** 3;
dataset$Level4 = dataset$Level ** 4;
dataset$Level5 = dataset$Level ** 5;
dataset$Level6 = dataset$Level ** 6;
dataset$Level7 = dataset$Level ** 7;
dataset$Level8 = dataset$Level ** 8;
dataset$Level9 = dataset$Level ** 9;


poly_reg = lm(formula = Salary ~., data=dataset)

library(ggplot2)
ggplot() + 
  geom_point(aes(x = dataset$Level, y = dataset$Salary), color ="red")+
  geom_line(aes(x=dataset$Level, y = predict(lin_reg, newdata= dataset)),color = "blue")+
  geom_line(aes(x=dataset$Level, y = predict(poly_reg,newdata=dataset)),color="green")+
  ggtitle("truth or bluff")+xlab("Level")+ylab("Salary")

#predicting a new result with Linear Regression
y_pred = predict(lin_reg, data.frame(Level=6.5))
y_pred2 = predict(poly_reg,data.frame(Level=6.5,
                                      Level2 = 6.5**2,
                                      Level3 = 6.5**3,
                                      Level4 = 6.5**4,
                                      Level5 = 6.5**5,
                                      Level6 = 6.5**6,
                                      Level7 = 6.5**7,
                                      Level8 = 6.5**8,
                                      Level9 = 6.5**9))
