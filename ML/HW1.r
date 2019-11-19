# 7.9.6a
######################################
### Library

library(ISLR) #ISLR Package
library(boot) #cv.glm을 사용하기 위한 boot package

######################################
### 과제 검사를 위한 Seed값 고정
### 과제마감일인 11월 21일을 참조함
set.seed(1) 

######################################
### MSE값이 가장 낮은 degree를 찾기 위해 CV 실시
### 1차~7차의 MSE는 poly_MSE에 저장된다.

poly_MSE=c()
for(degree in 1:7){
  fit=glm(wage~poly(age,degree,raw=T),data=Wage)
  MSE=cv.glm(fit,data = Wage,K=10) $ delta[1]
  poly_MSE=c(poly_MSE,MSE)
}

######################################
### poly_MSE에 저장된 MSE 중 Error 값이 가장 낮은 값을 찾는 과정

plot(poly_MSE,xlab='Degree',ylab='Error',type='l')
x=which.min(poly_MSE)
points(x,poly_MSE[x],col='red')

######################################
### 결과는 degree=6일 때, Error값이 가장 낮음
### 하지만, degree=4일 때부터 CV Error값이 크게 줄어들지 않기에 4가 적절하다고 생각함
### 3차~6차의 유의성을 판단하기 위해 anova 검사 실행

fit.3=lm(wage~poly(age,3),data=Wage) #3차
fit.4=lm(wage~poly(age,4),data=Wage) #4차
fit.5=lm(wage~poly(age,5),data=Wage) #5차
fit.6=lm(wage~poly(age,6),data=Wage) #6차
anova(fit.3, fit.4, fit.5, fit.6) 

#Pvalue가 0.05에 비슷한 4차 다항식이 가장 결과가 좋음을 알 수 있음

######################################

# 7.9.6b
######################################
step_MSE=c()

for(breaks in 2:10){
  Wage$age.cut = cut(Wage$age, breaks)
  lm.fit = glm(wage~age.cut, data=Wage)
  MSE=cv.glm(Wage, lm.fit, K=10)$delta[2]
  step_MSE=c(step_MSE, MSE)
}


plot(step_MSE,xlab='Number of Cut',ylab='Error',type='l')
x=which.min(step_MSE)
points(x,step_MSE[x],col='red')

lm.fit = glm(wage~cut(age, 7), data=Wage)
agelims = range(Wage$age)
age.grid = seq(from=agelims[1], to=agelims[2])
lm.pred = predict(lm.fit, data.frame(age=age.grid))
plot(wage~age, data=Wage, col="darkgrey")
lines(age.grid, lm.pred, col="red", lwd=2)

######################################

# 7.9.7
######################################

summary(Wage[,c('wage','age', 'year','jobclass','health_ins','health')])

par(mfrow = c(2, 2))
plot(Wage$year, Wage$wage)
plot(Wage$jobclass, Wage$wage)
plot(Wage$health_ins, Wage$wage)
plot(Wage$health, Wage$wage)


library(gam)
fit = gam(wage ~ health + health_ins + jobclass + s(age, 4) + s(year,4), data = Wage)
par(mfrow=c(1,3))
plot(fit, se=TRUE,col="blue")
#year는 linear function을 그려보았을 때, band 안에 포함되도록 그릴 수 없으므로 linear로 불가함

fit = gam(wage ~ health + health_ins + jobclass + s(age, 4) + lo(year, span = 0.7), data = Wage)
summary(fit)

par(mfrow = c(2, 3))
plot(fit, se = T, col = "blue")

fit2 = gam(I(wage>250) ~ health + health_ins + jobclass + s(age, 4) + lo(year, span = 0.7), data = Wage)
summary(fit2)

par(mfrow = c(2, 3))
plot(fit2, se = T, col = "blue")
