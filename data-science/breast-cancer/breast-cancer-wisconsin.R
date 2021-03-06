# 빅데이터 분별분석. 암 예측2.
#

rm(list=ls())

#gbm 개발자 버전
#remove.packages("gbm")
#install.packages("devtools", dependencies=TRUE)
#library(devtools)
#install_github("gbm-developers/gbm")

library(gbm)

rmse <- function(yi, yhat_i){
  sqrt(mean((yi - yhat_i)^2))
}

binomial_deviance <- function(y_obs, yhat){
  epsilon = 0.0001
  yhat = ifelse(yhat < epsilon, epsilon, yhat)
  yhat = ifelse(yhat > 1-epsilon, 1-epsilon, yhat)
  a = ifelse(y_obs==0, 0, y_obs * log(y_obs/yhat))
  b = ifelse(y_obs==1, 0, (1-y_obs) * log((1-y_obs)/(1-yhat)))
  return(2*sum(a + b))
}


panel.cor <- function(x, y, digits = 2, prefix = "", cex.cor, ...){
  usr <- par("usr"); on.exit(par(usr))
  par(usr = c(0, 1, 0, 1))
  r <- abs(cor(x, y))
  txt <- format(c(r, 0.123456789), digits = digits)[1]
  txt <- paste0(prefix, txt)
  if(missing(cex.cor)) cex.cor <- 0.8/strwidth(txt)
  text(0.5, 0.5, txt, cex = cex.cor * r)
}


library(tidyverse)
library(gridExtra)
library(MASS)
library(glmnet)
library(randomForest)
library(gbm)
library(rpart)
library(boot)
library(data.table)
library(ROCR)

URL <- "http://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/breast-cancer-wisconsin.data"
download.file(URL, destfile = "data2.csv")

data <- tbl_df(read.table(file.choose(), strip.white = TRUE,
                          sep=",", header = FALSE, na.strings = '?'))
glimpse(data)

names(data) <- c('id', 'thickness', 'unif_cell_size', 'unif_cell_shape',
                 'marginal_adhesion', 'cell_size', 'bare_nuclei',
                 'bland_cromatin', 'normal_nucleoli', 'mitoses', 'class')

# 1. 결측치 처리
data$bare_nuclei[is.na(data$bare_nuclei)] <- median(data$bare_nuclei, na.rm = TRUE)
# 2. id 변수 제거
data <- data %>% dplyr::select(-id)
# 3. class 변수를 인자 변수로 변환
data$class <- factor(ifelse(data$class == 2, 0, 1))

glimpse(data)

summary(data)

# 시각화1
pairs(data %>% sample_n(min(1000, nrow(data))),
      lower.panel=function(x,y){ points(x,y); abline(0, 1, col='red')},
      upper.panel = panel.cor)

dev.off()

#시각화2
library(ggplot2)
library(dplyr)
library(gridExtra)
#p1 : class 변수 확인
p1 <- data %>% ggplot(aes(class)) + geom_bar()
#p2 : class와 unif_cell_size 변수 관계 확인
p2 <- data %>% ggplot(aes(class, unif_cell_size)) +
  geom_jitter(col='gray') +
  geom_boxplot(alpha=.5)
#p3 : class와 bare_nuclei 변수 관계 확인
p3 <- data %>% ggplot(aes(class, bare_nuclei)) +
  geom_jitter(col='gray') +
  geom_boxplot(alpha=.5)
#p4 : unif_cell_size와 bare_nuclei 변수 관계 확ㅇ
p4 <- data %>% ggplot(aes(unif_cell_size, bare_nuclei)) +
  geom_jitter(col='gray') + geom_smooth()
grid.arrange(p1, p2, p3, p4, ncol=2)



# 트래인셋과 테스트셋의 구분
set.seed(1810)
n <- nrow(data)
idx <- 1:n
training_idx <- sample(idx, n * .60)
idx <- setdiff(idx, training_idx)
validate_idx <- sample(idx, n * .20)
test_idx <- setdiff(idx, validate_idx)
training <- data[training_idx,]
validation <- data[validate_idx,]
test <- data[test_idx,] 


#-----------------
# 로지스틱 회귀모형
data_lm_full <- glm(class ~ ., data=training, family=binomial)
summary(data_lm_full)

predict(data_lm_full, newdata = data[1:5,], type='response')

# 모형평가
y_obs <- as.numeric(as.character(validation$class))
yhat_lm <- predict(data_lm_full, newdata = validation, type='response')
pred_lm <- prediction(yhat_lm, y_obs)
performance(pred_lm, "auc")@y.values[[1]]
binomial_deviance(y_obs, yhat_lm)

rmse(y_obs, yhat_lm)

library(ROCR)
#-----------------
# 라쏘 모형 적합
# xx <- model.matrix(class ~ .^2-1, data)
xx <- model.matrix(class ~ .-1, data)
x <- xx[training_idx, ]
y <- as.numeric(training$class)
glimpse(x)

data_cvfit <- cv.glmnet(x, y, family = "binomial")
plot(data_cvfit)


coef(data_cvfit, s = c("lambda.1se"))
coef(data_cvfit, s = c("lambda.min"))


predict.cv.glmnet(data_cvfit, s="lambda.min", newx = x[1:5,])

yhat_glmnet <- predict(data_cvfit, s="lambda.min", newx=xx[validate_idx,], type='response')
yhat_glmnet <- yhat_glmnet[,1] # change to a vector from [n*1] matrix
pred_glmnet <- prediction(yhat_glmnet, y_obs)
performance(pred_glmnet, "auc")@y.values[[1]]

binomial_deviance(y_obs, yhat_glmnet)

rmse(y_obs, yhat_glmnet)

#-----------------
# 나무모형
data_tr <- rpart(class ~ ., data = training)
data_tr

printcp(data_tr)
summary(data_tr)

opar <- par(mfrow = c(1,1), xpd = NA)
plot(data_tr)
text(data_tr, use.n = TRUE)
par(opar)


yhat_tr <- predict(data_tr, validation)
rmse(y_obs, yhat_tr)


#-----------------
# 랜덤포레스트

set.seed(1810)
data_rf <- randomForest(class ~ ., training)
data_rf

opar <- par(mfrow=c(1,2))
plot(data_rf)
varImpPlot(data_rf)
par(opar)
dev.off()


yhat_rf <- predict(data_rf, newdata=validation, type='prob')[,'1']
pred_rf <- prediction(yhat_rf, y_obs)
performance(pred_rf, "auc")@y.values[[1]]
binomial_deviance(y_obs, yhat_rf)

rmse(y_obs, yhat_rf)
#-----------------
# 부스팅

set.seed(1810)

#만약 best_iter값 계산시 오류가 발생되면, factor가 아닌 int형으로 바꿔서 해볼 거
#training$class <- as.integer(ifelse(training$class == 0, 0, 1))

data_gbm <- gbm(class ~ ., data=training, distribution="bernoulli",
                n.trees=10000, cv.folds=3, verbose = TRUE)

(best_iter = gbm.perf(data_gbm, method="cv"))

yhat_gbm <- predict(data_gbm, n.trees=best_iter, newdata=validation, type='response')
pred_gbm <- prediction(yhat_gbm, y_obs)
performance(pred_gbm, "auc")@y.values[[1]]
binomial_deviance(y_obs, yhat_gbm)


rmse(y_obs, yhat_gbm)


# 최종 모형선택과  테스트셋 오차계산
data.frame(method=c('lm', 'glmnet', 'rf', 'gbm'),
           auc = c(performance(pred_lm, "auc")@y.values[[1]],
                   performance(pred_glmnet, "auc")@y.values[[1]],
                   performance(pred_rf, "auc")@y.values[[1]],
                   performance(pred_gbm, "auc")@y.values[[1]]),
           bin_dev = c(binomial_deviance(y_obs, yhat_lm),
                       binomial_deviance(y_obs, yhat_glmnet),
                       binomial_deviance(y_obs, yhat_rf),
                       binomial_deviance(y_obs, yhat_gbm)))


perf_lm <- performance(pred_lm, measure = "tpr", x.measure = "fpr")
perf_glmnet <- performance(pred_glmnet, measure="tpr", x.measure="fpr")
perf_rf <- performance(pred_rf, measure="tpr", x.measure="fpr")
perf_gbm <- performance(pred_gbm, measure="tpr", x.measure="fpr")

plot(perf_lm, col='black', main="ROC Curve")
plot(perf_glmnet, add=TRUE, col='blue')
plot(perf_rf, add=TRUE, col='red')
plot(perf_gbm, add=TRUE, col='cyan')
abline(0,1)
legend('bottomright', inset=.1,
       legend=c("GLM", "glmnet", "RF", "GBM"),
       col=c('black', 'blue', 'red', 'cyan'), lty=1, lwd=2) 

#최종모형 gbm으로 선택

y_obs_test <- as.numeric(as.character(test$class))
yhat_gbm_test <- predict(data_gbm, n.trees=best_iter, newdata=test, type='response')
pred_gbm_test <- prediction(yhat_gbm_test, y_obs_test)
performance(pred_gbm_test, "auc")@y.values[[1]]
binomial_deviance(y_obs_test, yhat_gbm_test)


# 회귀분석의 오차의 시각화
boxplot(list(lm = y_obs-yhat_lm,
             glmnet = y_obs-yhat_glmnet,
             rf = y_obs-yhat_rf,
             gbm = y_obs-yhat_gbm), ylab="Error in Validation Set")
abline(h=0, lty=2, col='blue')


# 예측값 시각ㅎ
pairs(data.frame(y_obs=y_obs,
                 yhat_lm=yhat_lm,
                 yhat_glmnet=c(yhat_glmnet),
                 yhat_rf=yhat_rf,
                 yhat_gbm=yhat_gbm),
      lower.panel=function(x,y){ points(x,y); abline(0, 1, col='red')},
      upper.panel = panel.cor) 
