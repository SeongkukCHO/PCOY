# 빅데이터 회귀분석. 피부 질병 예측
#

rm(list=ls())

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
library(ggplot2)
library(dplyr)
library(gridExtra)

# 데이터 Read
URL <- "https://archive.ics.uci.edu/ml/machine-learning-databases/dermatology/dermatology.data"
download.file(URL, destfile = "data6.csv")

data <- tbl_df(read.table(file.choose(), strip.white = TRUE,
                          sep=",", header = FALSE))
data = data[-34]
names(data) <- c('erythema', 'scaling', 'definite_borders', 'itching',
                 'koebner_phenomenon','polygonal_papules',
                 'follicular_papules', 'oral_mucosal_involvement',
                 'knee_and_elbow_involvement', 'scalp_involvement','family_history',
                 'melanin_incontinence','eosinophils_in_the_infiltrate','PNL_infiltrate',
                 'fibrosis_of_the_papillary_dermis','exocytosis','acanthosis',
                 'hyperkeratosis','parakeratosis','clubbing_of_the_rete_ridges',
                 'elongation_of_the_rete_ridges',
                 'thinning_of_the_suprapapillary_epidermis','spongiform_pustule',
                 'munro_microabcess','focal_hypergranulosis',
                 'disappearance_of_the_granular_layer',
                 'vacuolisation_and_damage_of_basal_layer','spongiosis',
                 'sawtooth_appearance_of_retes','follicular_horn_plug',
                 'perifollicular_parakeratosis','inflammatory_monoluclear_inflitrate',
                 'bandlike_infiltrate','Age')
glimpse(data)
summary(data)

# 시각화
set.seed(1810)

pairs(data %>% dplyr::select(1:10,34) %>%
        sample_n(min(1000, nrow(data))),
      lower.panel=function(x,y){ points(x,y); abline(0, 1, col='red')},
      upper.panel = panel.cor)
pairs(data %>% dplyr::select(11:20,34) %>%
        sample_n(min(1000, nrow(data))),
      lower.panel=function(x,y){ points(x,y); abline(0, 1, col='red')},
      upper.panel = panel.cor)
pairs(data %>% dplyr::select(21:34) %>%
        sample_n(min(1000, nrow(data))),
      lower.panel=function(x,y){ points(x,y); abline(0, 1, col='red')},
      upper.panel = panel.cor)
dev.off()

tmp <- as.data.frame(cor(data[,-34], as.numeric(data$Age)))
tmp <- tmp %>% rename(cor=V1)
tmp$var <- rownames(tmp)
tmp %>%
  ggplot(aes(reorder(var, cor), cor)) +
  geom_point() +
  coord_flip()

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
# 선형회귀 모형 lm
data_lm_full <- lm(Age ~ ., data=training)
summary(data_lm_full)

predict(data_lm_full, newdata = data[1:5,])

#모형 평가
y_obs <- validation$Age
yhat_lm <- predict(data_lm_full, newdata=validation)
rmse(y_obs, yhat_lm)

#-----------------
# 라쏘 모형 적합
xx <- model.matrix(Age ~ .^2-1, data)
x <- xx[training_idx, ]
y <- training$Age
glimpse(x)

data_cvfit <- cv.glmnet(x, y)
plot(data_cvfit)


coef(data_cvfit, s = c("lambda.1se"))
coef(data_cvfit, s = c("lambda.min"))

predict.cv.glmnet(data_cvfit, s="lambda.min", newx = x[1:5,])

y_obs <- validation$Age
yhat_glmnet <- predict(data_cvfit, s="lambda.min", newx=xx[validate_idx,])
yhat_glmnet <- yhat_glmnet[,1] # change to a vector from [n*1] matrix
rmse(y_obs, yhat_glmnet)

#-----------------
# 나무모형
data_tr <- rpart(Age ~ ., data = training)
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
data_rf <- randomForest(Age ~ ., training)
data_rf

opar <- par(mfrow=c(1,2))
plot(data_rf)
varImpPlot(data_rf)
par(opar)
dev.off()

yhat_rf <- predict(data_rf, newdata=validation)
rmse(y_obs, yhat_rf)
#-----------------
# 부스팅

set.seed(1810)
data_gbm <- gbm(Age ~ ., data=training,
                n.trees=30000, cv.folds=3, verbose = TRUE)

(best_iter = gbm.perf(data_gbm, method="cv"))

yhat_gbm <- predict(data_gbm, n.trees=best_iter, newdata=validation)
rmse(y_obs, yhat_gbm)


# 최종 모형선택과  테스트셋 오차계산
data.frame(lm = rmse(y_obs, yhat_lm),
           glmnet = rmse(y_obs, yhat_glmnet),
           rf = rmse(y_obs, yhat_rf),
           gbm = rmse(y_obs, yhat_gbm)) %>%
  reshape2::melt(value.name = 'rmse', variable.name = 'method')

rmse(test$Age, predict(data_rf, newdata = test))


boxplot(list(lm = y_obs-yhat_lm,
             glmnet = y_obs-yhat_glmnet,
             rf = y_obs-yhat_rf,
             gbm = y_obs-yhat_gbm), ylab="Error in Validation Set")
abline(h=0, lty=2, col='blue')
dev.off()

pairs(data.frame(y_obs=y_obs,
                 yhat_lm=yhat_lm,
                 yhat_glmnet=c(yhat_glmnet),
                 yhat_rf=yhat_rf,
                 yhat_gbm=yhat_gbm),
      lower.panel=function(x,y){ points(x,y); abline(0, 1, col='red')},
      upper.panel = panel.cor)
dev.off()
