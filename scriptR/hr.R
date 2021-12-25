#==========================================Installation des bibliothèques========================================

library(dplyr)          #pour manipuler les données
library(ggplot2)        #pour les plots
library(caTools)        #pour calculer l'AUC
library(RColorBrewer)   #pour le coloriage des figures
library(rpart.plot)     #pour tracer les arbre rpart
library(ellipse)        #traçer des ellipses pour les modèles linéaires, linéaires généralisés et éventuellement d'autres modèles.
library(car)            #pour faire le Machine Learning
library(faraway)        #pour la regression pratique et ANOVA
library(ROCR)           #pour créer des courbes de performance en 2D 

#=========================================Chargement des données dans R===========================================

dataHR <- read.csv("C:/dataHR.csv")
str(dataHR)
dataHR

#==============================================Summarisation======================================================
summary(dataHR)

dataHR = filter(dataHR, dataHR$role == "Ind" | dataHR$role == "Manager" | dataHR$role == "Director")
dataHR$role <- factor(dataHR$role)

summary(dataHR)


#================================================Missing values===================================================
which(is.na(dataHR)) #--> pas de valeurs manquantes



#===========================================Analyse & Visualisation===============================================

#+++++++++++++++++++++Performance vs Départ volontaire++++++++++++++++++++++++++
agg_perf = aggregate(vol_leave ~ perf, data = dataHR, mean)
agg_perf

ggplot(agg_perf, aes(x = perf, y = vol_leave)) + geom_bar(stat ="identity", fill = 'purple', colour = 'black') + ggtitle("Performance v/s Départ volontaire") + labs(y = "Départ volontaire", x ="Évaluation de performance")

#++++++++++++++++++Départ volontaire vs sexe++++++++++++++++++++++++++++++++++++
agg_sex = aggregate(vol_leave ~ sex, data = dataHR, mean)
agg_sex

ggplot(agg_sex, aes(x = sex, y = vol_leave)) + geom_bar(stat = "identity",fill = 'red', colour = 'black') + ggtitle("Sexe vs Départ volontaire") + labs(y = "Quitter", x = "Sexe")

#++++++++++++++++++Départ volontaire vs département du travail++++++++++++++++++
agg_area = aggregate(vol_leave ~ area, data = dataHR, mean)
agg_area

ggplot(agg_area, aes(x = area, y = vol_leave, fill = area)) + geom_bar(stat =
"identity", colour = "black") + scale_fill_brewer() + ggtitle("Département du travail vs Départ volontaire") + labs(y = "Quitter", x = "Département")


#++++++++++++++++++Départ volontaire vs Role++++++++++++++++++++++++++++++++++++
agg_role = aggregate(vol_leave ~ role, data = dataHR, mean)
agg_role

ggplot(agg_role, aes(x = role, y = vol_leave, fill = role)) + geom_bar(stat ="identity", width = .7, 
colour = 'black') + scale_fill_brewer() + ggtitle("Role vs Départ volontaire") + labs (y = "Quitter", x= "Role")


#+++++++++++++++++++++++++Départ volontaire vs Age++++++++++++++++++++++++++++++
hist(dataHR$age, breaks = 100, main = "Distribution par age", border = F,
     xlab = "Age", col = 'orange')

quantile(dataHR$age, probs = seq(0,1,.1))

library(e1071)
skewness(dataHR$age)

dataHR$log_age = log(dataHR$age)
summary(dataHR$log_age)

boxplot(age ~ role, data = dataHR, col = 'pink', xlab = "Role dans la société",
        ylab = "Age du salarié", main = 'Distribution par Age en termes de Role')

#agréger la variable d'âge pour voir la relation avec le départ des employés
agg_age = aggregate(x = dataHR$vol_leave, by = list(cut(dataHR$age, 10)), mean)
agg_age

names(agg_age) = c("Age", "Probabilité")
ggplot(agg_age, aes(x = Age, y = Probabilité, fill = Age)) + geom_bar(stat =
        "identity", width = .1, colour = 'black') + scale_fill_brewer() +
        ggtitle("Age vs Départ volontaire") + labs(y = "Quitter", x = "Age")


#++++++++++++++++++Départ volontaire vs Salaire+++++++++++++++++++++++++++++++++
summary(dataHR$salary)
quantile(dataHR$salary, probs = seq(0,1,.2))

hist(dataHR$salary, breaks = 50, col = 'turquoise', main = "Analyse de la Variable salaire", xlab = "Salaire")


#=================================================Modèles=====================================================

#--------------------Diviser le dataset en 2 (test & apprentissage)-------------------------------------
set.seed(42)
div_dataHR = sample.split(dataHR$vol_leave, 2/3)
train = dataHR[div_dataHR,]
test = dataHR[!div_dataHR,]


#-----------------------------------Régression Logistique------------------------------------------------
test_mean = mean(test$vol_leave)
train_mean = mean(train$vol_leave)
print(c(test_mean, train_mean))

#Ajustement du modèle en utilisant GLM
ajst = glm(vol_leave ~ role + perf + area + sex + log_age + salary, data= dataHR, family = 'binomial')
summary(ajst)

#Analyser la deviance
anova(ajst, test = "Chisq")

#Analyser la capacité prédictive du modèle 
pred_model = predict(fit, test, type = 'response')
pred_model = ifelse(pred_model > 0.5,1,0)
MCE = mean(pred_model != test$vol_leave)

table(actual = test$vol_leave, prediction = pred_model)

#Calcul de la précision du modèle
print(paste('Accuracy', 1 - MCE))

#Courbe ROC
plot1 = predict(ajst, test, type = "response")
plot2 = prediction(plot1, test$vol_leave)
plot3 = performance(plot2, measure = "tpr", x.measure = "fpr")
plot(plot3)

#Calculer AUC 
AUC = performance(plot2, measure = "auc")
AUC = AUC@y.values[[1]]
AUC


#-----------------------------------Arbre de décision------------------------------------------------
#Ajuster le modèle
set.seed(42)
dt = rpart(vol_leave ~ role + perf + age + sex + area + salary, data = train, method = "class")
dt

#Tracer l'arbre de décision
library(rattle)
par(mar = c(5,4,1,2))
fancyRpartPlot(dt, sub = NULL, main = "Arbre de déciion de base")

#Analyser la capacité prédictive du modèle en utilisant la matrice de confusion
pred_dt = predict(dt, test, type = 'class')
cm_dt = table(actual = test$vol_leave, prediction = pred_dt)
cm_dt

#Calculer l'accuracy du modèle
accuracy = sum(diag(cm_dt))/sum(cm_dt)
accuracy

