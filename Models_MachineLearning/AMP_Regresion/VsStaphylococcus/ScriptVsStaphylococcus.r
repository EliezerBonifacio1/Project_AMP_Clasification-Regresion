# -*- coding: utf-8 -*-
# Librerias a instalar
list.of.packages <- c("tidyverse",  "readxl", "xlsx",
                      "recipes", "rstatix",
                      "caret", "caretEnsemble", "doParallel",
                      "pls", "elasticnet", "kernlab",
                      "randomForest", "klaR", "MASS", "gbm","svglite")

# Revisa e instala librerias faltantes
new.packages <- list.of.packages[!(list.of.packages %in% installed.packages()[,"Package"])]
if(length(new.packages)) install.packages(new.packages)

# Cargar librerias
packages_load <- c("tidyverse",  "readxl", "recipes", "rstatix",
                   "caret", "caretEnsemble", "doParallel", "gbm")

for (i in packages_load) {
  library(i, character.only  = TRUE)
}

#Cargar los datos
input <- read_excel("DataSet_AntimicrobialPeptides_Descriptors.xlsx")

# +
t0 = now()
print(" ################ LOADED DATA ###############")

# Filtro inicio
input_selected <- input %>%
  rename(logMIC = "Mesure_(log)") %>%
  rename(hoopwoods_1_mean = "hopp-woods_1_mean") %>%
  filter(Cuantification_Type == "MIC" &
           Genero =="Staphylococcus" &
           Unity == "µg/ml"  &
           is.na(Notes))
# -

# Delete NA values in logMIC and filter Unity
input_naomit <- input_selected %>%
  mutate(logMIC = as.numeric(logMIC)) %>%
  drop_na(logMIC)

# Eliminar duplicados
input_naomit_out_dup <- input_naomit %>%
  distinct(Sequences,.keep_all = TRUE) 

input_naomit_out_dup

ggpubr::gghistogram(input_naomit_out_dup$logMIC)

# Remove outlier in general
input_naomit_out_dup <-  input_naomit_out_dup %>%
  rstatix::identify_outliers(logMIC) %>%
  anti_join(input_naomit_out_dup,.)


# print(table(input$Genero))
# ##############

set.seed(321)
index <- createDataPartition(input_naomit_out_dup$logMIC,
                             p = .8, list = FALSE)

train_input <- input_naomit_out_dup[ index,]
test_input  <- input_naomit_out_dup[-index,]

# Check distributions
train_input %>% mutate(set = "Train") %>%
  rbind(test_input %>% mutate(set = "Test")) %>%
  ggplot(aes(logMIC, color = set)) +
  geom_density()


# Seleccionar variables de informacion
train_info <- train_input %>%
  select(Data_ID:References)
test_info <- test_input %>%
  select(Data_ID:References)


# Seleccionar variable respuesta y descriptores
train_set <- train_input %>%
  select(logMIC, Y:KGA)

test_set <- test_input %>%
  select(logMIC, Y:KGA)


# PREPROCESAMIENTO

rec_train <- recipe(logMIC ~ . , data = train_set) %>%
  step_nzv(all_predictors()) %>%
  step_corr(all_numeric_predictors(), threshold = 0.85) %>%
  #step_normalize(all_numeric_predictors()) %>%
  step_scale(all_predictors()) %>%
  prep()

train_prep <- juice(rec_train)
test_prep <- bake(rec_train, new_data = test_input)

save(rec_train, file = "InicialPreprocessing_algoritm.rda")
save(train_prep, file = "train_prep.rda")
save(test_prep, file = "test_prep.rda")


# #######         SELECCION DE CARACTERISTICAS

# Cargar librerias
packages_load <- c("caret", "caretEnsemble", "doParallel", "gbm",
                   "dplyr")

for (i in packages_load) {
  library(i, character.only  = TRUE)
}

# Usar varios nucleos
no_cores <- detectCores()

cl <- makePSOCKcluster(no_cores, outfile = "modeling.log")
registerDoParallel(cl)

# Train control (cross validation)
rf_rfe_ctrl <- rfeControl(functions = rfFuncs,  #lmFuncs , rfFuncs
                          method = "repeatedcv",
                          number = 2,
                          repeats = 1,
                          verbose = TRUE,
                          allowParallel = TRUE)

set.seed(321)
t1 = now()

print("RUNNING RECURSIVE FEATURES SELECTION")
rf_fs <- rfe(logMIC ~., data = train_prep,
             sizes = seq(from=1,to=200, 10),
             rfeControl = rf_rfe_ctrl)
t2 = now()
print(t2-t1)


rf_fs

save(rf_fs, file = "rf_fs.rda")

# Plot
plot(rf_fs)

svglite::svglite(filename = "rfe_reg.svg",
                 width = 8, height = 6)
plot(rf_fs)
dev.off()

t3 = now()
print(t3-t0)

# SELECCION DE VARIABLES : CRITERIO (N= 75 mejores)

x_rfe <- train_prep %>% select(predictors(rf_fs)[1:75])
y_rfe <- train_prep$logMIC


# ---------------------------------------------------------------------
# ALGORITMO GENETICO


print("RUNNING GENETIC ALGORITHM")

t1=now()
rf_ga_ctrl <- gafsControl(functions = rfGA, method = "repeatedcv",
                          number = 3,
                          repeats = 1, 
                          verbose = TRUE,
                          genParallel = TRUE,
                          allowParallel = TRUE)

tr_ctrl <- trainControl(method = "cv", number = 2,
                        allowParallel = TRUE,
                        verboseIter	= TRUE)
set.seed(321)

fs_ga <- gafs(x = x_rfe,
              y = y_rfe,
              iters = 200,
              gafsControl = rf_ga_ctrl,
              trControl = tr_ctrl,
              popSize = 20
              
)
t2=now()
t2-t1

fs_ga

save(fs_ga, file = "fs_ga.rda")

# El grafico muestra en el eje x: la generacion y, en el eje y: la meadia del RMSE.
plot(fs_ga)

svglite::svglite(filename = "plt_fs_ga.svg",
                 width = 8, height = 6)
plot(fs_ga)
dev.off()



# ---------------------------------------------------------------------
# MODELOS


# NUEVO ALGORITMO DE PREPROCESAMIENTO CON MENOS VARIABLES (los valores no cambian)

train_set2 <- train_set %>%
  select(logMIC,fs_ga$optVariables) # Seleccionando las mejores variables segun AG

rec_train2 <- recipe( ~ . , data = train_set2) %>%
  #step_nzv(all_predictors()) %>%
  #step_corr(all_numeric_predictors(), threshold = 0.85) %>%
  #step_normalize(all_numeric_predictors()) %>%
  step_scale(all_predictors()) %>%
  prep()

train_prep2 <- juice(rec_train2)

save(rec_train2, file = "FinalPreprocessing_algoritm.rda")


# List of models and hyperparameters
mod_tune <- 20
# Methods
models <- list(
  lm = caretModelSpec(method = "lm"),
  pls = caretModelSpec(method = "pls", tuneLength = 15),
  svmLinear = caretModelSpec(method = "svmLinear", tuneLength = 10),
  svmRadial = caretModelSpec(method = "svmRadial", tuneLength = 10),
  svmPoly = caretModelSpec(method = "svmPoly", tuneLength = 2),
  rf = caretModelSpec(method = "rf", tuneLength = 20),
  gbm = caretModelSpec(method = "gbm")
)

# Cross validation paramters
outcome <- "logMIC"
y_vector <- train_prep %>% as.data.frame() %>% .[,outcome]

set.seed(123)
fivecv_5_multi <- trainControl(method = "repeatedcv",
                               number = 2,
                               repeats = 1,
                               verboseIter = TRUE,
                               allowParallel = TRUE,
                               savePredictions = "final",
                               index =
                                 createMultiFolds(y_vector, 2, 1))


# Final dataset
Train_fs_logMic <- train_prep2 %>% select(logMIC,
                                         fs_ga$optVariables) %>% as.data.frame()


# Regression logMIC
models_logMIC <- caretList(logMIC~., data = Train_fs_logMic,
                           trControl = fivecv_5_multi,
                           tuneList = models,
                           continue_on_fail = TRUE)

save(models_logMIC, file = "models_logMIC.rda")


#_-------------------------------------------
#METRICAS

get_metric <- function(result_models, train_fs, outcome){
  models_sel <- result_models %>% names()
  
  tr_metric <- result_models %>%
    .[models_sel] %>%
    map(predict, train_fs) %>%
    map(postResample, train_fs[,outcome]) %>%
    bind_rows(.id = "model") %>%
    mutate(set = "Train")
  
  ts_metric <- result_models %>%
    .[models_sel] %>%
    map(predict, test_prep) %>%
    map(postResample, test_prep %>% pull(outcome)) %>%
    bind_rows(.id = "model") %>%
    mutate(set = "Test")
  
  if(outcome == "logMIC") {
    cv_metric <- result_models %>%
      .[models_sel] %>%
      map(pluck, "resample") %>%
      map(summarise, RMSE = mean(RMSE),
          Rsquared = mean(Rsquared), MAE = mean(MAE)) %>%
      bind_rows(., .id = "model") %>% mutate(set = "cv")
  }
  
  if(outcome != "logMIC") {
    cv_metric <- result_models %>%
      .[models_sel] %>%
      map(pluck, "resample") %>%
      map(summarise, Accuracy = mean(Accuracy),
          Kappa = mean(Kappa)) %>%
      bind_rows(., .id = "model") %>% mutate(set = "cv")
  }
  metric_models <- rbind(tr_metric, cv_metric, ts_metric) %>%   arrange(model)
  
  
}

metric_logMIC <- get_metric(models_logMIC, Train_fs_logMic, "logMIC")
print(metric_logMIC,n=20)


xlsx::write.xlsx(metric_logMIC,file='metricslogMIC.xlsx')



#_-------------------------------------------
#DISPERSION GRAFICOS

# Tema personalizado
theme_Publication <- function(base_size=14, base_family="sans") {
  library(grid)
  library(ggthemes)
  (theme_foundation(base_size=base_size, base_family=base_family)
    + theme(plot.title = element_text(face = "bold",
                                      size = rel(1.2), hjust = 0.5, margin = margin(0,0,20,0)),
            text = element_text(),
            panel.background = element_rect(colour = NA),
            plot.background = element_rect(colour = NA),
            panel.border = element_rect(colour = NA),
            axis.title = element_text(face = "bold",size = rel(1)),
            axis.title.y = element_text(angle=90,vjust =2),
            axis.title.x = element_text(vjust = -0.2),
            axis.text = element_text(),
            axis.line.x = element_line(colour="black"),
            axis.line.y = element_line(colour="black"),
            axis.ticks = element_line(),
            panel.grid.major = element_line(colour="#f0f0f0"),
            panel.grid.minor = element_blank(),
            legend.key = element_rect(colour = NA),
            legend.position = "bottom",
            legend.direction = "horizontal",
            legend.box = "vetical",
            legend.key.size= unit(0.5, "cm"),
            #legend.margin = unit(0, "cm"),
            legend.title = element_text(face="italic"),
            # plot.margin=unit(c(10,5,5,5),"mm"),
            strip.background=element_rect(colour="#f0f0f0",fill="#f0f0f0"),
            strip.text = element_text(face="bold")
    ))
  
}

Train_pred <- models_logMIC %>%
  # .[models_sel] %>%
  map(predict, train_prep) %>%
  bind_cols(set = "Train") %>%
  mutate(logMIC_exp = train_prep$logMIC)

Test_pred <- models_logMIC %>%
  # .[models_sel] %>%
  map(predict, test_prep) %>%
  bind_cols(set = "Test") %>%
  mutate(logMIC_exp = test_prep$logMIC)


set_pred <- rbind(Train_pred, Test_pred) %>%
  # na.omit() %>%
  gather(key = "model", value = "logMIC_pred", -set, -logMIC_exp)

# Cambiar nombre al titulo de cada modelo (opcional)
model_names <- c(
  `lm` = "MLR",
  `pls` = "PLS",
  `svmPoly` = "SVM-Poly",
  `svmLinear` = "SVM-Linear",
  `rf` = "RF",
  `gbm` = "GBM",
  `svmRadial` = "SVM-Rad")


# cambiar el orden de los modelos en el grafico (opcional)
# lev_model <- c('lm','pls','svmRadial',
#                'rf', 'gbm')

models_sel <- names(models_logMIC)
lev_model <- models_sel
# Run plot
plt_scatter <- set_pred %>%
  ggplot(aes(logMIC_exp, logMIC_pred, color = set)) +
  geom_point(size = 1.0, alpha = 0.45, stroke = 0) +
  geom_abline(size = 0.3) +
  facet_wrap( ~ factor(model, levels = lev_model),
              labeller = as_labeller(model_names)
  ) +
  
  labs(x = "Experimental", y = "Predicted",
       color = NULL) +
  theme_Publication()


svglite::svglite(filename = "scatter_aLL.svg",
                 width = 8, height = 8)
plt_scatter
dev.off()



#SECOND PLOT

# Cargar la biblioteca ggplot2
library(ggplot2)

# Datos de ejemplo
x <- set_pred %>% filter(set=="Test", model == "rf") %>% select(logMIC_exp)
y <- set_pred %>% filter(set=="Test", model == "rf") %>% select(logMIC_pred)


# Crear un data frame con los datos
data <- data.frame(x, y)


ml = lm(logMIC_pred~logMIC_exp, data = data) 
# Extracting R-squared parameter from summary 
r2 <- summary(ml)$r.squared 

# Crear el gráfico de dispersión
scatter_plot <- ggplot(data, aes(x = logMIC_pred , y = logMIC_exp)) +
  geom_point(aes(color = logMIC_pred), size = 3) + # Colorear por valores de "y"
  labs( x = "LogMIC Predicted", y = "LogMIC Experimental") +
  scale_color_gradient(low = "yellow", high = "red",name= "LogMIC \nPredicho \npor RF") + # Colores de amarillo a rojo
  geom_smooth(method=lm , color="#85929E", fill="#AEB6BF", se=T) +
  geom_text(x = 0.75, y = 3, label = paste("R² =", round(r2, 3)))+
  theme_minimal()

# Mostrar el gráfico
scatter_plot

svglite::svglite(filename = "Scatterplot3.svg",
                 width = 8, height = 7)
plot(scatter_plot)
dev.off()




#IMPORTANCIA DE LAS VARIABLES

# Variable importance ------
get_imp <- function(model){
  require(caret)
  result <- varImp(model) %>%
    .$importance %>%
    tibble::rownames_to_column("variables") %>%
    # arrange(desc(Overall)) %>%
    as_tibble()
}

# List with variable importance
varimp_logMIC <- models_logMIC %>%
  map(get_imp)



# Clean file
save_varimp <- function(varimp_list, file) {
  models_sel <- names(varimp_list)
  
  if(file.exists(file)) file.remove(file)
  
  for (i in models_sel) {
    
    xlsx::write.xlsx(varimp_list[[i]], file = file,
                     sheetName = i, append = TRUE)
  }
  
}

save_varimp(varimp_logMIC, file = "varimp_logMIC.xlsx")

help(gafsControl) 
