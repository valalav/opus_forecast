#Очищаем рабочее пространство
rm(list = ls())
Sys.setLanguage("en")

#Загрузка пакетов
library(xlsx) #экспорт данных в книгу Excel
library(readxl) #импорт данных Excel: команда read_excel
library(dplyr) #обработка данных
library(xts) #формат данных для работы с временными рядами
library(lubridate) #работа с датами: функция ymd()
library(RJDemetra) #сезонное сглаживание, алгоритмы JDemetra

# ------------------------------------------------------------------------------------------------------
# ОПРЕДЕЛЕНИЕ КОНСТАНТ
#Дата актульного значения ИПЦ
smpl_end_date <- "2025-04-01"

#Конечная дата прогнозной траектории
fc_end_date <- "2026-12-01"


# ------------------------------------------------------------------------------------------------------
# ОПРЕДЕЛЕНИЕ ФУНКЦИЙ

# 1) Функция приведения к базисному индексу
# Параметры: mydata - таблица данных, 
#где первые 12 строк - накопительное произведение индексов ММ,
# последующие строки индексы ГГ;
# myvar - имя переменной.
f.calc_base <- function(mydata,myvar) {
  for (i in 13:nrow(mydata)) {
    myvar[i] <- myvar[i] * myvar[i - 12] / 100
  }
  return(myvar)
}

# 2) Функция - вычисление длины прогноза Деметры
# Параметры: mydata - таблица с базисными индексами,
# myvar - имя переменной (вводится в кавычках).
f.JD.FcLength <- function(mydata,myvar){
  k_JD_fc <- (as.yearmon(fc_end_date) -
                as.yearmon(time(mydata)[length(mydata[,myvar][!is.na(mydata[,myvar])])])) * 12 # При подсчёте длины пропускаем NA
  return(k_JD_fc)
}


# 3) Функция: создаёт сезонно сглаженный ряд (в соотвесвии с заданной спецификацией сглаживания), 
# который включает фактическую и прогнозную траекторию.
# Параметры: data_for_SA - таблица с базисными индексами для сезонного сглаживания,
# myvar - имя переменной (в текстовом формате, вводится в кавычках).
f.SA <- function(data_for_SA, varname) {
  series.jd <- x13(data_for_SA[, varname], 
                   spec = myspec_list[[varname]]) #Получили массив данных оценок Demetra
  #Объединяем сглаженный ряд-факт (SA) с со сглаженным прогнозом ряда (SA_f)
  series.base.SA <- ts(c(series.jd[["final"]][["series"]][, "sa"], 
                         series.jd[["final"]][["forecasts"]][, "sa_f"]), 
                       start = as.Date(data_for_SA)[1],
                       frequency = 12)
  return(series.base.SA)
}


# --------------------------------------------------------------------------------------------------------------
# ИМПОРТ И ПОДГОТОВКА ДАННЫХ

# Индексы ММ:
# Убираем лишные переменные под спецификацию, 
# отдельные переменные заменяем на реальные -> корректируем на ИПЦ
dt_mom <- read_excel("Data.xlsx", sheet = 1, skip = 11) %>%
          dplyr::select(!c(ort_prod, ort_neprod, ipp_obrab, ipp_dobycha) &
                          !contains("agro")) %>%
          mutate(
            wage = wage / ipc * 100,
            dep_ind = dep_ind / ipc * 100,
            dolg_potreb = dolg_potreb / ipc * 100,
            dolg_ijk = dolg_ijk / ipc * 100,
            budget = budget / ipc * 100
          ) 

# Индексы ГГ:
# Убираем лишние переменные под спецификацию, 
# отдельные переменные заменяем на реальные -> корректируем на ИПЦ
dt_yoy <- read_excel("Data.xlsx", sheet = 2, skip = 11) %>%
          dplyr::select(!c(ort_prod, ort_neprod, ipp_obrab, ipp_dobycha) &
                          !contains("agro")) %>%
          mutate(
            wage = wage / ipc * 100,
            dep_ind = dep_ind / ipc * 100,
            dolg_potreb = dolg_potreb / ipc * 100,
            dolg_ijk = dolg_ijk / ipc * 100,
            budget = budget / ipc * 100
          ) %>% dplyr::select(!contains("ipc"))

# Переходим к вычислению базисных индексов: 
# первые 12 наблюдений вычисляются как произведение индексов ММ.
# последующие наблюдения домножаются на индексы ГГ.
# В начале создаём произведение первых 12 месячных индексов

# При этом исключаем usd, reer, ruonia, индексы ИПЦ их добавим потом
dt_base <- dt_mom %>% dplyr::select(!c(date, usd, reer, ruonia) & 
                               !contains("ipc")) %>% 
           apply(
                MARGIN = 2,
                FUN = function(x)
                  cumprod(x / 100) * 100) %>% 
           cbind(dplyr::select(dt_mom, date), .) %>%   #Точка "." передаёт результат конвейра в функцию cbind
           slice_head(n = 12) %>% #Берём первые 12 точек
           rbind(dt_yoy[-(1:12),]) %>% # К набору-произведение 12 индексов ММ добавляем ГГ индексы для последующих (13, 14,...) наблюдений
           dplyr::select(!date) %>% 
           apply( #пересчитываем ряд по заданной функции 
             MARGIN = 2,
             FUN = f.calc_base,
             mydata = dt_yoy) %>% 
          cbind(dplyr::select(dt_mom,date),.) 


# Добавим теперь базисные индексы для usd, reer, ruonia, индексы ИПЦ
# Но они будут рассчитываться как просто произведение индексов ММ
dt_base <- dt_mom %>% dplyr::select(usd,reer,ruonia,contains("ipc")) %>% 
           apply( #Применяем функцию накопительного произведения для этих индексов ММ
             MARGIN = 2,
             FUN = function(x)
               cumprod(x/100)*100) %>% 
           cbind(dt_base,.) %>%  #К базисным индексам на основе ГГ добавляем полученные базисные индексы на основе ММ
            dplyr::select(!date) %>% relocate(ipc, ipc_prod, ipc_neprod, ipc_uslugi) %>% 
  xts(., order.by = as.yearmon(dt_mom$date))      

write.xlsx(as.data.frame(dt_base), "dt_base.xlsx")

# ---------------------------------------------------------------------------------
# СЕЗОННОЕ СГЛАЖИВАНИЕ ПО АЛГОРИТМАМ JDEMETRA

# Спецификации сезонного сглаживания
myspec_list <- list(
  ipc = x13_spec(
    spec = "RSA5c",
    outlier.enabled = TRUE,
    outlier.ao = TRUE,
    outlier.tc = FALSE,
    outlier.ls = TRUE,
    outlier.so = FALSE,
    x11.fcasts = f.JD.FcLength(dt_base, "ipc")
  ),
  ipc_prod = x13_spec(
    spec = "RSA5c",
    outlier.enabled = TRUE,
    outlier.ao = TRUE,
    outlier.tc = FALSE,
    outlier.ls = TRUE,
    outlier.so = FALSE,
    x11.fcasts = f.JD.FcLength(dt_base, "ipc_prod")
  ),
  ipc_neprod = x13_spec(
    spec = "RSA5c",
    outlier.enabled = TRUE,
    outlier.ao = TRUE,
    outlier.tc = FALSE,
    outlier.ls = TRUE,
    outlier.so = FALSE,
    x11.fcasts = f.JD.FcLength(dt_base, "ipc_neprod")
  ),
  ipc_uslugi = x13_spec(
    spec = "RSA5c",
    outlier.enabled = TRUE,
    outlier.ao = TRUE,
    outlier.tc = FALSE,
    outlier.ls = TRUE,
    outlier.so = FALSE,
    x11.fcasts = f.JD.FcLength(dt_base, "ipc_uslugi")
  ),
  ort = x13_spec(
    spec = "RSA5c",
    outlier.enabled = TRUE,
    outlier.ao = TRUE,
    outlier.tc = FALSE,
    outlier.ls = TRUE,
    outlier.so = FALSE,
    x11.fcasts = f.JD.FcLength(dt_base, "ort")
  ),
  uslugi = x13_spec(
    spec = "RSA5c",
    outlier.enabled = TRUE,
    outlier.ao = TRUE,
    outlier.tc = FALSE,
    outlier.ls = TRUE,
    outlier.so = FALSE,
    x11.fcasts = f.JD.FcLength(dt_base, "uslugi")
  ),
  oop = x13_spec(
    spec = "RSA5c",
    outlier.enabled = TRUE,
    outlier.ao = TRUE,
    outlier.tc = FALSE,
    outlier.ls = TRUE,
    outlier.so = FALSE,
    x11.fcasts = f.JD.FcLength(dt_base, "oop")
  ),
  wage = x13_spec(
    spec = "RSA5c",
    outlier.enabled = TRUE,
    outlier.ao = TRUE,
    outlier.tc = FALSE,
    outlier.ls = TRUE,
    outlier.so = FALSE,
    x11.fcasts = f.JD.FcLength(dt_base, "wage")
  ),
  ipp = x13_spec(
    spec = "RSA5c",
    outlier.enabled = TRUE,
    outlier.ao = TRUE,
    outlier.tc = FALSE,
    outlier.ls = TRUE,
    outlier.so = FALSE,
    x11.fcasts = f.JD.FcLength(dt_base, "ipp")
  ),
  stroika = x13_spec(
    spec = "RSA5c",
    outlier.enabled = TRUE,
    outlier.ao = TRUE,
    outlier.tc = FALSE,
    outlier.ls = TRUE,
    outlier.so = FALSE,
    x11.fcasts = f.JD.FcLength(dt_base, "stroika")
  ),
  prom_price_food = x13_spec(
    spec = "RSA5c",
    outlier.enabled = TRUE,
    outlier.ao = TRUE,
    outlier.tc = FALSE,
    outlier.ls = TRUE,
    outlier.so = FALSE,
    x11.fcasts = f.JD.FcLength(dt_base, "prom_price_food")
  ),
  gruz_price = x13_spec(
    spec = "RSA5c",
    outlier.enabled = TRUE,
    outlier.ao = TRUE,
    outlier.tc = FALSE,
    outlier.ls = TRUE,
    outlier.so = FALSE,
    x11.fcasts = f.JD.FcLength(dt_base, "gruz_price")
  ),
  dep_ind = x13_spec(
    spec = "RSA5c",
    outlier.enabled = TRUE,
    outlier.ao = TRUE,
    outlier.tc = FALSE,
    outlier.ls = TRUE,
    outlier.so = FALSE,
    x11.fcasts = f.JD.FcLength(dt_base, "dep_ind")
  ),
  dolg_potreb = x13_spec(
    spec = "RSA5c",
    outlier.enabled = TRUE,
    outlier.ao = TRUE,
    outlier.tc = FALSE,
    outlier.ls = TRUE,
    outlier.so = FALSE,
    x11.fcasts = f.JD.FcLength(dt_base, "dolg_potreb")
  ),
  dolg_ijk = x13_spec(
    spec = "RSA5c",
    outlier.enabled = TRUE,
    outlier.ao = TRUE,
    outlier.tc = FALSE,
    outlier.ls = TRUE,
    outlier.so = FALSE,
    x11.fcasts = f.JD.FcLength(dt_base, "dolg_ijk")
  ),
  budget = x13_spec(
    spec = "RSA5c",
    outlier.enabled = TRUE,
    outlier.ao = TRUE,
    outlier.tc = FALSE,
    outlier.ls = TRUE,
    outlier.so = FALSE,
    x11.fcasts = f.JD.FcLength(dt_base, "budget")
  ),
  usd = x13_spec(
    spec = "RSA5c",
    outlier.enabled = TRUE,
    outlier.ao = TRUE,
    outlier.tc = FALSE,
    outlier.ls = TRUE,
    outlier.so = FALSE,
    x11.fcasts = f.JD.FcLength(dt_base, "usd")
  ),
  reer = x13_spec(
    spec = "RSA5c",
    outlier.enabled = TRUE,
    outlier.ao = TRUE,
    outlier.tc = FALSE,
    outlier.ls = TRUE,
    outlier.so = FALSE,
    x11.fcasts = f.JD.FcLength(dt_base, "reer")
  ),
  ruonia = x13_spec(
    spec = "RSA5c",
    outlier.enabled = TRUE,
    outlier.ao = TRUE,
    outlier.tc = FALSE,
    outlier.ls = TRUE,
    outlier.so = FALSE,
    x11.fcasts = f.JD.FcLength(dt_base, "ruonia")
  )
)


#Формируем таблицу с сезонно сглаженными рядами (SA)
# data_for_sa <- dt_base
dt_base_SA <- sapply(colnames(dt_base), #подаём для сглаживания названия столбцов
                     f.SA, #применяем пользовательскую функцию сглаживания
                     data_for_SA = as.ts(dt_base,#дополнительно указываем откуда брать данные для сглаживания
                                         start = start(dt_base))) %>% #нужно обязательно указывать start, иначе неверно формируется временной ряд
                     xts(., 
                         order.by = as.yearmon(seq(as.Date(dt_mom$date[1]), ymd(fc_end_date), 
                         by = 'months'
                      ))) # формируем объект xts

write.xlsx(as.data.frame(dt_base_SA), "dt_base_SA.xlsx")


# Вычисляем dt_mom_SA
  dt_mom_SA <- rbind(dt_base_SA[1, ], 
                     na.omit(dt_base_SA / stats::lag(dt_base_SA, k = 1) 
                             * 100)) %>% 
                xts(., order.by = zoo::index(dt_base_SA)) #если подгрузить пакет tsibble будет конфликт с командой index
  

# Сразу добавляем фиктивные переменные на периоды
  dt_mom_SA$m201412 <-  if_else(index(dt_mom_SA) == as.yearmon("2014-12-01"), 1, 0)
  dt_mom_SA$m201412_15 <- if_else(index(dt_mom_SA) %in% as.yearmon(c("2014-12-01", "2015-01-01")), 1, 0)
  dt_mom_SA$m201501 <-  if_else(index(dt_mom_SA) == as.yearmon("2015-01-01"), 1, 0)
  dt_mom_SA$m201707 <-  if_else(index(dt_mom_SA) == as.yearmon("2017-07-01"), 1, 0)
  dt_mom_SA$m202203 <-  if_else(index(dt_mom_SA) == as.yearmon("2022-03-01"), 1, 0)
  dt_mom_SA$m202204 <-  if_else(index(dt_mom_SA) == as.yearmon("2022-04-01"), 1, 0)


  # Создаём таблицу фактических значений
dt_mom_SA_fact <- dt_mom_SA %>%
    subset(index(.) <= smpl_end_date)
    
# Актуализируем dt_mom_SA: не сглаживаем (usd,reer,ruonia), 
# восстанавливаем их значения из dt_mom  
dt_mom_SA_fact$usd <- dt_mom$usd
dt_mom_SA_fact$reer <- dt_mom$reer
dt_mom_SA_fact$ruonia <- dt_mom$ruonia

# Экспорт результатов сглаживания в Excel
write.xlsx(dt_mom_SA_fact, "dt_mom_SA_fact.xlsx")

