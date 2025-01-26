#install.packages("betafunctions")
library(betafunctions)
data <- read.csv("C:\\Users\\hehaa\\OneDrive\\git\\BBClassify_Python\\testing_files\\act288m_for_r")
reliability <- 0.87217
head(data)
data = as.numeric(data[, 1])
head(data)
hist(data)

test = HB.CA.MC(data, reliability, c(22, 24, 26), 40, "2", l = 0.223317, u = 1, modelfit = 0)
test

params = list("alpha" = 0.52378, "beta" = 1.62569, "l" = 0.22317, "u" = 1, "k" = 0, "N" = 40)
params

test$parameters

HB.CA.MC(params, reliability, 24, 40, "2")

caStats
barplot(table(data))
unique(data)
