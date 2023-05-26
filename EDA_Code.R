# EDA

# Clean console environment
rm(list = ls())

# Import library
library(janitor)
library(csv)
library(dplyr)
library(plyr)
library(ggplot2)
library(corrplot)
library(RColorBrewer)
library(car)
library(tidyverse)
library(leaps)
library(gvlma)
library(caTools)
library(caret)
library(stats)
library(ISLR)
library(gridExtra)
library(pROC)
library(InformationValue)
library(nortest)
library(ggstatsplot)
library(statsExpressions)
library(insight)
library(parameters)
library(bayestestR)
library(datawizard)
library(zeallot)
library(correlation)
library(paletteer)
library(rematch2)
library(patchwork)
library(performance)
library(prismatic)
library(imputeR)
library(gplots)
library(RColorBrewer)
library(Hmisc)
library(latticeExtra)
library(png)
library(jpeg)
library(interp)
library(deldir)
library(htmlTable)
library(psych)
library(mnormt)
library(kableExtra)
library(svglite)
library(lares)
library(h2o)

# Import dataset
df <- read.csv("C:/Users/Yunan/Desktop/NEU/Aly6040/project/attrition.csv")

# Check the structure of dataset
summary(df)
dim(df)
str(df)
names(df)

# Change the column names
names(df)[1] <- "Age"
df$BusinessTravel <- gsub("Non-Travel","Non_Travel",df$BusinessTravel)

# Remove Empty Rows and Columns of Data
df2 <- remove_empty(df)
df2

# Remove Duplicate Rows of Data
df3 <- distinct(df2)
df3

# Check missing values in numeric variables
df3[df3 == ""]<-NA
colSums(is.na(df3))
which(colSums(is.na(df3))>0)
names(which(colSums(is.na(df3))>0))

# Check distributions and outliers
boxplot(df3$Age, ylab ='Age', main = 'Distribution of Age')$out
boxplot.stats(df3$Age)$out

boxplot(df3$DailyRate, ylab ='Daily Rate', main = 'Distribution of Daily Rate')$out
boxplot.stats(df3$DailyRate)$out

boxplot(df3$DistanceFromHome, ylab ='Distance From Home', main = 'Distribution of Distance From Home')$out
boxplot.stats(df3$DistanceFromHome)$out

boxplot(df3$HourlyRate, ylab ='Hourly Rate', main = 'Distribution of Hourly Rate')$out
boxplot.stats(df3$HourlyRate)$out

boxplot(df3$MonthlyIncome, ylab ='Monthly Income', main = 'Distribution of Monthly Income')$out
boxplot.stats(df3$MonthlyIncome)$out

boxplot(df3$MonthlyRate, ylab ='Monthly Rate', main = 'Distribution of Monthly Rate')$out
boxplot.stats(df3$MonthlyRate)$out

boxplot(df3$NumCompaniesWorked, ylab ='Number of Companies Worked', main = 'Distribution of Number of Companies Worked')$out
boxplot.stats(df3$NumCompaniesWorked)$out

boxplot(df3$TotalWorkingYears, ylab ='Total Working Years', main = 'Distribution of Number of Total Working Years')$out
boxplot.stats(df3$TotalWorkingYears)$out

boxplot(df3$YearsAtCompany, ylab ='Years At Company', main = 'Distribution of Years At Company')$out
boxplot.stats(df3$YearsAtCompany)$out

boxplot(df3$YearsInCurrentRole, ylab ='Number of Years In Current Role', main = 'Distribution of Number of Years In CurrentRole')$out
boxplot.stats(df3$YearsInCurrentRole)$out

boxplot(df3$YearsWithCurrManager, ylab ='Years With Curr Manager', main = 'Distribution of Years With Cur rManager')
boxplot.stats(df3$YearsWithCurrManager)$out

# Remove outliers
df4 <- df3[-df3$MonthlyIncome[df3$MonthlyIncome %in% boxplot.stats(df3$MonthlyIncome)$out],]
df5 <- df4[-df4$NumCompaniesWorked[df4$NumCompaniesWorked %in% boxplot.stats(df4$NumCompaniesWorked)$out],]
df6 <- df5[-df5$TotalWorkingYears[df5$TotalWorkingYears %in% boxplot.stats(df5$TotalWorkingYears)$out],]
df7 <- df6[-df6$YearsAtCompany[df6$YearsAtCompany %in% boxplot.stats(df6$YearsAtCompany)$out],]
df8 <- df7[-df7$YearsInCurrentRole[df7$YearsInCurrentRole %in% boxplot.stats(df7$YearsInCurrentRole)$out],]
clean <- df8[-df8$YearsWithCurrManager[df8$YearsWithCurrManager %in% boxplot.stats(df8$YearsWithCurrManager)$out],]

# Encode categorical values.
str(clean)
unique(clean$Attrition)
unique(clean$BusinessTravel)
unique(clean$Department)
unique(clean$EducationField)
unique(clean$Gender)
unique(clean$JobRole)
unique(clean$MaritalStatus)
unique(clean$Over18)
unique(clean$OverTime)

clean$Attrition <- replace(clean$Attrition , clean$Attrition  == "No", 0)
clean$Attrition <- replace(clean$Attrition , clean$Attrition  == "Yes", 1)
clean$BusinessTravel <- replace(clean$BusinessTravel , clean$BusinessTravel  == "Non_Travel", 1)
clean$BusinessTravel <- replace(clean$BusinessTravel , clean$BusinessTravel  == "Travel_Rarely", 2)
clean$BusinessTravel <- replace(clean$BusinessTravel , clean$BusinessTravel  == "Travel_Frequently", 3)
clean$Gender <- replace(clean$Gender, clean$Gender  == "Female", 0)
clean$Gender <- replace(clean$Gender, clean$Gender  == "Male", 1)
clean$MaritalStatus <- replace(clean$MaritalStatus, clean$MaritalStatus  == "Single", 1)
clean$MaritalStatus <- replace(clean$MaritalStatus, clean$MaritalStatus  == "Married", 2)
clean$MaritalStatus <- replace(clean$MaritalStatus, clean$MaritalStatus  == "Divorced", 3)
clean$OverTime <- replace(clean$OverTime, clean$OverTime  == "No", 0)
clean$OverTime <- replace(clean$OverTime, clean$OverTime  == "Yes", 1)
names(clean)[12] <- "Maleness"
clean <- clean[,-22]
str(clean)

new <- as.data.frame(lapply(clean[,c(2,3,12,18,22)],as.numeric))
clean <- clean[,-c(2,3,12,18,22)]
final <- cbind(new, clean)
str(final)

# define one-hot encoding function
dummy <- dummyVars(" ~ .", data = final)

# perform one-hot encoding
final_df <- data.frame(predict(dummy, newdata = final))
str(final_df)

# Remove useless variables
final_df <- final_df[,-c(19,20,41)]

# Distributions of the variables
hist(final_df$Age)
hist(final_df$MonthlyIncome)
hist(final_df$TotalWorkingYears)

# T2-6: Store the preprocessed dataset into a new CSV file
write.csv(final_df, "C:/Users/Yunan/Desktop/NEU/Aly6040/project/attrition_clean.csv", row.names=FALSE)

#######################################################
# EDA
# Research question 1: Which are the most important factors affect monthly income?
# create correlate matrix and plot
corr <- cor(final_df, use = "pairwise")
corrplot(corr, type = "upper", tl.cex = 0.5)
corrmatrix <- corr.test(final_df, method="pearson")
corrmatrix

for (i in 1:nrow(corr)){
  correlations <-  which((corr[i,] > 0.60) & (corr[i,] != 1))
  
  if(length(correlations)> 0){
    print(colnames(final_df)[i])
    print(correlations)
  }
}

# Distribution of monthly income by department
df3 %>%
  group_by(Department) %>% 
  mutate(avg = mean(MonthlyIncome)) %>%
  ggplot() + 
  geom_boxplot(aes(reorder(Department, avg), MonthlyIncome, fill = avg)) + 
  coord_flip() + theme_minimal() + 
  scale_fill_continuous(low = '#ffffcc', high = '#fc4e2a', name = "Average Salary Level") + 
  labs(x = 'Department', y = 'MonthlyIncome')+
  expand_limits(y = c(0,5))

# Distribution of monthly income by education level
df3 %>%
  group_by(Education) %>% 
  mutate(avg = mean(MonthlyIncome)) %>%
  ggplot() + 
  geom_boxplot(aes(reorder(Education, avg), MonthlyIncome, fill = avg)) + 
  coord_flip() + theme_minimal() + 
  scale_fill_continuous(low = '#ffffcc', high = '#fc4e2a', name = "Average Salary Level") + 
  labs(x = 'Education Level', y = 'MonthlyIncome')+
  expand_limits(y = c(0,5))

# Distribution of monthly income by job level
df3 %>%
  group_by(JobLevel) %>% 
  mutate(avg = mean(MonthlyIncome)) %>%
  ggplot() + 
  geom_boxplot(aes(reorder(JobLevel, avg), MonthlyIncome, fill = avg)) + 
  coord_flip() + theme_minimal() + 
  scale_fill_continuous(low = '#ffffcc', high = '#fc4e2a', name = "Average Salary Level") + 
  labs(x = 'JobLevel', y = 'MonthlyIncome')+
  expand_limits(y = c(0,5))

# Distribution of monthly income by performance rating
df3 %>%
  group_by(WorkLifeBalance) %>% 
  mutate(avg = mean(MonthlyIncome)) %>%
  ggplot() + 
  geom_boxplot(aes(reorder(WorkLifeBalance, avg), MonthlyIncome, fill = avg)) + 
  coord_flip() + theme_minimal() + 
  scale_fill_continuous(low = '#ffffcc', high = '#fc4e2a', name = "Average Salary Level") + 
  labs(x = 'WorkLifeBalance', y = 'MonthlyIncome')+
  expand_limits(y = c(0,5))

# Monthly Income vs Years at Company
ggplot(final_df,aes(TotalWorkingYears, MonthlyIncome)) + 
  geom_point() +labs(title = 'Monthly Income vs Years at Company')+
  geom_smooth(method = "lm", se = FALSE)


################################################################
# Research question 2: What are the most important factors affect Attrition
# OverTime vs Attrition
ggplot(df3,aes(x=OverTime, fill=Attrition))+
  geom_bar()+
  theme()+ 
  labs(     # labs() labels the...
    title = "OverTime Count across Attrition", 
    x = "OverTime",        
    y = "Count",    
  )

# BusinessTravel vs Attrition
ggplot(df3,aes(x=BusinessTravel, fill=Attrition))+
  geom_bar()+
  theme()+ 
  labs(     # labs() labels the...
    title = "Business Travel Level Count across Attrition", 
    x = "BusinessTravel",        
    y = "Count",    
  )

# MaritalStatus vs Attrition
ggplot(df3,aes(x=MaritalStatus, fill=Attrition))+
  geom_bar()+
  theme()+ 
  labs(     # labs() labels the...
    title = "Marital Status Count across Attrition", 
    x = "MaritalStatus",        
    y = "Count",    
  )

# JobSatisfaction vs Attrition
ggplot(df3,aes(factor(Attrition),JobSatisfaction,fill=Attrition))+geom_boxplot()+
  labs( x = "Attrition",        # ...x-axis
        y = "Employee Job Satisfaction",    # ...y-axis
        title="Attrition & Job Satisfaction")


