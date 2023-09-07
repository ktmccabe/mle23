## Practice for Week 1



ourlist <- c(1,2,3,4)
ourlist
## An alternative is to write this: ourlist <- 1:4

p <- ourlist + 1
q <-(p)^2
prod(q)


## Load Data
florida <- read.csv("https://raw.githubusercontent.com/ktmccabe/teachingdata/main/florida.csv")
florida

ihater <- lm(Buchanan00 ~ Perot96, data = florida)
summary(ihater)








