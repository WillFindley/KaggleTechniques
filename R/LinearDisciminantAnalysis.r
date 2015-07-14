LDA <- function(trainDirectory,testdirectory) {

	memory <- read.csv(trainDirectory)

	parameters <- calculateLinearDiscriminantFunctions(memory)

	probes <- read.csv(testdirectory)

	runLDA(parameters,probes)
}

calculateLinearDiscriminantFunctions <- function(memory) {

	N <- nrow(memory)
	K <- 10

	piK <- matrix(data=0,ncol=1,nrow=K)
	meanK <- matrix(data=0,ncol=(ncol(memory)-1),nrow=K)
	for (row in 1:N) {
		piK[as.integer(memory[row,1])+1,1] <- piK[as.integer(memory[row,1])+1,1] + 1
		meanK[as.integer(memory[row,1])+1,] <- as.integer(meanK[as.integer(memory[row,1])+1,]) + as.integer(memory[row,2:ncol(memory)])
	}

	for (whichK in 1:K) {
		meanK[whichK,] <- meanK[whichK,] / piK[whichK,1]
	}
	piK <- piK / N

	covariance = 0
	for (row in 1:N) {
		distanceFromCentroid <- as.integer(memory[row,2:ncol(memory)]) - meanK[as.integer(memory[row,1])+1,]
		covariance <- covariance + distanceFromCentroid %*% distanceFromCentroid
	}
	covariance <- covariance / (N-K)

	probeIndependentTerm <- matrix(data=0,nrow=K,ncol=1)
	for (whichK in 1:K) {
		probeIndependentTerm[whichK,1] <- log(piK[whichK,1]) - ((meanK[whichK,] %*% meanK[whichK,]) / (2*covariance))
	}

	parameters <- list(piK=piK,meanK=meanK,covariance=covariance,probeIndependentTerm=probeIndependentTerm)
	return(parameters)
}

runLDA <- function(parameters,probes) {

	df = data.frame(matrix(ncol = 2, nrow = nrow(probes)))
	for (toProbe in 1:nrow(probes)) {
		answer <- maxLDFunction(parameters,as.integer(probes[toProbe,]))
		df[toProbe,] <- c(toProbe,answer)
		print(paste(toString(answer),toString(toProbe/nrow(probes)),sep="\t"))
	}
	colnames(df) <- c("ImageId", "Label")
	write.csv(df,file="answersLDA.csv",row.names=FALSE)
}

maxLDFunction <- function(parameters,probe) {

	answer <- c(-1,-Inf)
	for (whichK in 1:nrow(parameters$probeIndependentTerm)) {
		thisLDFunction <- (probe %*% as.integer(parameters$meanK[whichK,])) / parameters$covariance + as.integer(parameters$probeIndependentTerm[whichK,1])
		if (thisLDFunction >= answer[2]) {
			answer = c(whichK,thisLDFunction)
		}
	}
	return(answer[1])
}


LDA("../Data/train.csv","../Data/test.csv")
