KNN <- function(trainDirectory,testdirectory) {

	memory <- read.csv(trainDirectory)

	# this choice has roughly as many effective parameters as image dimensions * categories
	k <- nrow(memory)/(10*ncol(memory))
	k <- k - (k %% 2) + 1;

	probes <- read.csv(testdirectory)

	runKNN(memory,k,probes)
}

runKNN <- function(memory,k,probes) {

	df = data.frame(matrix(ncol = 2, nrow = nrow(probes)))
	for (toProbe in 1:nrow(probes)) {
		answer <- knnVote(memory,k,as.integer(probes[toProbe,]))
		df[toProbe,] <- c(toProbe,answer)
		print(paste(toString(answer),toString(toProbe/nrow(probes)),sep="\t"))
	}
	colnames(df) <- c("ImageId", "Label")
	write.csv(df,file="answers.csv",row.names=FALSE)
}

knnVote <- function(memory,k,probe) {

	q <- MaxHeap(k)
	for (row in 1:nrow(memory)) {
		memoryLoc <- as.integer(memory[row,2:ncol(memory)])
		distance <- norm(as.matrix(probe-memoryLoc))
		q <- pushPop.MaxHeap(q,c(distance,as.integer(memory[row,1])))
	}
	categories <- as.integer(q$data[,2])
	uniqueCategories <- unique(categories)
	return(uniqueCategories[which.max(tabulate(match(categories,uniqueCategories)))])
}

MaxHeap <- function(k) {

	ma <- matrix(c(.Machine$integer.max,15), nrow=1)
	data <- ma[rep(1,k),]

	me <- list(data=data)

	class(me) <- "MaxHeap"

	return(me)
}

pushPop <- function(x) UseMethod("pushPop")
pushPop.default <- function(x) "Unknown class"

pushPop.MaxHeap <- function(maxHeap,entry) {

			
	if (entry[1] < maxHeap$data[1,1]) {
		maxHeap$data <- addToHeap(maxHeap$data,entry,1)
	}
	
	return(maxHeap)
}

addToHeap <- function(data,entry,vertex) {

	if (2*vertex <= nrow(data) && data[2*vertex,1] > entry[1]) {
		data[vertex,] <- data[2*vertex,]
		addToHeap(data,entry,2*vertex)
	} else if (2*vertex+1 <= nrow(data) && data[2*vertex+1,1] > entry[1]) {
		data[vertex,] <- data[2*vertex+1,]
		addToHeap(data,entry,2*vertex+1)
	} else {
		data[vertex,] <- entry
		return(data)
	}
}


KNN("../Data/train.csv","../Data/test.csv")
