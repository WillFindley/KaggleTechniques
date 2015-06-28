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
		q$pushPop(c(distance,as.integer(memory[row,1])),q$data)
	}
	categories <- as.integer(q$data[,2])
	uniqueCategories <- unique(categories)
	return(uniqueCategories[which.max(tabulate(match(categories,uniqueCategories)))])
}

MaxHeap <- function(k) {

	ma <- matrix(c(.Machine$integer.max,15), nrow=1)
	data <- ma[rep(1,k),]

	pushPop <- function(entry,data) {

			addToHeap <- function(entry,vertex) {
		
				if (2*vertex <= nrow(data) && data[2*vertex,1] > entry[1]) {
					data[vertex,] <- data[2*vertex,]
					addToHeap(entry,2*vertex)
				} else if (2*vertex+1 <= nrow(data) && data[2*vertex+1,1] > entry[1]) {
					data[vertex,] <- data[2*vertex+1,]
					addToHeap(entry,2*vertex+1)
				} else {
					data[vertex,] <- entry
				}
			}
			
			if (entry[1] < data[1,1]) {
				addToHeap(entry,1)
			}
		}

	me <- list(data=data, pushPop=pushPop)

	return(me)
}

KNN("../Data/train.csv","../Data/test.csv")
