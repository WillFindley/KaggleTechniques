KNN <- function(trainDirectory,testdirectory) {

	memory <- read.csv(trainDirectory)

	# this choice has roughly as many effective parameters as image dimensions * categories
	k <- nrow(memory)/(10*ncol(memory))
	k <- k - (k %% 2) + 1;

	probes <- read.csv(testdirectory)

	runKNN(memory,k,probes)
}

runKNN <- function(memory,k,probes) {

	answerFile <- file("answers")
	for (toProbe in 1:nrow(probes)) {
		answer <- knnVote(memory,k,as.integer(probes[toProbe,]))
		writeLines(answer,answerFile)
		print(answer + '\t' + (toProbe/nrow(probes)))
	}
	close(answerFile)
}

knnVote <- function(memory,k,probe) {

	q <- MaxHeap(k)
	for (row in 1:nrow(memory)) {
		memoryLoc <- as.integer(memory[row,2:ncol(memory)])
		distance <- norm(as.matrix(probe-memoryLoc))
		q$pushPop(c(distance,as.integer(memory[row,1])))
	}
	tempTable <- table(as.vector(q$data))
	return names(tempTable)[tempTable == max(tempTable)]
}

MaxHeap <- function(k) {

	data <- rep(c(.Machine$integer.max,15),c(k,1))

	addToHeap <- function(self,entry,vertex) {
		
		if (2*vertex <= nrow(data) && data[2*vertex,0] > entry[0]) {
			data[vertex,] <- data[2*vertex,]
			addToHeap(entry,2*vertex)
		} else if (2*vertex+1 <= nrow(data) && data[2*vertex+1,0] > entry[0]) {
			data[vertex,] <- data[2*vertex+1,]
			addToHeap(entry,2*vertex+1)
		} else {
			data[vertex,] <- entry
		}
	}

	me <- list(	
		data <- data,
		
		pushPop <- function(entry) {

			if (entry[0] < data[0][0]) {
				addToHeap(entry,1)
			}
		}
	)

	class(me) <- append(class(me),"MaxHeap")
	return(me)
}

KNN("../Data/train.csv","../Data/test.csv")
