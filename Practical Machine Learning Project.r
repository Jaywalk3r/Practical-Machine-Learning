library( caret)

library( mda)

library( kernlab)

library( adabag)

library( rpart)

library( doMC)

registerDoMC( cores = 2)



setwd( "~/Documents/School/Coursera/Data Science Specialization/Practical Machine Learning/Project")

filePath1 = "./pml-training.csv"

filePath2 = "./pml-testing.csv"

if (! file.exists( filePath1)) {

	fileUrl1 = "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv"
	
	download.file( fileUrl1, filePath1, method = "curl")
	
}

if (! file.exists( filePath1)) {

	fileUrl2 = "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv"
	
	download.file( fileUrl2, filePath2, method = "curl")
	
}

training.raw = read.csv( filePath1, na.strings = c( NA, "#DIV/0!"))

checking.raw = read.csv( filePath2, na.strings = c( NA, "#DIV/0!"))



#str( training.raw, list.len = dim( training.raw)[ 2])

range( training.raw$X) # looks like observation indices; row numbers suffice

# columns 2:7 do not appear to be useful for future predictions

del.index = 1:7
	# initialize a vector containing column indices of predictors in which we
	#	have no interest

na.proportion = rep( 0 , 159)

for (i in 8:159) {

	na.proportion[ i] = mean( is.na( training.raw[ , i]))

	if (na.proportion[ i] > .6) {
	
		del.index = c( del.index, i)
	
	}

} # add any variable with greater than 60% NA values

min( na.proportion[ na.proportion != 0])

max( na.proportion[ na.proportion != 1])
	# We see variables with any NA values have mostly NA values.
	
length( na.proportion[ na.proportion > 0 & na.proportion < .6])


length( nearZeroVar( training.raw[ , -c( del.index, 160)]))
	# check to see how many remaining features have near zero variance

training = training.raw[ , -del.index]

checking = checking.raw[ , -del.index]

sum( ! complete.cases( training)) # count of non-complete observations: 0



dim( training)[ 1] / dim( training)[ 2]
	# ratio of observations to predictors
	



# We would like a larger test set than testing, which is only 20 observations.
#	Our training set contains 19,622 observations, so we'll hold some of them
#	back for testing and validation.



set.seed( 1123)

inTrain = createDataPartition( training$classe, p = .75, list = FALSE)

testSet = training[ -inTrain,]

training = training[ inTrain,]

dim( training)[ 1] / dim( training)[ 2]
	# New ratio of observations to predictors

set.seed( 11235)

inTest = createDataPartition( testSet$classe, p = .5, list = FALSE)

testing = testSet[ inTest,]

validation = testSet[ -inTest,]





ctrl = trainControl( method = "repeatedcv", number = 10, repeats = 5, savePredictions = TRUE)


# Mixture Discriminant Analysis

set.seed( 112358)

mda1Fit = train( training[ , -53], training$classe,
				method = "mda",
				trControl = ctrl,
				tuneLength = 10)

saveRDS( mda1Fit, "mda1Fit.rds")

mda1Results = confusionMatrix( predict( mda1Fit, testing), testing$classe)



# mda2Fit provides best results among MDA models

set.seed( 112358)

mda2Fit = train( training[ , -53], training$classe,
				method = "mda",
				preProcess = c( "BoxCox", "center", "scale"),
				trControl = ctrl,
				tuneLength = 10)

saveRDS( mda2Fit, "mda2Fit.rds")
				
mda2Results = confusionMatrix( predict( mda2Fit, testing), testing$classe)


set.seed( 112358)

mda3Fit = train( training[ , -53], training$classe,
				method = "mda",
				preProcess = c( "center", "scale", "pca"),
				trControl = ctrl,
				tuneLength = 10)

saveRDS( mda3Fit, "mda3Fit.rds")
				
mda3Results = confusionMatrix( predict( mda3Fit, testing), testing$classe)


set.seed( 112358)

mda4Fit = train( training[ , -53], training$classe,
				method = "mda",
				preProcess = c( "BoxCox", "center", "scale", "pca"),
				trControl = ctrl,
				tuneLength = 10)
				
saveRDS( mda4Fit, "mda4Fit.rds")

mda4Results = confusionMatrix( predict( mda4Fit, testing), testing$classe)				




# KNN

set.seed( 112358)

knnFit = train( training[, -53], training$classe,
				 method = "knn",
				 preProcess = c( "BoxCox", "center", "scale"),
				 trControl = ctrl,
				 tuneLength = 10)

saveRDS( knnFit, "knnFit.rds")

knnResults = confusionMatrix( predict( knnFit, testing[ ,-53]), testing$classe)



# Need to test for k < 5
knnGrid = data.frame( k = 1:10)



set.seed( 112358)

knn1Fit = train( training[, -53], training$classe,
				 method = "knn",
				 preProcess = c( "BoxCox", "center", "scale"),
				 trControl = ctrl,
				 tuneLength = 10,
				 tuneGrid = knnGrid)
				 
saveRDS( knn1Fit, "knn1Fit.rds")



knn1Results = confusionMatrix( predict( knn1Fit, testing[ ,-53]), testing$classe)



set.seed( 112358)

knn2Fit = train( training[, -53], training$classe,
				 method = "knn",
				 preProcess = c( "BoxCox", "center", "scale", "pca"),
				 trControl = ctrl,
				 tuneLength = 10,
				 tuneGrid = knnGrid)
				 
saveRDS( knn2Fit, "knn2Fit.rds")

knn2Results = confusionMatrix( predict( knn2Fit, testing[ ,-53]), testing$classe)


set.seed( 112358)

knn3Fit = train( training[, -53], training$classe,
				 method = "knn",
				 preProcess = c( "center", "scale"),
				 trControl = ctrl,
				 tuneLength = 10,
				 tuneGrid = knnGrid)
				 
saveRDS( knn3Fit, "knn3Fit.rds")

predictionKNN3 = predict( knn3Fit, testing[ ,-53])
				 
knn3Results = confusionMatrix( predictionKNN3, testing$classe)





# Partial Least Squares

set.seed( 112358)

pls1Fit = train( training[, -53], training$classe,
				 method = "pls",
				 preProcess = c( "center", "scale"),
				 trControl = ctrl,
				 tuneLength = 10,
				 tuneGrid = expand.grid( .ncomp = 30:52))

saveRDS( pls1Fit, "pls1Fit.rds")

pls1Results = confusionMatrix( predict( pls1Fit, testing[ ,-53]), testing$classe)

# PLS not a good choice, best accuracy @ ncomp = 52 .6978

set.seed( 112358)

pls2Fit = train( training[, -53], training$classe,
				 method = "pls",
				 preProcess = c( "BoxCox", "center", "scale"),
				 trControl = ctrl,
				 tuneLength = 10,
				 tuneGrid = expand.grid( .ncomp = 42:52))

saveRDS( pls2Fit, "pls2Fit.rds")

pls2Results = confusionMatrix( predict( pls2Fit, testing[ ,-53]), testing$classe)

# very slightly better, accuracy still < .7



# Random Forest

set.seed( 112358)

rfFit = train( training[, -53], training$classe,
			   method = "rf",
			   preProcess = c( "center", "scale"),
			   trControl = ctrl,
			   tuneLength = 10)

saveRDS( rfFit, "rfFit.rds")

rfResults = confusionMatrix( predict( rfFit, testing[ , -53]), testing$classe)




rfGrid = data.frame( mtry = 3:12)

set.seed( 112358)

rf1Fit = train( training[, -53], training$classe,
			    method = "rf",
			    preProcess = c( "center", "scale"),
			    trControl = ctrl,
			    tuneGrid = rfGrid)

saveRDS( rf1Fit, "rf1Fit.rds")

predictionRF1 = predict( rf1Fit, testing[ , -53])

rf1Results = confusionMatrix( predictionRF1, testing$classe)




# Boosting (AdaBoost.M1)

set.seed( 112358)

adaboostFit = train( training[, -53], training$classe,
					 method = "AdaBoost.M1",
					 preProcess = c( "center", "scale"),
					 trControl = ctrl)

saveRDS( adaboostFit, "adaboostFit.rds")

adaboostResults = confusionMatrix( predict( adaboostFit, testing[ , -53]), testing$classe)





adaboostGrid = data.frame( coeflearn = rep( "Zhu", 6), maxdepth = rep( 3, 6), mfinal = 100 * 5:10)




set.seed( 112358)

adaboost2Fit = train( training[, -53], training$classe,
					 method = "AdaBoost.M1",
					 preProcess = c( "center", "scale"),
					 trControl = ctrl,
					 tuneGrid = adaboostGrid)

saveRDS( adaboost2Fit, "adaboost2Fit.rds")

predictionAdaBoost = predict( adaboost2Fit, testing[ , -53])

adaboost2Results = confusionMatrix( predictionAdaBoost, testing$classe)




# CART

set.seed( 112358)

cartFit = train( training[, -53], training$classe,
				 method = "rpart",
				 preProcess = c( "center", "scale"),
				 trControl = ctrl,
				 tuneLength = 10)

saveRDS( cartFit, "cartFit.rds")

cartResults = confusionMatrix( predict( cartFit, testing[ , -53]), testing$classe)


# Bagged CART

set.seed( 112358)

bagcartFit = train( training[, -53], training$classe,
				 method = "rpart",
				 preProcess = c( "center", "scale"),
				 trControl = ctrl)

saveRDS( bagcartFit, "bagcartFit.rds")

bagcartResults = confusionMatrix( predict( bagcartFit, testing[ , -53]), testing$classe)



# Support Vector Machines

sigmaRange = sigest( as.matrix( training[ , -53])) # sigma estimate

svmRGrid = expand.grid( .sigma = sigmaRange[ 1], .C = 2 ^ ( seq( -4, 4)))
								   
set.seed( 112358)

svm1Fit = train( training[, -53], training$classe,
				 method = "svmRadial",
				 preProcess = c( "center", "scale"),
				 trControl = ctrl,
				 tuneGrid = svmRGrid)

saveRDS( svm1Fit, "svm1Fit.rds")

svm1Results = confusionMatrix( predict( svm1Fit, testing[ , -53]), testing$classe)



svmRGrid2 = expand.grid( .sigma = sigmaRange[ 1], .C = c( 20, 24, 32, 36))
								   
set.seed( 112358)

svm2Fit = train( training[, -53], training$classe,
				 method = "svmRadial",
				 preProcess = c( "center", "scale"),
				 trControl = ctrl,
				 tuneGrid = svmRGrid2)

saveRDS( svm2Fit, "svm2Fit.rds")

predictionSVM2 = predict( svm2Fit, testing[ , -53])

svm2Results = confusionMatrix( predictionSVM2, testing$classe)



accuracy = rbind( knn3Results$overall[ c( 1, 3, 4)], rf1Results$overall[ c( 1, 3, 4)], adaboost2Results$overall[ c( 1, 3, 4)], svm2Results$overall[ c( 1, 3, 4)])

rownames( accuracy) = c("KNN", "RF", "AdaBoost", "SVM")

colnames( accuracy) = c( "Accuracy", "Lower", "Upper")

variance = function( x) {

	x[ 1] = xBar
	
	x[ 2] = lower
	
	

}






wrongIndex = (testing$classe != predictionSVM2 | testing$classe != predictionAdaBoost | testing$classe != predictionKNN3 | testing$classe != predictionRF1)


predictionWrong = data.frame( true = testing$classe[ wrongIndex],
							  SVM = predictionSVM2[ wrongIndex],
							  AdaBoost.M1 = predictionAdaBoost[ wrongIndex],
							  KNN = predictionKNN3[ wrongIndex],
							  RF = predictionRF1[ wrongIndex])

sum( predictionSVM2[ wrongIndex] != testing$classe[ wrongIndex])

sum( predictionAdaBoost[ wrongIndex] != testing$classe[ wrongIndex])

sum( predictionKNN3[ wrongIndex] != testing$classe[ wrongIndex])

sum( predictionRF1[ wrongIndex] != testing$classe[ wrongIndex])

dim( predictionWrong)[ 1]

# Ensembling

# Positive Predictive Value (PPV) matrix

PPV = cbind( svm2Results$byClass[ , "Pos Pred Value"], knn3Results$byClass[ , "Pos Pred Value"], rf1Results$byClass[ , "Pos Pred Value"])

colnames( PPV) = c( "SVM", "KNN", "RF")

rownames( PPV) = c( "A", "B", "C", "D", "E")



# Check to make sure no model's PPV for prediction of k matches another model's
#	PPV for a prediction of l, k != l

count = 0

for (i in 1:4) {

	for (j in (i + 1):5) {
	
		count = count + sum( PPV[ j,] %in% PPV[ i,])
	
	}

}

count


# combine predictions of best models into data.frame,
#	each column representing a model

allPredictions = cbind( as.character( predictionSVM2), as.character( predictionKNN3), as.character( predictionRF1))

# Alternative?

modelList = list( SVM = svm2Fit, KNN = knn3Fit, RF = rf1Fit)

allPredictions = sapply( modelList, predict, newdata = testing[ , -53])


# Case 1: model predictions all agree
#		  prediction is consensus

# Case 2: inconsistent prediction among models
#		  weighted voting


# This function is more general than the one submitted for the project
#	In particular, it contains a tie breaking protocol for the unusual case in
#	which there is a tie after vote weighting.
prediction = function( council) {
		# council is vector of predictions for single observation

	if (length( unique( council)) == 1) {
			# all models agree on predicted class
			
		return( council[ 1])
	
	} else {
			# disagreement among models regarding predicted class results in
			#	weighted vote
			
		ballotBox = data.frame( candidate = c( "A", "B", "C", "D", "E"), votes = rep( 0, 5))
		
		k = length( council)
		
		for( i in 1:k) {
		
			ballotBox$votes[ ballotBox$candidate == council[ i]] = ballotBox$votes[ ballotBox$candidate
					== council[ i]] + PPV[ council[ i], i]
				# each model's vote weight/value is equal to model's positive
				#	predictive value for the class it predicts.
			
		}
		
		returnIndex = max( ballotBox$votes) == ballotBox$votes
			# determine which class received most votes
			
			if (sum( returnIndex == 1)) {
				
				return( as.character( ballotBox$candidate[ returnIndex]))
					# if as.character() is not used, then value is returned as
					#	integer.
				
			} else {
					# in unlikely event of a tie
				
				ballotBox = data.frame( candidate = c( "A", "B", "C", "D", "E"), votes = rep( 0, 5))
				
				for( i in 1:k) {
		
					ballotBox$votes[ ballotBox$candidate == council[ i]] = ballotBox$votes[ ballotBox$candidate == council[ i]] + PPV[ council[ i], i] ^ 2
							# tiebreaker goes to model with highest PPV
							# 	length( unique( PPV)) must be equal to
							# 	dim( PPV)[ 1] * dim( PPV)[ 2] to guarantee
							# 	this step breaks tie. This tiebreaker assumes at
							# 	least four models are involved in tie.
			
				}
				
				return( as.character( ballotBox$candidate[ returnIndex]))
			
			}
	
	}

}

predictionVote = as.factor( apply( allPredictions, 1, prediction))

confusionMatrix( predictionVote, testing$classe)

set.seed( 112358)

ensemble2 = train( allPredictions, testing$classe,
				   method = "AdaBoost.M1",
				   trControl = ctrl)

saveRDS( ensemble2, "ensemble2.rds")


set.seed( 112358)

ensemble3 = train( allPredictions, testing$classe,
				   method = "rpart",
				   trControl = ctrl,
				   tuneLength = 10)

saveRDS( ensemble3, "ensemble3.rds")


predMatrix = sapply( modelList, predict, newdata = validation[ , -53])

validationVote = as.factor( apply( predMatrix, 1, prediction))

validationEn2pred = predict( ensemble2, predMatrix)

validationEn3pred = predict( ensemble3, predMatrix)

svmConfusion = confusionMatrix( predMatrix[ , 1], validation$classe)

knnConfusion = confusionMatrix( predMatrix[ , 2], validation$classe)

rfConfusion = confusionMatrix( predMatrix[ , 3], validation$classe)

voteConfusion = confusionMatrix( validationVote, validation$classe)

en2Confusion = confusionMatrix( validationEn2pred, validation$classe)

en3Confusion = confusionMatrix( validationEn3pred, validation$classe)

AccCompMat = rbind( svmConfusion$overall[c( 3, 1, 4, 2)],
					knnConfusion$overall[c( 3, 1, 4, 2)],
					rfConfusion$overall[c( 3, 1, 4, 2)],
					voteConfusion$overall[c( 3, 1, 4, 2)],
					en2Confusion$overall[c( 3, 1, 4, 2)],
					en3Confusion$overall[c( 3, 1, 4, 2)])
					
rownames( AccCompMat) = c( "SVM", "KNN", "RF", "Vote", "EnsAB", "EnsCART")

AccCompMat


n = length( validation$classe)

prop.test( n * AccCompMat[ , 2], rep( n, 6))$p.value

prop.test( n * AccCompMat[ 2:6, 2], rep( n, 5))$p.value

prop.test( n * AccCompMat[ 3:6, 2], rep( n, 4))$p.value



checkPredMatrix = sapply( modelList, predict, newdata = checking[ , -53])

checkingVote = as.factor( apply( checkPredMatrix, 1, prediction))

checkingEn2pred = predict( ensemble2, checkPredMatrix)

checkingEn3pred = predict( ensemble3, checkPredMatrix)

cbind( checkPredMatrix, checkingVote, checkingEn2pred, checkingEn3pred)
