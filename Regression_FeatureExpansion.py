import sys
from sklearn.datasets import load_boston
import pandas as pd
import numpy as np
import itertools
import operator
import matplotlib.pyplot as plt
from collections import OrderedDict
import time

	
def standardize(trainOrTestFeature, normDict, featureNames):
	for x in featureNames:
		trainOrTestFeature[x] = (trainOrTestFeature[x] - normDict[x]["MEAN"])/(normDict[x]["SD"] * 1.0)
	return trainOrTestFeature

def calculateMSE(trainOrTestFeature, theta, trainOrTestTarget):
	predictedTarget = np.dot(trainOrTestFeature.as_matrix(), theta)
	squaredError = (predictedTarget.flatten() - trainOrTestTarget["target"])**2
	sumOfSquaredError = squaredError.sum()
	meanSquareError = (sumOfSquaredError*1.0)/len(predictedTarget)
	return meanSquareError

def calculateResidue(trainOrTestFeature, theta, trainOrTestTarget):
	predictedTarget = np.dot(trainOrTestFeature.as_matrix(), theta)
	residue = (predictedTarget.flatten() - trainOrTestTarget["target"])
	residueDF = pd.DataFrame(residue,columns=["target"])
	return residueDF

def calculateTheta(trainingFeature, trainingTarget, regressor, lamda):
	if regressor == "LINEAR":
		innerProduct = np.dot(trainingFeature.transpose().as_matrix(),trainingFeature.as_matrix())
	elif regressor == "RIDGE":
		N = trainingFeature.columns.size
		identityMatrix = lamda * np.identity(N)
		innerProduct = np.dot(trainingFeature.transpose().as_matrix(),trainingFeature.as_matrix()) + identityMatrix

	inverseVal = np.linalg.pinv(innerProduct)
	theta = np.dot(np.dot(inverseVal,trainingFeature.transpose().as_matrix()),trainingTarget.as_matrix())
	return theta

def calculateNormDict(trainingFeature, featureNames):
	normDict = {}
	for x in featureNames:
		normDict.setdefault(x,{})
		normDict[x]["MEAN"] = trainingFeature[x].mean()
		normDict[x]["SD"] = trainingFeature[x].as_matrix().std()
	return normDict

def linearRegression(trainingFeature, trainingTarget, testFeature, testTarget, normDict, featureNames):
	###### STANDARDIZING TRAINING DATA #############
	trainingFeature = standardize(trainingFeature, normDict, featureNames)
	theta = calculateTheta(trainingFeature, trainingTarget, "LINEAR", 0)
	trainingMSE = calculateMSE(trainingFeature, theta, trainingTarget)

	###### STANDARDIZING TRAINING DATA #############
	testFeature = standardize(testFeature, normDict, featureNames)
	testMSE = calculateMSE(testFeature, theta, testTarget)
	
	result = [trainingMSE,testMSE]
	return result

def ridgeRegression(trainingFeature, trainingTarget, testFeature, testTarget, normDict, featureNames):
	lamdaValues = [0.0001, 0.001, 0.01, 0.1, 1.0, 10]
	print "==========   Ridge Regression ============"
	for lamda in lamdaValues:
		print "For LAMBDA = "+ str(lamda)
		theta = calculateTheta(trainingFeature, trainingTarget, "RIDGE", lamda)
		trainingMSE = calculateMSE(trainingFeature, theta, trainingTarget)
		testMSE = calculateMSE(testFeature, theta, testTarget)
		print "     Training MSE:" + str(trainingMSE)
		print "     Test MSE:" + str(testMSE)
		print ""

def ridgeRegression_with_CV(featureMatrix, targetMatrix, featureNames, trainingIndex, testIndex):
	trainFeatureForCV = pd.DataFrame(featureMatrix,index = trainingIndex)
	trainTargetForCV = pd.DataFrame(targetMatrix,index = trainingIndex)

	trainFeatureForCV["target"] = trainTargetForCV["target"].copy()
	np.random.seed(0)
	trainMatrix = trainFeatureForCV.as_matrix()
	np.random.shuffle(trainMatrix)
	newFeatureList = []
	newFeatureList = list(np.copy(featureNames))
	newFeatureList.append("x0")
	newFeatureList.append("target")
	trainFeatureForCV = pd.DataFrame(trainMatrix,columns = newFeatureList)
	
	trainTargetForCV["target"] = list(trainFeatureForCV["target"])
	trainFeatureForCV.drop(["target"], axis=1, inplace=True)

	bins = np.array_split(trainFeatureForCV.as_matrix(),10)
	targetBins = np.array_split(trainTargetForCV.as_matrix(),10)
	
	fNames = np.append(featureNames,"x0")
	testMSEDict = {}

	for i in range(0,len(bins)):
		testFBin = np.copy(bins[i])
		testTBin = np.copy(targetBins[i])

		testBinDF = pd.DataFrame(columns=fNames)
		testTargetBinDF =  pd.DataFrame(columns=["target"])

		testBinDF = pd.DataFrame(testFBin,columns= fNames)
		testTargetBinDF = pd.DataFrame(testTBin, columns =["target"])
		
		trainBinDF =  pd.DataFrame(columns=fNames)
		trainTargetBinDF = pd.DataFrame(columns=["target"])

		for j in range(0,len(bins)):
			if j != i:
				trainFBin = np.copy(bins[j])
				trainTBin = np.copy(targetBins[j])
				trainBinDF = trainBinDF.append(pd.DataFrame(trainFBin,columns= fNames))
				trainTargetBinDF = trainTargetBinDF.append(pd.DataFrame(trainTBin,columns= ["target"]))

		normDictForCV = calculateNormDict(trainBinDF, featureNames)
		trainBinDF = standardize(trainBinDF, normDictForCV, featureNames)
		testBinDF = standardize(testBinDF, normDictForCV, featureNames)

		bestLamda = 7.908244
		
		lamdaValuesForCV = [0.0001, 0.001, 0.01, 0.1, 1.0, 10, bestLamda]
		for lamda in lamdaValuesForCV:
			thetaForCV = calculateTheta(trainBinDF, trainTargetBinDF, "RIDGE", lamda)
			testMSEForCV = calculateMSE(testBinDF, thetaForCV, testTargetBinDF)
			testMSEDict.setdefault(lamda, 0.0)
			testMSEDict[lamda] = testMSEDict[lamda] + testMSEForCV
		
	print "==========   Ridge Regression with 10-fold Cross-Validation   ============"
	for lamda in sorted(testMSEDict.keys()):
		print "For LAMBDA = "+ str(lamda)+ " Test MSE:" + str(testMSEDict[lamda]/ len(bins))

	print ""
	print "Best lambda:" + str(bestLamda)

	normDict = calculateNormDict(trainFeatureForCV, featureNames)
	testFeatureForCV = pd.DataFrame(featureMatrix,index = testIndex)
	testTargetForCV = pd.DataFrame(targetMatrix,index = testIndex)

	trainFeatureForCV = standardize(trainFeatureForCV, normDict, featureNames)
	testFeatureForCV = standardize(testFeatureForCV, normDict, featureNames)

	
	thetaForCV = calculateTheta(trainFeatureForCV, trainTargetForCV, "RIDGE", bestLamda)
	trainMSEForCV = calculateMSE(trainFeatureForCV, thetaForCV, trainTargetForCV)
	testMSEForCV = calculateMSE(testFeatureForCV, thetaForCV, testTargetForCV)
	print "For LAMBDA = "+ str(bestLamda)
	print "     Training MSE:" + str(trainMSEForCV)
	print "     Test MSE:" + str(testMSEForCV)
	print ""

def findPearsonCorrelation(trainFeatureForFS, trainTargetForFS, featureNames, takeAbs):
	normDictForFS = calculateNormDict(trainFeatureForFS, featureNames)
	targetMean = trainTargetForFS["target"].mean()
	targetSD = trainTargetForFS["target"].as_matrix().std()

	correlationDict = {}
	for col in featureNames:
		correlationDict.setdefault(col,0.0)
		numerator = 0.0
		for row in range(0, trainFeatureForFS.index.size):
			numerator = numerator + ((trainFeatureForFS.iloc[row][col]-normDictForFS[col]["MEAN"]) * (trainTargetForFS.iloc[row]["target"] - targetMean))

		numerator = float(numerator) / float(trainFeatureForFS.index.size)
		denominator = normDictForFS[col]["SD"] * targetSD
		if takeAbs == True:
			correlationDict[col] = np.fabs(float(numerator) / float(denominator))
		else:
			correlationDict[col] = (float(numerator) / float(denominator))
	return correlationDict

def featureSelection(featureMatrix, targetMatrix, featureNames, trainingIndex, testIndex):
	trainFeatureForFS = pd.DataFrame(featureMatrix,index = trainingIndex)
	trainTargetForFS = pd.DataFrame(targetMatrix,index = trainingIndex)

	correlationDict = findPearsonCorrelation(trainFeatureForFS, trainTargetForFS, featureNames, True)

	correlatedFeatures = sorted(correlationDict, key=correlationDict.get, reverse=True)[0:4]
	print "==========   Feature Selection   ============"
	print "Highest Correlated Features (Top 4): " + str(", ".join(correlatedFeatures))

	normDictForFS = calculateNormDict(trainFeatureForFS, featureNames)

	dropFeatures = [feature for feature in featureNames if feature not in correlatedFeatures]
	trainFeatureForFS.drop(dropFeatures, axis=1, inplace=True)

	testFeatureForFS = pd.DataFrame(featureMatrix,index = testIndex)
	testFeatureForFS.drop(dropFeatures, axis=1, inplace=True)

	testTargetForFS = pd.DataFrame(targetMatrix,index = testIndex)
	result = linearRegression(trainFeatureForFS, trainTargetForFS, testFeatureForFS, testTargetForFS, normDictForFS, correlatedFeatures)
	print "Training MSE:" + str(result[0])
	print "Test MSE:" + str(result[1])

def featureSelectionWithResidue(featureMatrix, targetMatrix, featureNames, trainingIndex, testIndex):
	trainFeatureForFS = pd.DataFrame(featureMatrix,index = trainingIndex)
	trainTargetForFS = pd.DataFrame(targetMatrix,index = trainingIndex)

	correlationDict = findPearsonCorrelation(trainFeatureForFS, trainTargetForFS, featureNames, True)

	highestFeature = sorted(correlationDict, key=correlationDict.get, reverse=True)[0]
	normDictForFS = calculateNormDict(trainFeatureForFS, featureNames)

	dropFeatures = [feature for feature in featureNames if feature not in [highestFeature]]
	newTrainFeature = trainFeatureForFS.copy()
	newTrainFeature.drop(dropFeatures, axis=1, inplace=True)
	
	newTrainFeature = standardize(newTrainFeature, normDictForFS, [highestFeature])
	theta = calculateTheta(newTrainFeature, trainTargetForFS, "LINEAR", 0)
	residueDF = calculateResidue(newTrainFeature, theta, trainTargetForFS)

	correlatedFeatures = [highestFeature]
	while True:
		newTrainFeature = trainFeatureForFS.copy()
		newTrainFeature.drop(correlatedFeatures, axis=1, inplace=True)
		correlationDict = findPearsonCorrelation(newTrainFeature, residueDF, dropFeatures, True)
		newHighestFeature = sorted(correlationDict, key=correlationDict.get, reverse=True)[0]
		correlatedFeatures.append(newHighestFeature)
		if len(correlatedFeatures) == 4:
			break
		dropFeatures = [feature for feature in featureNames if feature not in correlatedFeatures]
		newTrainFeature = trainFeatureForFS.copy()
		newTrainFeature.drop(dropFeatures, axis=1, inplace=True)
		newTrainFeature = standardize(newTrainFeature, normDictForFS, correlatedFeatures)
		theta = calculateTheta(newTrainFeature, trainTargetForFS, "LINEAR", 0)
		residueDF = calculateResidue(newTrainFeature, theta, trainTargetForFS)
	
	print "Residual Based Features (Top 4): " + str(", ".join(correlatedFeatures))

	dropFeatures = [feature for feature in featureNames if feature not in correlatedFeatures]
	trainFeatureForFS.drop(dropFeatures, axis=1, inplace=True)

	testFeatureForFS = pd.DataFrame(featureMatrix,index = testIndex)
	testFeatureForFS.drop(dropFeatures, axis=1, inplace=True)

	testTargetForFS = pd.DataFrame(targetMatrix,index = testIndex)
	result = linearRegression(trainFeatureForFS, trainTargetForFS, testFeatureForFS, testTargetForFS, normDictForFS, correlatedFeatures)
	print "Training MSE:" + str(result[0])
	print "Test MSE:" + str(result[1])

def featureSelectionWithBruteForce(featureMatrix, targetMatrix, featureNames, trainingIndex, testIndex):
	
	trainFeatureForFS = pd.DataFrame(featureMatrix,index = trainingIndex)
	trainTargetForFS = pd.DataFrame(targetMatrix,index = trainingIndex)

	testFeatureForFS = pd.DataFrame(featureMatrix,index = testIndex)
	testTargetForFS = pd.DataFrame(targetMatrix,index = testIndex)

	normDictForFS = calculateNormDict(trainFeatureForFS, featureNames)

	corrFeatures = ""
	minMSE = np.inf
	combinations = itertools.combinations(featureNames, 4)
	for featureSet in combinations:
		correlatedFeatures = list(featureSet)
		dropFeatures = [feature for feature in featureNames if feature not in correlatedFeatures]
		newTrainFeature = pd.DataFrame(columns=correlatedFeatures)
		newTestFeature = pd.DataFrame(columns=correlatedFeatures)
		for feature in correlatedFeatures:
			newTrainFeature[feature] = trainFeatureForFS[feature]
			newTestFeature[feature] = testFeatureForFS[feature]

		result = linearRegression(newTrainFeature, trainTargetForFS, newTestFeature, testTargetForFS, normDictForFS, correlatedFeatures)
		key = ", ".join(correlatedFeatures)
		if result[0] < minMSE:
			minMSE = result[0]
			corrFeatures = key
	
	print "Brute-force Based Features (Top 4): " + str(corrFeatures)

	correlatedFeatures = corrFeatures.split(", ")
	dropFeatures = [feature for feature in featureNames if feature not in correlatedFeatures]
	trainFeatureForFS.drop(dropFeatures, axis=1, inplace=True)

	testFeatureForFS = pd.DataFrame(featureMatrix,index = testIndex)
	testFeatureForFS.drop(dropFeatures, axis=1, inplace=True)

	testTargetForFS = pd.DataFrame(targetMatrix,index = testIndex)
	result = linearRegression(trainFeatureForFS, trainTargetForFS, testFeatureForFS, testTargetForFS, normDictForFS, correlatedFeatures)
	print "Training MSE:" + str(result[0])
	print "Test MSE:" + str(result[1])

def featureExpansion(featureMatrix, targetMatrix, featureNames, trainingIndex, testIndex):
	trainFeatureForFS = pd.DataFrame(featureMatrix,index = trainingIndex)
	trainTargetForFS = pd.DataFrame(targetMatrix,index = trainingIndex)

	testFeatureForFS = pd.DataFrame(featureMatrix,index = testIndex)
	testTargetForFS = pd.DataFrame(targetMatrix,index = testIndex)

	normDictForFS = calculateNormDict(trainFeatureForFS, featureNames)

	newFeatureList = []
	for featureSet in itertools.combinations(featureNames, 2):
		correlatedFeatures = list(featureSet)
		feature = ",".join(correlatedFeatures)
		newFeatureList.append(feature)
	for f in featureNames:
		newFeatureList.append(f+","+f)

	newTrainFeature = trainFeatureForFS.copy()
	newTrainFeature = standardize(newTrainFeature, normDictForFS, featureNames)

	newTestFeature = testFeatureForFS.copy()
	newTestFeature = standardize(newTestFeature, normDictForFS, featureNames)

	for col in newFeatureList:
		f = col.split(",")
		f1 = f[0]
		f2 = f[1]
		trainFeatureForFS[col] = newTrainFeature[f1]*newTrainFeature[f2]
		testFeatureForFS[col] = newTestFeature[f1]*newTestFeature[f2]

	for f in featureNames:
		newFeatureList.append(f)

	normDictForFS = calculateNormDict(trainFeatureForFS, newFeatureList)
	result = linearRegression(trainFeatureForFS, trainTargetForFS, testFeatureForFS, testTargetForFS, normDictForFS, newFeatureList)
	print "==========   Feature Expansion   ============"
	print "Training MSE:" + str(result[0])
	print "Test MSE:" + str(result[1])


if __name__ == "__main__":

	boston = load_boston()
	featureNames = boston.feature_names
	featureMatrix = pd.DataFrame(boston.data, columns= featureNames)
	featureMatrix["x0"] = 1
	targetMatrix = pd.DataFrame(boston.target,columns=["target"])

	testIndex = list(range(0, featureMatrix.index.size, 7))
	totalIndex = list(range(0,featureMatrix.index.size))
	trainingIndex = [index for index in totalIndex if index not in testIndex]

	testFeature = pd.DataFrame(featureMatrix,index = testIndex)
	trainingFeature = pd.DataFrame(featureMatrix,index = trainingIndex)

	testTarget =  pd.DataFrame(targetMatrix,index = testIndex)
	trainingTarget = pd.DataFrame(targetMatrix,index = trainingIndex)
	
	normDict = calculateNormDict(trainingFeature, featureNames)

	print "==========   Histogram  ============"
	colNo = 0
	for f in featureNames:
		print "Plotting histogram for feature (Bins = 10): " + f 
		subplot = plt.subplot(3,5,colNo+1)
		subplot.hist(trainingFeature[f],10)
		subplot.set_xlabel(f)
		subplot.set_ylabel("Frequency")
		colNo=colNo + 1
	plt.show()

	print "==========   Pearson Correlation with Target Values ============"
	correlationDict = findPearsonCorrelation(trainingFeature, trainingTarget, featureNames, False)
	for f in featureNames:
		print "Feature: "+str(f) + " Coeff.: "+ str(correlationDict[f])

	print "==========   Linear Regression ============"
	result = linearRegression(trainingFeature, trainingTarget, testFeature, testTarget, normDict, featureNames)
	print "Training MSE:" + str(result[0])
	print "Test MSE:" + str(result[1])

	ridgeRegression(trainingFeature, trainingTarget, testFeature, testTarget, normDict, featureNames)
	ridgeRegression_with_CV(featureMatrix, targetMatrix, featureNames, trainingIndex,testIndex)
	featureSelection(featureMatrix, targetMatrix, featureNames, trainingIndex, testIndex)
	featureSelectionWithResidue(featureMatrix, targetMatrix, featureNames, trainingIndex, testIndex)
	featureSelectionWithBruteForce(featureMatrix, targetMatrix, featureNames, trainingIndex, testIndex)
	featureExpansion(featureMatrix, targetMatrix, featureNames, trainingIndex, testIndex)
