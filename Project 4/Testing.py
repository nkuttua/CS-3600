from NeuralNetUtil import buildExamplesFromCarData,buildExamplesFromPenData
from NeuralNet import buildNeuralNet
from math import pow, sqrt

def average(argList):
    return sum(argList)/float(len(argList))

def stDeviation(argList):
    mean = average(argList)
    diffSq = [pow((val-mean),2) for val in argList]
    return sqrt(sum(diffSq)/len(argList))

penData = buildExamplesFromPenData()
def testPenData(hiddenLayers = [24]):
    return buildNeuralNet(penData, maxItr = 200, hiddenLayerList = hiddenLayers)

carData = buildExamplesFromCarData()
def testCarData(hiddenLayers = [16]):
    return buildNeuralNet(carData, maxItr = 200,hiddenLayerList = hiddenLayers)

testCarData()

## TESTING

q5penData = []
q5carData = []

for n in range(5):
	nnet, testAccuracy = testPenData()
	q5penData.append(testAccuracy)

	nnet2, testAccuracy2 = testCarData()
	q5carData.append(testAccuracy2)

print("Pen Data", q5penData)
print("max: ", max(q5penData))
print("average: ", average(q5penData))
print("stDev: ", stDeviation(q5penData))

print("\n")

print("Car Data", q5carData)
print("max: ", max(q5carData))
print("average: ", average(q5carData))
print("stDev: ", stDeviation(q5carData))


pen_output = {}
car_output = {}
perceptrons = 0
while perceptrons <= 40:
	q6penData = []
	q6carData = []

	for n in range(5):
		nnet, testAccuracy = testPenData(hiddenLayers = [perceptrons])
		q6penData.append(testAccuracy)

		nnet2, testAccuracy2 = testCarData(hiddenLayers = [perceptrons])
		q6carData.append(testAccuracy2)

	pen_stats = {}
	pen_stats['max'] = max(q6penData)
	pen_stats['average'] = average(q6penData)
	pen_stats['stDev'] = stDeviation(q6penData)
	pen_output[perceptrons] = (q6penData, pen_stats)

	car_stats = {}
	car_stats['max'] = max(q6carData)
	car_stats['average'] = average(q6carData)
	car_stats['stDev'] = stDeviation(q6carData)
	car_output[perceptrons] = (q6carData, car_stats)

	perceptrons += 5

print("Q6 Numbers")
print("\nPEN DATA")
for p in pen_output:
    print(f"Perceptrons: {p}")
    print(pen_output[p])
print("\nCAR DATA")
for p in car_output:
    print(f"Perceptrons: {p}")
    print(car_output[p])