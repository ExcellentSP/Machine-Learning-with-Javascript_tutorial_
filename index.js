require('@tensorflow/tfjs-node')
const tf = require('@tensorflow/tfjs')
const plot = require('node-remote-plot')
const loadCSV = require('./load-csv')
const LinearRegression = require('./LinearRegression')

let { features, labels, testFeatures, testLabels } = loadCSV('./cars.csv', {
	shuffle: true,
	splitTest: 50,
	dataColumns: ['horsepower', 'modelyear', 'weight'],
	labelColumns: ['mpg'],
})

const regression = new LinearRegression(features, labels, {
	learningRate: 0.1,
	maxIterations: 30,
})

regression.train()
const r2 = regression.test(testFeatures, testLabels)

plot({
	x: regression.mseHistory.reverse(),
	xLabel: 'Iteration #',
	yLabel: 'MSE',
	name: 'plots/mse-per-iteration'
})

// plot({
// 	x: regression.bHistory,
// 	xLabel: 'B Value',
// 	y: regression.mseHistory.reverse(),
// 	yLabel: 'MSE',
// 	name: 'mse-per-b'
// })

console.log('r2 is: ', r2)