const tf = require('@tensorflow/tfjs')

class LinearRegression {
	constructor(features, labels, options) {
		this.features = this.processFeatures(features)
		this.labels = tf.tensor(labels)
		this.mseHistory = []
		this.bHistory = []

		this.options = {
			learningRate: 0.1,
			maxIterations: 1000,
			...options,
		}

		this.weights = tf.zeros([this.features.shape[1], 1])
	}

	gradientDescent() {
		const currentGuesses = this.features.matMul(this.weights)
		const differences = currentGuesses.sub(this.labels)
		const gradients = this.features
			.transpose()
			.matMul(differences)
			.div(this.features.shape[0])

		this.weights = this.weights.sub(
			gradients.mul(this.options.learningRate)
		)

		// W/o tensorflow
		// let bDerivSum = 0
		// let mDerivSum = 0
		// for (let i = 0; i < this.features.length; i++) {
		// 	bDerivSum +=
		// 		this.m * this.features[i][0] + this.b - this.labels[i][0]
		// 	mDerivSum +=
		// 		-1 *
		// 		this.features[i][0] *
		// 		(this.labels[i][0] - (this.m * this.features[i][0] + this.b))
		// }
		// const bSlope = (2 * bDerivSum) / this.features.length
		// const mSlope = (2 * mDerivSum) / this.features.length
		// this.b = this.b - bSlope * this.options.learningRate
		// this.m = this.m - mSlope * this.options.learningRate
	}

	train() {
		for (let i = 0; i < this.options.maxIterations; i++) {
			this.bHistory.push(this.weights.get(0,0))
			this.gradientDescent()
			this.recordMSE()
			this.updateLearningRate()
		}
	}

	test(testFeatures, testLabels) {
		const ttFeatures = this.processFeatures(testFeatures)
		const ttLabels = tf.tensor(testLabels)
		
		const predictions = ttFeatures.matMul(this.weights)

		predictions.print()
		
		const res = ttLabels.sub(predictions).pow(2).sum().get()
		const tot = ttLabels.sub(ttLabels.mean()).pow(2).sum().get()
		
		return 1 - (res / tot)
	}

	predict() {}

	processFeatures(features){
		const tFeatures = tf.tensor(features)
		const standardizedFeatures = this.standardize(tFeatures)
		const onesFeatures = tf.ones([standardizedFeatures.shape[0], 1]).concat(standardizedFeatures, 1)
		return onesFeatures
	}

	standardize(features) {
		const { mean, variance } = tf.moments(features, 0)

		if(!this.mean && !this.variance){
			this.mean = mean
			this.variance = variance
		}

		return features.sub(this.mean).div(this.variance.sqrt())
	}

	recordMSE() {
		const mse = this.features.matMul(this.weights).sub(this.labels).pow(2).sum().div(this.features.shape[0]).get()
		this.mseHistory.unshift(mse)
	}

	updateLearningRate() {
		if (this.mseHistory.length < 2) {
			return
		}

		if (this.mseHistory[0] > this.mseHistory[1]) {
			this.options.learningRate /= 2
		} else if (this.mseHistory[0] < this.mseHistory[1]){
			this.options.learningRate *= 1.05
		}
	}
}

module.exports = LinearRegression
