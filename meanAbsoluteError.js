const float_MaxValue = 3.40282347e38; // largest positive number in float32
const float_MinValue = -3.40282347e38;

const callbacks = {
  onEpochEnd: async (epoch, logs) => {
    tfvis.show.fitCallbacks(epoch, logs);
    console.log("epoch: " + epoch + JSON.stringify(logs));
  },
};

const onTrainBegin = (logs) => {
  console.log("onTrainBegin");
};

// Generate some synthetic data for training.
const inputTensor = tf.randomUniform([4096, 2], float_MinValue, float_MaxValue);
//tf.tensor2d([[1], [2], [3], [4]], [4, 1]);

const inputMax = inputTensor.max();
const inputMin = inputTensor.min();
const normalizedInputs = inputTensor.sub(inputMin).div(inputMax.sub(inputMin));

//const normalizedInputs = inputTensor.softmax();

const surface = { name: "show.fitCallbacks", tab: "Training" };

const model = tf.sequential();

var m_predict;

// Build and compile model.
async function start() {
  // Build a sequential model
  //model.add(tf.layers.dense({ units: 2, inputShape: [2] }));

  model.add(tf.layers.dense({ units: 4, inputShape: [2] }));

  model.add(tf.layers.dense({ units: 8 }));
  model.add(tf.layers.dense({ units: 4 }));
  model.add(tf.layers.dense({ units: 2 }));

  model.add(tf.layers.dense({ units: 1 }));

  model.add(tf.layers.dense({ units: 2 }));
  model.add(tf.layers.dense({ units: 4 }));
  model.add(tf.layers.dense({ units: 8 }));
  model.add(tf.layers.dense({ units: 4 }));

  // Add an output layer
  model.add(tf.layers.dense({ units: 2 }));

  //adam / sgd / adagrad / rmsprop / adamax
  //meanSquaredError / meanAbsoluteError /
  //0.000001
  model.compile({
    optimizer: tf.train.adam(0.01),
    loss: "meanAbsoluteError",
    metrics: ["acc","precision"],
  });

  model.summary();

  tfvis.show.modelSummary(surface, model);

  // Train model with fit().
  await model.fit(normalizedInputs, normalizedInputs, {
    batchSize: 512,
    epochs: 512,
    //shuffle: true,
    callbacks: tfvis.show.fitCallbacks(surface, ["loss", "acc","precision"]), //callbacks
  });

  // Run inference with predict().
  m_predict = await model.predict(tf.tensor2d([[3, 4]])).array();
  console.log(m_predict);

  const predictedPoints = m_predict.map((val, i) => {
    return { x: i, y: val[0] };
  });

  tfvis.render.scatterplot(
    { name: "Model Predictions vs Original Data" },
    {
      values: [normalizedInputs, predictedPoints],
      series: ["original", "predicted"],
    }
  );
}

// Create a basic regression model
start();
