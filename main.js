const float_MaxValue = 3.4028235e38;
const float_MinValue = -3.4028235e38;
const size = Math.pow(2, 16);
const train = 0.1;
var seed = 0;

// Generate some synthetic data for training.
var inputTensor = tf.randomUniform(
  [size, 2],
  float_MinValue,
  float_MaxValue,
  "float32",
  seed
);
seed++;
//tf.tensor2d([[1], [2], [3], [4]], [4, 1]);

const inputMax = tf.tensor(float_MaxValue); //inputTensor.max();
const inputMin = tf.tensor(float_MinValue); //inputTensor.min();
const float_MValue = tf.tensor(6.805647e38); // inputMax.sub(inputMin)
var normalizedInputs = inputTensor.sub(inputMin).div(float_MValue);

//const normalizedInputs = inputTensor.softmax();

//const surface = { name: "show.fitCallbacks", tab: "Training" };
const surface = { name: "show.history live", tab: "Training" };

const model = tf.sequential();

// Build a sequential model
//model.add(tf.layers.dense({ units: 2, inputShape: [2] }));
model.add(tf.layers.dense({ units: 2, inputShape: [2] }));
model.add(tf.layers.dense({ units: 1 }));
model.add(tf.layers.dense({ units: 2 }));
// Add an output layer
model.add(tf.layers.dense({ units: 2 }));

model.summary();

tfvis.show.modelSummary(surface, model);

var m_predict;

var history_train = [];

// Build and compile model.
async function start() {
  console.log("Backend", tf.getBackend());

  //adam / sgd / rmsprop / adamax
  //meanSquaredError / meanAbsoluteError /
  //0.000001
  model.compile({
    optimizer: tf.train.adam(train),
    loss: "meanAbsoluteError",
    metrics: ["acc"],
  });

  // Train model with fit().
  await model.fit(normalizedInputs, normalizedInputs, {
    batchSize: size,
    epochs: 1024,
    //shuffle: true,
    callbacks: {
      onEpochEnd: (epoch, log) => {
        history_train.push(log);
        console.log("Epoch: " + epoch + " Loss: " + log.loss);

        inputTensor = tf.randomUniform(
          [size, 2],
          float_MinValue,
          float_MaxValue,
          "float32",
          seed
        );

        seed++;

        normalizedInputs = inputTensor.sub(inputMin).div(float_MValue);

        console.log("Epoch:", epoch, "Loss:", log.loss);
        tfvis.show.history(surface, history_train, ["loss","acc"]);
      },
    },
  });

  // Run inference with predict().
  m_predict = await model
    .predict(
      tf.tensor2d([
        [3, 4],
        [7, 9],
      ])
    )
    .array();
  console.log(m_predict);

  const predictedPoints = m_predict.map((val, i) => {
    return { x: val[0], y: val[0] };
  });

  tfvis.render.scatterplot(
    { name: "Model Predictions vs Original Data" },
    {
      values: [
        [
          { x: 3, y: 4 },
          { x: 7, y: 9 },
        ],
        predictedPoints,
      ],
      series: ["original", "predicted"],
    }
  );
}

// Create a basic regression model
tf.setBackend("wasm").then(() => start());
