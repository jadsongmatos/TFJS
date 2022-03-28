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
const inputTensor = tf.randomUniform(
  [65536, 2],
  float_MinValue,
  float_MaxValue
);
//tf.tensor2d([[1], [2], [3], [4]], [4, 1]);

const inputMax = inputTensor.max();
const inputMin = inputTensor.min();
const normalizedInputs = inputTensor.sub(inputMin).div(inputMax.sub(inputMin));

//const normalizedInputs = inputTensor.softmax();

const surface = { name: "show.fitCallbacks", tab: "Training" };

var model;

var m_predict;

// Build and compile model.
async function start() {
  console.log("Backend", tf.getBackend());

  model = await tf.loadLayersModel(
    "http://127.0.0.1:5500/models/meanSquaredError/sgd/2-1-2/mymodel.json"
  );

  model.summary();

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

  const predictedPoints = m_predict[0].map((val, i) => {
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
