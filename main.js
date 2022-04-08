const float_MaxValue = 1; //Math.pow(2, 112)//3.4028235e38;//65500.0
const float_MinValue = -1; //-Math.pow(2,112)//-3.4028235e38;//-65500.0
const size = Math.pow(2, 17);
const train = 0.1;
var seed = 0;

// Generate some synthetic data for training.
var inputTensor = tf.randomUniform(
  [size, 2],
  float_MinValue,
  float_MaxValue,
  "float32"
  //seed
);

//var out2Tensor = tf.tensor([0,0])
var out2Tensor = tf.fill([size, 1], 0);

const validation_data = tf.randomUniform(
  [128, 2],
  float_MinValue,
  float_MaxValue,
  "float32"
);

seed++;

//const surface = { name: "show.fitCallbacks", tab: "Training" };
const surface = { name: "show.history live", tab: "Training" };

const input = tf.input({ shape: [2] });
const dense0 = tf.layers.dense({ units: 2 }).apply(input);
const dense1 = tf.layers.dense({ units: 2 }).apply(dense0);
const dense2 = tf.layers.dense({ units: 1 }).apply(dense1);
const dense3 = tf.layers.dense({ units: 2 }).apply(dense2);
const dense4 = tf.layers.dense({ units: 2 }).apply(dense3);

const model = tf.model({ inputs: input, outputs: [dense4, dense2] });

model.summary();

tfvis.show.modelSummary(surface, model);

var m_predict;
var result;

var history_train = [];

const testOut1 = tf.tensor2d([
  [3, 4],
  [7, 9],
  [-128, -1808],
]);

const testOut2 = tf.tensor([0, 0, 0]);

function lossC(labels, predictions, weights, reduction) {
  labels.print();
  predictions.print();
  //console.log(labels.print(),predictions.print())
  return tf.metrics.meanSquaredError(labels, predictions);
}

// Build and compile model.
async function start() {
  console.log("Backend", tf.getBackend());

  //adam / sgd / rmsprop / adamax
  //meanSquaredError / meanAbsoluteError
  //0.000001
  model.compile({
    optimizer: tf.train.adam(train),
    loss: "meanSquaredError", //lossC,
    metrics: ["acc"],
  });

  // Train model with fit().
  await model.fit(inputTensor, [inputTensor, out2Tensor], {
    batchSize: size,
    epochs: 1024,
    //shuffle: true,
    //validationData: validation_data,
    callbacks: {
      onEpochEnd: (epoch, log) => {
        history_train.push(log);
        console.log("Epoch: " + epoch + " Loss: " + log.loss);

        inputTensor = tf.randomUniform(
          [size, 2],
          float_MinValue,
          float_MaxValue,
          "float32"
          // seed
        );
        //seed++;

        console.log("Epoch:", epoch, "Loss:", log.loss);
        tfvis.show.history(surface, history_train, ["loss", "acc"]);
      },
    },
  });

  model.summary();

  result = await model.evaluate(testOut1, [testOut1, testOut2], {
    batchSize: 3,
  });

  console.log("evaluate");
  result[0].print()
  result[1].print()
  result[2].print()
  result[3].print()
  result[4].print()
  result[5].print()

  // Run inference with predict().
  console.log("predict");
  m_predict = await model.predict(testOut1);
  m_predict[0].print();
  m_predict[1].print();

  const predictedPoints = m_predict[0].arraySync().map((val, i) => {
    return { x: val[0], y: val[1] };
  });

  tfvis.render.scatterplot(
    { name: "Model Predictions vs Original Data" },
    {
      values: [
        [
          { x: 3, y: 4 },
          { x: 7, y: 9 },
          { x: -128, y: -1808 },
        ],
        predictedPoints,
      ],
      series: ["original", "predicted"],
    }
  );
}

// Create a basic regression model
tf.setBackend("wasm").then(() => start());
