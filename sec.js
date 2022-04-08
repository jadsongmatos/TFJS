const float_MaxValue = 1; //Math.pow(2, 112)//3.4028235e38;//65500.0
const float_MinValue = -1; //-Math.pow(2,112)//-3.4028235e38;//-65500.0
const size = Math.pow(2, 17);
var train = 0.1;
var seed = 131074;

// Generate some synthetic data for training.
var inputTensor = tf.randomUniform(
  [size, 2],
  float_MinValue,
  float_MaxValue,
  "float32"
  //seed
);

const validation_data = tf.randomUniform(
  [128, 2],
  float_MinValue,
  float_MaxValue,
  "float32"
);

seed++;

//const surface = { name: "show.fitCallbacks", tab: "Training" };
const surface = { name: "show.history live", tab: "Training" };

const model = tf.sequential();

// Build a sequential model
//model.add(tf.layers.dense({ units: 2, inputShape: [2] }));
model.add(tf.layers.dense({ units: 2, inputShape: [2] }));
model.add(tf.layers.dense({ units: 2 }));
model.add(tf.layers.dense({ units: 1 }));
model.add(tf.layers.dense({ units: 2 }));
// Add an output layer
model.add(tf.layers.dense({ units: 2 }));

model.summary();

tfvis.show.modelSummary(surface, model);

var m_predict;
var result;

var history_train = [];

function lossC(labels,predictions){
  console.log(labels,predictions)
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
    loss: lossC,
    metrics: ["acc"],
  });

  // Train model with fit().
  await model.fit(inputTensor, inputTensor, {
    batchSize: size,
    epochs: 128,//1024,
    //shuffle: true,
    validationData: validation_data,
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

  const test = tf.tensor2d([
    [3, 4],
    [7, 9],
    [-128, -1808],
  ]);

  result = await model.evaluate(test, test, {
    batchSize: 3,
  });

  result[0].print();

  // Run inference with predict().
  m_predict = await model.predict(test);
  m_predict.print();

  const predictedPoints = m_predict.arraySync().map((val, i) => {
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
