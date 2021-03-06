const size = Math.pow(2, 8); //Math.pow(2, 17);
const size_m = 64;
const train = 0.1;
var seed = 0;
var m_predict;
var m_evaluate;
var i_history = 0;
var history_train = [];

// Generate some synthetic data for training.
var out2Tensor = tf.fill([size, size_m / 2], 0);

var inputTensor = tf.randomUniform([size, size_m], -2, 3, "int32", seed);
seed++;

const testOut1 = inputTensor.slice([0], [3]);
const testOut2 = tf.fill([3, 1, size_m / 2], 0, "bool");

//const surface = { name: "show.fitCallbacks", tab: "Training" };
const surface = { name: "show.history live", tab: "Training" };

const input = tf.input({ shape: [size_m] });
const input_a = tf.layers.dense({ units: size_m }).apply(input);

const med = tf.layers
  .dense({
    units: size_m / 2,
  })
  .apply(input_a);

const out1 = tf.layers.dense({ units: size_m }).apply(med);

const model = tf.model({ inputs: input, outputs: [out1, med] });

model.summary();
tfvis.show.modelSummary(surface, model);

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
    epochs: 1024, //26075,
    shuffle: false,
    //validationData: validation_data,
    callbacks: {
      onEpochEnd: (epoch, log) => {
        //history_train[i_history % 256] = log;
        //i_history++;
        history_train.push(log)

        inputTensor = tf.randomUniform([size, size_m], 0, 2, "int32", seed);
        seed++;

        console.log("Epoch:", epoch, "Loss:", log.loss);
        tfvis.show.history(surface, history_train, [
          "dense_Dense3_loss",
          "dense_Dense3_acc",
        ]);
      },
    },
  });

  model.summary();

  m_evaluate = await model.evaluate(inputTensor, [inputTensor, out2Tensor], {
    batchSize: size,
  });

  console.log("evaluate", m_evaluate);
  m_evaluate[0].print();
  m_evaluate[1].print();
  m_evaluate[2].print();
  m_evaluate[3].print();
  m_evaluate[4].print();
  m_evaluate[5].print();

  // Run inference with predict().
  m_predict = await model.predict(testOut1);
  console.log("predict", m_predict);
  testOut1.print();
  m_predict[0].print();
  m_predict[1].print();
}

// Create a basic regression model
tf.setBackend("webgl").then(() => start());
