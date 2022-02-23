const callbacks = {
  onEpochEnd: async (epoch, logs) => {
    tfvis.show.fitCallbacks(epoch, logs);
    console.log("epoch: " + epoch + JSON.stringify(logs));
  },
};

// Generate some synthetic data for training.
const xs = tf.tensor2d([[1], [2], [3], [4]], [4, 1]);
const ys = tf.tensor2d([[1], [3], [5], [7]], [4, 1]);

const surface = { name: "show.fitCallbacks", tab: "Training" };

// Build and compile model.
async function basicRegression() {
  // Build a sequential model
  const model = tf.sequential();
  model.add(tf.layers.dense({ units: 1, inputShape: [1], useBias: true }));
  // Add an output layer
  model.add(tf.layers.dense({ units: 1, useBias: true }));

  model.compile({ optimizer: "sgd", loss: "meanSquaredError" });

  tfvis.show.modelSummary(surface, model);

  // Train model with fit().
  await model.fit(xs, ys, {
    batchSize: 32,
    epochs: 50,
    shuffle: true,
    callbacks: tfvis.show.fitCallbacks(surface, ["loss", "acc"]), //callbacks
  });

  // Run inference with predict().
  const xss = await model
    .predict(tf.tensor1d([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]))
    .array();
  console.log(xss);

  const predictedPoints = xss.map((val, i) => {
    return { x: i, y: val[0] };
  });

  tfvis.render.scatterplot(
    { name: "Model Predictions vs Original Data" },
    {
      values: [
        [
          { x: 1, y: 1 },
          { x: 2, y: 3 },
          { x: 3, y: 5 },
          { x: 4, y: 7 },
        ],
        predictedPoints,
      ],
      series: ["original", "predicted"],
    }
  );
}

// Create a basic regression model
basicRegression();
