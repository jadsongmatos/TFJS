const surface = { name: "show.fitCallbacks", tab: "Training" };

var model;

var m_predict;

// Build and compile model.
async function start() {
  console.log("Backend", tf.getBackend());

   //adam / sgd / rmsprop / adamax
  //meanSquaredError / meanAbsoluteError /
  model = await tf.loadLayersModel(
    "../models/meanSquaredError/adam/1,-1/2-2-1-2-2/0/mymodel.json"
  );

  model.summary();

  // Run inference with predict().
  m_predict = await model
    .predict(
      tf.tensor2d([
        [3, 4],
        [7, 9],
        [-128, -1808],
      ])
    )
    .array();

  console.log(m_predict);

  const predictedPoints = m_predict.map((val, i) => {
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
