const generator = tf.sequential();
const discriminator = tf.sequential();

generator.add(
  tf.layers.dense({ units: 256, inputShape: [100], activation: "relu" })
);
generator.add(tf.layers.dense({ units: 500, activation: "relu" }));
generator.add(tf.layers.dense({ units: 500, activation: "relu" }));
generator.add(tf.layers.dense({ units: 500, activation: "relu" }));
generator.add(tf.layers.dense({ units: 1000, activation: "softmax" }));
generator.compile({ loss: "binaryCrossentropy", optimizer: "adam" });

discriminator.add(
  tf.layers.dense({ units: 256, inputShape: [1000], activation: "relu" })
);
discriminator.add(tf.layers.dense({ units: 500, activation: "relu" }));
discriminator.add(tf.layers.dense({ units: 500, activation: "relu" }));
discriminator.add(tf.layers.dense({ units: 500, activation: "relu" }));
discriminator.add(tf.layers.dense({ units: 1, activation: "sigmoid" }));
discriminator.compile({ loss: "binaryCrossentropy", optimizer: "adam" });

const gan = tf.sequential();
gan.add(generator);
gan.add(discriminator);
gan.compile({ loss: "binaryCrossentropy", optimizer: "adam" });

async function training(epochs, batchSize) {
  const xs = tf.randomNormal([batchSize, 100]);
  let noise = tf.randomNormal([batchSize, 100]);

  const ys = tf.ones([batchSize, 1]);

  for (let i = 0; i < epochs; i++) {
    let noise = tf.randomNormal([batchSize, 100]);

    let generatedImages = generator.predict(noise);

    let combinedImages = tf.concat([generatedImages, xs], 0);

    discriminator.fit(combinedImages, [ys, ys], {
      batchSize: batchSize,
      epochs: 1,
      callbacks: {
        onBatchEnd: function (batch, logs) {
          if (batch % 100 === 0) {
            console.log("epoch: " + i + " Discriminator Loss: " + logs.loss);
          }
        },
      },
    });
    noise = tf.randomNormal([batchSize, 100]);
    gan.fit(noise, ys, {
      batchSize: batchSize,
      epochs: 1,
      callbacks: {
        onBatchEnd: function (batch, logs) {
          console.log("epoch: " + i + " Gan Loss: " + logs.loss);
        },
      },
    });
  }
}

// Create a basic regression model
tf.setBackend("wasm").then(() => training(100, 1000));
