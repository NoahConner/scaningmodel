const fs = require('fs');
const tf = require('@tensorflow/tfjs-node');

const dataset = [
  { fileName: './dataset/wave2.jpeg', label: 'Wave 2' },
  { fileName: './dataset/wave3.png', label: 'Wave 3' },
  { fileName: './dataset/wave4.jpeg', label: 'Wave 4' },
];

const IMAGE_HEIGHT = 100;
const IMAGE_WIDTH = 100;
const NUM_CLASSES = dataset.length;

async function loadAndPreprocessImages() {
  const images = [];
  const labels = [];

  for (const { fileName, label } of dataset) {
    const image = await loadImage(fileName);
    const preprocessedImage = preprocessImage(image);
    images.push(preprocessedImage);
    labels.push(label);
  }

  const imagesTensor = tf.stack(images);
  const labelsTensor = tf.tensor1d(labels.map(label => dataset.findIndex(item => item.label === label)), 'int32');

  return { imagesTensor, labelsTensor };
}

async function loadImage(fileName) {
  try {
    const filePath = `${__dirname}/${fileName}`; // Assuming dataset directory is in the same directory as this script
    const imageBuffer = fs.readFileSync(filePath);
    const imageTensor = tf.node.decodeImage(imageBuffer);
    return imageTensor;
  } catch (error) {
    console.error('Error loading image:', error);
    return null;
  }
}

function preprocessImage(image) {
  return tf.tidy(() => {
    const resizedImage = tf.image.resizeBilinear(image, [IMAGE_HEIGHT, IMAGE_WIDTH]);
    return resizedImage.toFloat().div(255);
  });
}

function defineModel() {
  const model = tf.sequential();
  model.add(tf.layers.flatten({ inputShape: [IMAGE_HEIGHT, IMAGE_WIDTH, 3] }));
  model.add(tf.layers.dense({ units: 128, activation: 'relu' }));
  model.add(tf.layers.dense({ units: NUM_CLASSES, activation: 'softmax' }));
  return model;
}

async function trainModel() {
  const { imagesTensor, labelsTensor } = await loadAndPreprocessImages();
  const model = defineModel();

  model.compile({
    optimizer: 'adam',
    loss: 'sparseCategoricalCrossentropy',
    metrics: ['accuracy']
  });

  // Convert labelsTensor to float32
  const floatLabelsTensor = labelsTensor.toFloat();

  await model.fit(imagesTensor, floatLabelsTensor, { epochs: 10 });

  await model.save('file://./saved-model');
}

trainModel().then(() => {
  console.log('Model trained and saved.');
});
