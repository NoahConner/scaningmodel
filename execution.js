const fs = require('fs');
const tf = require('@tensorflow/tfjs-node');

const IMAGE_HEIGHT = 100;
const IMAGE_WIDTH = 100;
const MODEL_PATH = 'file://./saved-model/model.json';

async function loadModel() {
  try {
    const model = await tf.loadLayersModel(MODEL_PATH);
    return model;
  } catch (error) {
    console.error('Error loading model:', error);
    return null;
  }
}

async function loadImage(fileName) {
  try {
    const imageBuffer = fs.readFileSync(fileName);
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

async function predictImageLabel(imageFileName) {
  const model = await loadModel();
  if (!model) {
    console.error('Model not loaded.');
    return null;
  }

  const image = await loadImage(imageFileName);
  if (!image) {
    console.error('Image not loaded.');
    return null;
  }

  const preprocessedImage = preprocessImage(image);
  const prediction = model.predict(preprocessedImage.expandDims(0));
  const predictedClass = prediction.argMax(-1).dataSync()[0];
  
  // Mapping predicted class index to label
  const dataset = [
    { fileName: './dataset/wave2.jpeg', label: 'Wave 2' },
    { fileName: './dataset/wave3.png', label: 'Wave 3' },
    { fileName: './dataset/wave4.jpeg', label: 'Wave 4' },
  ];
  const predictedLabel = dataset[predictedClass].label;

  return predictedLabel;
}

// Usage example
const imageFileName = './samples/wave2.jpeg'; // Change to your image file
predictImageLabel(imageFileName).then((label) => {
  if (label) {
    console.log('Predicted label:', label);
  } else {
    console.log('Failed to predict label.');
  }
});
