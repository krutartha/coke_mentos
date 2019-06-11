import * as tf from '@tensorflow/tfjs';
import 'bootstrap/dist/css/bootstrap.css';
import { deflateRaw } from 'zlib';

const model = tf.sequential();
model.add(tf.layers.dense({units: 1, inputShape: [1]}));

const learningRate = 0.0000001;
const optimizer = tf.train.sgd(learningRate);

model.compile({loss: 'meanSquaredError', optimizer: optimizer});

//const xs = tf.tensor2d([311, 317, 312, 416, 415, 398, 518, 505, 506], [9, 1]);
//const ys = tf.tensor2d([111, 107.3, 111.2, 118.9, 114.2, 122.1, 127.5, 128.9, 128], [9, 1]);

//const xs = tf.tensor2d([100, 200,300, 400], [4, 1]);
//const ys = tf.tensor2d([200, 400, 600, 800], [4, 1]);

const xs = tf.tensor2d([313.3, 409.6, 509.6], [3, 1]);
//const ys = tf.tensor2d([109.9, 118.3, 128.1], [3, 1]);
const ys = tf.tensor2d([16.56, 24.05, 33.4], [3, 1]);

model.fit(xs, ys, {epochs: 500}).then(() => {
    document.getElementById('predictButton').disabled = false;
    document.getElementById('predictButton').innerText = "Predict";
});

document.getElementById('predictButton').addEventListener('click', (el, ev) => {
    let val = document.getElementById('inputValue').value;
    document.getElementById('output').innerText = model.predict(tf.tensor2d([val], [1,1]));



})

