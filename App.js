import React, { useEffect, useState } from 'react';
import { StyleSheet, Text, View, Image, Button, ActivityIndicator, Alert } from 'react-native';
import * as tf from '@tensorflow/tfjs';
import { bundleResourceIO } from '@tensorflow/tfjs-react-native';
import * as jpeg from 'jpeg-js';
import Svg, { Rect } from 'react-native-svg';

export default function App() {
  const imageUrl =
    'https://raw.githack.com/JoaoVitorAguiar/object-detection-app/main/assets/images_test/image.jpg';
  const [model, setModel] = useState(null);
  const [objects, setObjects] = useState([]);
  const [loading, setLoading] = useState(true);
  const [tensorDims, setTensorDims] = useState({ width: 0, height: 0 });

  useEffect(() => {
    const loadData = async () => {
      try {
        await loadModel();
        setLoading(false);
      } catch (error) {
        console.error('Error:', error);
        Alert.alert('Error', 'An error occurred while loading the model.');
      }
    };

    loadData();

    return () => {
      if (model) {
        model.dispose();
      }
    };
  }, [model]);

  const loadModel = async () => {
    try {
      await tf.ready();
      const modelJson = require('./assets/models/model.json');
      const modelWeights = require('./assets/models/group1-shard.bin');
      const loadedModel = await tf.loadGraphModel(bundleResourceIO(modelJson, modelWeights));
      setModel(loadedModel);
    } catch (error) {
      console.error('Error loading TensorFlow model:', error);
      throw error;
    }
  };

  const imageToTensor = (rawImageData) => {
    const TO_UINT8ARRAY = true;
    const { width, height, data } = jpeg.decode(rawImageData, TO_UINT8ARRAY);
    const buffer = new Uint8Array(width * height * 3);
    let offset = 0;
    for (let i = 0; i < buffer.length; i += 3) {
      buffer[i] = data[offset];
      buffer[i + 1] = data[offset + 1];
      buffer[i + 2] = data[offset + 2];
      offset += 4;
    }
    return { tensor: tf.tensor3d(buffer, [height, width, 3]), width, height };
  };

  const classifyImage = async () => {
    if (!model) {
      Alert.alert('Error', 'The model is not loaded yet.');
      return;
    }

    if (!imageUrl) {
      Alert.alert('Error', 'No image URL available.');
      return;
    }

    try {
      setLoading(true);

      const response = await fetch(imageUrl);
      const rawImageData = await response.arrayBuffer();
      const { tensor, width, height } = imageToTensor(rawImageData);

      if (tensor) {
        const resizedTensor = tf.image
          .resizeBilinear(tensor, [640, 640])
          .reshape([1, 640, 640, 3])
          .div(255.0);

        const predictions = await model.executeAsync(resizedTensor);

        const boxes = predictions[0].dataSync();
        const scores = predictions[1].dataSync();
        const classes = predictions[2].dataSync();

        const maxScoreIndex = scores.indexOf(Math.max(...scores));

        if (scores[maxScoreIndex] > 0.5) {
          const maxScoreObject = {
            ymin: boxes[maxScoreIndex * 4] * height,
            xmin: boxes[maxScoreIndex * 4 + 1] * width,
            ymax: boxes[maxScoreIndex * 4 + 2] * height,
            xmax: boxes[maxScoreIndex * 4 + 3] * width,
            class: classes[maxScoreIndex],
            score: scores[maxScoreIndex],
          };

          setObjects([maxScoreObject]);
          setTensorDims({ width, height });
        }

        setLoading(false);
      }
    } catch (error) {
      console.error('Error classifying the image:', error);
      Alert.alert('Error', 'An error occurred while classifying the image.');
      setLoading(false);
    }
  };

  return (
    <View style={styles.container}>
      {loading ? (
        <ActivityIndicator size="large" color="#0000ff" />
      ) : (
        <>
          <View style={styles.imageContainer}>
            <Image source={{ uri: imageUrl }} style={styles.image} />
            <Svg height="100%" width="100%" viewBox={`0 0 ${tensorDims.width} ${tensorDims.height}`}>
              {objects.map((object, index) => (
                <Rect
                  key={index}
                  x={object.xmin}
                  y={object.ymin}
                  width={object.xmax - object.xmin}
                  height={object.ymax - object.ymin}
                  stroke="red"
                  strokeWidth="20"
                  fill="none"
                />
              ))}
            </Svg>
          </View>
          <Button title="Classify Image" onPress={classifyImage} color="#841584" />
        </>
      )}
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    backgroundColor: '#F5FCFF',
  },
  imageContainer: {
    position: 'relative',
    width: '90%',
    height: '50%',
    marginBottom: 20,
    borderRadius: 10,
    overflow: 'hidden',
  },
  image: {
    position: 'absolute',
    width: '100%',
    height: '100%',
  },
});
