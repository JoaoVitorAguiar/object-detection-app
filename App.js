import React, { useEffect, useState } from 'react';
import { StyleSheet, Text, View, Image, Button, ActivityIndicator, Alert } from 'react-native';
import * as tf from '@tensorflow/tfjs';
import { bundleResourceIO } from '@tensorflow/tfjs-react-native';
import * as jpeg from 'jpeg-js';

export default function App() {
  const imageUrl =
    'https://raw.githack.com/JoaoVitorAguiar/object-detection-app/main/assets/images_test/image.jpg';
  const [model, setModel] = useState(null);
  const [objects, setObjects] = useState([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const loadData = async () => {
      try {
        await loadModel();
        // await loadAndDisplayImage();
        setLoading(false);
      } catch (error) {
        console.error('Error:', error);
        Alert.alert('Error', 'An error occurred while loading the model or image.');
      }
    };

    loadData();

    return () => {
      // Clean up resources (e.g., release the model)
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
      throw error; // Propagate the error to the caller
    }
  };

  const imageToTensor = (rawImageData) => {
    try {
      const TO_UINT8ARRAY = true;
      const { width, height, data } = jpeg.decode(rawImageData, TO_UINT8ARRAY);
      // Drop the alpha channel info for mobilenet
      const buffer = new Uint8Array(width * height * 3);
      let offset = 0; // offset into the original data
      for (let i = 0; i < buffer.length; i += 3) {
        buffer[i] = data[offset];
        buffer[i + 1] = data[offset + 1];
        buffer[i + 2] = data[offset + 2];
        offset += 4;
      }
      return tf.tensor3d(buffer, [height, width, 3]);
    } catch (error) {
      console.error('Error decoding the image:', error);
      Alert.alert('Error', 'Unable to process the image.');
      return null;
    }
  };

  // const loadAndDisplayImage = async () => {
  //   try {
  //     const response = await fetch(imageUrl);

  //     if (!response.ok) {
  //       throw new Error('Failed to fetch image');
  //     }

  //     const rawImageData = await response.arrayBuffer();
  //     const tensor = imageToTensor(rawImageData);

  //     if (tensor) {
  //       // Process the image using TensorFlow.js (if needed)
  //       // You can perform TensorFlow.js operations here if necessary
  //     }
  //   } catch (error) {
  //     console.error('Error loading and displaying the image:', error);
  //     throw error; // Propagate the error to the caller
  //   }
  // };

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

    // Load and preprocess the image
    const response = await fetch(imageUrl);
    const rawImageData = await response.arrayBuffer();
    const tensor = imageToTensor(rawImageData);

    if (tensor) {
      // Resize the tensor to match the expected input shape [1, 640, 640, 3]
      const resizedTensor = tf.image.resizeBilinear(tensor, [640, 640]).reshape([1, 640, 640, 3]);

      // Make predictions using the model (executeAsync instead of predict)
      const predictions = await model.executeAsync(resizedTensor);

      // Process predictions as needed
      // For simplicity, let's log the predictions to the console
      console.log(predictions);

      // Update state or perform other actions based on predictions
      setObjects(predictions);

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
          <Image source={{ uri: imageUrl }} style={styles.image} />
          <Button title="Classify Image" onPress={classifyImage} />
          {/* Display detected objects here if applicable */}
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
  },
  image: {
    width: '100%',
    height: '50%',
  },
  texto: {
    textAlign: 'center',
    fontSize: 25,
  },
});
