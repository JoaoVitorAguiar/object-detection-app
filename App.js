import React, { useEffect, useState } from 'react';
import { StyleSheet, Text, View, Image, Button, ActivityIndicator, Alert } from 'react-native';
import * as tf from '@tensorflow/tfjs';
import { bundleResourceIO } from '@tensorflow/tfjs-react-native';

export default function App() {
  const imageUri  = require('./assets/images_test/image.jpg');
  const img = './assets/images_test/image.jpg';
  const [model, setModel] = useState(null);
  const [objects, setObjects] = useState([]);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    loadModel();
  }, []);

  const loadModel = async () => {
    try {
      await tf.ready();
      const modelJson = require('./assets/models/model.json');
      const modelWeights = require('./assets/models/group1-shard.bin');
      const loadedModel = await tf.loadGraphModel(bundleResourceIO(modelJson, modelWeights));
      setModel(loadedModel);
    } catch (error) {
      console.error('Error loading TensorFlow model:', error);
      Alert.alert('Erro', 'Não foi possível carregar o modelo TensorFlow.');
    }
  };

  const detectObjects = async () => {
    try {
      setLoading(true);

      // Load the image directly as a local resource
      const fileUri = img // Convert to string

      if (!model) {
        throw new Error('O modelo não foi carregado corretamente.');
      }

      const imgBuffer = tf.util.encodeString(fileUri, 'base64').buffer;
      const raw = new Uint8Array(imgBuffer);
      console.log('imgBuffer:', raw);
      // Convert the image to a tensor
      const imageTensor = decodeJpeg(raw);

      // Execute the model on the image tensor
      const predictions = await model.executeAsync(imageTensor);

      // Update the state with the detected objects
      setObjects(predictions);
    } catch (error) {
      console.error('Error detecting objects:', error);
      Alert.alert('Erro', 'Não foi possível detectar objetos.');
    } finally {
      setLoading(false);
    }
  };

  const convertImageToTensor = (raw) => {
    const img = tf.browser.fromPixels(raw);
    const resizedImg = tf.image.resizeBilinear(img, [224, 224]);
    const expandedImg = resizedImg.expandDims();
    const normalizedImg = expandedImg.div(255.0);
    return normalizedImg;
  };

  const decodeJpeg = (raw) => {
    
    const imageTensor = tf.node.decodeJpeg(raw);
    return convertImageToTensor(imageTensor);
  };

  return (
    <View style={styles.container}>
      <Button title='Detectar Objetos' onPress={detectObjects} />
      <Image source={imageUri} style={styles.image} />
      {loading && <ActivityIndicator size="large" color="#0000ff" />}
      <Text style={styles.texto}>
        {objects.length > 0 ? `Objetos detectados: ${objects.length}` : ''}
      </Text>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    justifyContent: 'center',
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
