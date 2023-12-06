import React, { useEffect, useState } from 'react';
import { StyleSheet, Text, View, Image, Button, ActivityIndicator, Alert } from 'react-native';
import * as tf from '@tensorflow/tfjs';
import { bundleResourceIO } from '@tensorflow/tfjs-react-native';
import * as jpeg from 'jpeg-js';

export default function App() { // assets\images_test\wrost_image.jpg
  const imageUrl =
    'https://raw.githack.com/JoaoVitorAguiar/object-detection-app/main/assets/images_test/wrost_image.jpg';
  const [model, setModel] = useState(null);
  const [objects, setObjects] = useState([]);
  const [loading, setLoading] = useState(true);

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
      // Limpar os recursos (por exemplo, liberar o modelo)
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
    try {
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
      return tf.tensor3d(buffer, [height, width, 3]);
    } catch (error) {
      console.error('Error decoding the image:', error);
      Alert.alert('Error', 'Unable to process the image.');
      return null;
    }
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

      // Carregar e pré-processar a imagem
      const response = await fetch(imageUrl);
      const rawImageData = await response.arrayBuffer();
      const tensor = imageToTensor(rawImageData);

      if (tensor) {
        // Redimensionar o tensor para corresponder à forma de entrada esperada [1, 640, 640, 3]
        const resizedTensor = tf.image
          .resizeBilinear(tensor, [640, 640])
          .reshape([1, 640, 640, 3])
          .div(255.0);

        // Fazer previsões usando o modelo (executeAsync em vez de predict)
        const predictions = await model.executeAsync(resizedTensor);

        // Processar as previsões
        const boxes = predictions[0].dataSync();
        const scores = predictions[1].dataSync();
        const classes = predictions[2].dataSync();

        // Encontrar o índice da detecção com o maior escore
        const maxScoreIndex = scores.indexOf(Math.max(...scores));

        // Verificar se o escore é maior que 0.5
        if (scores[maxScoreIndex] > 0.5) {
          // Obter detalhes da detecção com o maior escore
          const maxScoreObject = {
            ymin: boxes[maxScoreIndex * 4] * tensor.shape[0],
            xmin: boxes[maxScoreIndex * 4 + 1] * tensor.shape[1],
            ymax: boxes[maxScoreIndex * 4 + 2] * tensor.shape[0],
            xmax: boxes[maxScoreIndex * 4 + 3] * tensor.shape[1],
            class: classes[maxScoreIndex],
            score: scores[maxScoreIndex],
          };

          // Atualizar o estado com a detecção de maior escore
          setObjects([maxScoreObject]);

            setLoading(false);
          }
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

          {objects.map((object, index) => (
            <View key={index}>
              <Text>{`Class: ${object.class}, Score: ${object.score.toFixed(2)}`}</Text>
              <Text>{`Bounding Box: (${object.xmin.toFixed(2)}, ${object.ymin.toFixed(2)}, ${object.xmax.toFixed(2)}, ${object.ymax.toFixed(2)})`}</Text>
            </View>
          ))}
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
});
