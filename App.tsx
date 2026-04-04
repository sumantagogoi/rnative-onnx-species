import React, { useState, useEffect } from 'react';
import { StyleSheet, Text, View, TouchableOpacity, Image, ActivityIndicator, SafeAreaView, StatusBar, ScrollView } from 'react-native';
import * as ImagePicker from 'expo-image-picker';
import * as FileSystem from 'expo-file-system';
import * as ImageManipulator from 'expo-image-manipulator';
import { InferenceSession, Tensor } from 'onnxruntime-react-native';
import { Asset } from 'expo-asset';
import { Buffer } from 'buffer';
import * as jpeg from 'jpeg-js';

// Load model asset
const modelAsset = require('./assets/speciesnet_quantized.onnx');
const labelsAsset = require('./assets/labels.txt');

export default function App() {
  const [imageUri, setImageUri] = useState<string | null>(null);
  const [session, setSession] = useState<InferenceSession | null>(null);
  const [labels, setLabels] = useState<string[]>([]);
  const [prediction, setPrediction] = useState<string | null>(null);
  const [confidence, setConfidence] = useState<number | null>(null);
  const [isProcessing, setIsProcessing] = useState<boolean>(false);
  const [modelReady, setModelReady] = useState<boolean>(false);

  useEffect(() => {
    initModel();
  }, []);

  const initModel = async () => {
    try {
      // Load Labels
      const labelsAssetInfo = await Asset.loadAsync(labelsAsset);
      const labelsPath = labelsAssetInfo[0].localUri || labelsAssetInfo[0].uri;
      const labelsText = await FileSystem.readAsStringAsync(labelsPath);
      const parsedLabels = labelsText.split('\n').map(l => l.trim()).filter(l => l.length > 0);
      setLabels(parsedLabels);

      // Load ONNX Session
      const modelAssetInfo = await Asset.loadAsync(modelAsset);
      const uri = modelAssetInfo[0].localUri || modelAssetInfo[0].uri;
      
      // @ts-ignore - property exists at runtime
      const modelPath = FileSystem.documentDirectory + 'speciesnet_quantized.onnx';
      if (uri.startsWith('http')) {
        await FileSystem.downloadAsync(uri, modelPath);
      } else {
        await FileSystem.copyAsync({ from: uri, to: modelPath });
      }

      const newSession = await InferenceSession.create(modelPath, {
        executionProviders: ['cpu'], 
      });
      setSession(newSession);
      setModelReady(true);
    } catch (e) {
      console.error("Failed to init model", e);
    }
  };

  const pickImage = async () => {
    let result = await ImagePicker.launchImageLibraryAsync({
      mediaTypes: ImagePicker.MediaTypeOptions.Images,
      allowsEditing: true,
      aspect: [1, 1],
      quality: 1,
    });

    if (!result.canceled && result.assets && result.assets.length > 0) {
      setImageUri(result.assets[0].uri);
      setPrediction(null);
      setConfidence(null);
    }
  };
  
  const takePhoto = async () => {
    let result = await ImagePicker.launchCameraAsync({
      allowsEditing: true,
      aspect: [1, 1],
      quality: 1,
    });

    if (!result.canceled && result.assets && result.assets.length > 0) {
      setImageUri(result.assets[0].uri);
      setPrediction(null);
      setConfidence(null);
    }
  }

  const runModel = async () => {
    if (!imageUri || !session) return;
    setIsProcessing(true);

    try {
      // 1. Resize/Format Image
      const manipResult = await ImageManipulator.manipulateAsync(
        imageUri,
        [{ resize: { width: 480, height: 480 } }],
        { format: ImageManipulator.SaveFormat.JPEG, base64: true }
      );

      if (!manipResult.base64) throw new Error("Image to base64 conversion failed");

      // 2. Decode Image to Pixels
      const rawImageData = Buffer.from(manipResult.base64, 'base64');
      const decodedImage = jpeg.decode(rawImageData, { useTArray: true });
      const { data } = decodedImage;
      
      // 3. Prepare Float32Array Tensor Data (NHWC [1, 480, 480, 3])
      const imageBufferData = new Float32Array(1 * 480 * 480 * 3);
      for (let i = 0; i < 480 * 480; i++) {
        const p = i * 4;
        imageBufferData[i * 3 + 0] = data[p + 0] / 255.0;     // R
        imageBufferData[i * 3 + 1] = data[p + 1] / 255.0;     // G
        imageBufferData[i * 3 + 2] = data[p + 2] / 255.0;     // B
      }

      // 4. Create Tensor
      const inputName = session.inputNames[0];
      const tensor = new Tensor('float32', imageBufferData, [1, 480, 480, 3]);

      // 5. Run Inference
      const feeds: Record<string, Tensor> = {};
      feeds[inputName] = tensor;

      const outputData = await session.run(feeds);
      const outputName = session.outputNames[0];
      const outputTensor = outputData[outputName];
      const predictions = outputTensor.data as Float32Array;

      // 6. Argmax
      let maxIdx = 0;
      let maxVal = -Infinity;
      for (let i = 0; i < predictions.length; i++) {
        if (predictions[i] > maxVal) {
          maxVal = predictions[i];
          maxIdx = i;
        }
      }

      setPrediction(labels[maxIdx] || `Class ${maxIdx}`);
      setConfidence(maxVal);

    } catch (e) {
      console.error(e);
      setPrediction("Error during inference");
    } finally {
      setIsProcessing(false);
    }
  };

  return (
    <SafeAreaView style={styles.container}>
      <StatusBar barStyle="light-content" />
      <ScrollView contentContainerStyle={styles.scrollContent}>
        <View style={styles.header}>
          <Text style={styles.title}>Species Recognizer</Text>
          <Text style={styles.subtitle}>Discover wildlife with AI</Text>
        </View>

        <View style={styles.imageContainer}>
          {imageUri ? (
            <Image source={{ uri: imageUri }} style={styles.image} />
          ) : (
            <View style={styles.placeholder}>
              <Text style={styles.placeholderText}>No Image Selected</Text>
            </View>
          )}
        </View>

        <View style={styles.controls}>
          <TouchableOpacity style={[styles.button, styles.secondaryBtn]} onPress={pickImage}>
            <Text style={styles.buttonText}>Gallery</Text>
          </TouchableOpacity>
          <TouchableOpacity style={[styles.button, styles.secondaryBtn]} onPress={takePhoto}>
            <Text style={styles.buttonText}>Camera</Text>
          </TouchableOpacity>
        </View>

        {imageUri && (
          <TouchableOpacity 
            style={[styles.button, styles.primaryBtn, (!modelReady || isProcessing) && styles.disabledBtn]} 
            onPress={runModel}
            disabled={!modelReady || isProcessing}
          >
            {isProcessing ? (
              <ActivityIndicator color="#fff" />
            ) : (
              <Text style={styles.primaryBtnText}>Recognize Species</Text>
            )}
          </TouchableOpacity>
        )}

        {prediction && (
          <View style={styles.resultContainer}>
            <Text style={styles.resultLabel}>Classification:</Text>
            <Text style={styles.resultText}>{prediction}</Text>
            {confidence !== null && (
              <Text style={styles.confidenceText}>Confidence: {(confidence * 100).toFixed(1)}%</Text>
            )}
          </View>
        )}
      </ScrollView>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#121212',
  },
  scrollContent: {
    padding: 24,
    alignItems: 'center',
  },
  header: {
    marginTop: 40,
    marginBottom: 40,
    alignItems: 'center',
  },
  title: {
    fontSize: 28,
    fontWeight: '800',
    color: '#ffffff',
    letterSpacing: 0.5,
  },
  subtitle: {
    fontSize: 16,
    color: '#a0a0a0',
    marginTop: 8,
  },
  imageContainer: {
    width: 300,
    height: 300,
    borderRadius: 20,
    overflow: 'hidden',
    backgroundColor: '#1e1e1e',
    elevation: 8,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 4 },
    shadowOpacity: 0.3,
    shadowRadius: 8,
    marginBottom: 30,
    borderWidth: 1,
    borderColor: '#333',
  },
  image: {
    width: '100%',
    height: '100%',
    resizeMode: 'cover',
  },
  placeholder: {
    flex: 1,
    alignItems: 'center',
    justifyContent: 'center',
  },
  placeholderText: {
    color: '#666',
    fontSize: 16,
  },
  controls: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    width: '100%',
    marginBottom: 20,
    gap: 12,
  },
  button: {
    borderRadius: 12,
    paddingVertical: 16,
    paddingHorizontal: 20,
    alignItems: 'center',
    justifyContent: 'center',
  },
  secondaryBtn: {
    backgroundColor: '#2c2c2c',
    flex: 1,
  },
  primaryBtn: {
    backgroundColor: '#007AFF',
    width: '100%',
    marginTop: 10,
    shadowColor: '#007AFF',
    shadowOffset: { width: 0, height: 4 },
    shadowOpacity: 0.4,
    shadowRadius: 8,
  },
  disabledBtn: {
    backgroundColor: '#2a4d7c',
    shadowOpacity: 0,
  },
  buttonText: {
    color: '#e0e0e0',
    fontSize: 16,
    fontWeight: '600',
  },
  primaryBtnText: {
    color: '#ffffff',
    fontSize: 18,
    fontWeight: '700',
  },
  resultContainer: {
    marginTop: 40,
    padding: 24,
    backgroundColor: '#1e1e1e',
    borderRadius: 16,
    width: '100%',
    alignItems: 'center',
    borderWidth: 1,
    borderColor: '#333',
  },
  resultLabel: {
    fontSize: 14,
    color: '#888',
    textTransform: 'uppercase',
    letterSpacing: 1,
    marginBottom: 8,
  },
  resultText: {
    fontSize: 24,
    fontWeight: 'bold',
    color: '#4caf50',
    textAlign: 'center',
    marginBottom: 8,
  },
  confidenceText: {
    fontSize: 16,
    color: '#a0a0a0',
  }
});
