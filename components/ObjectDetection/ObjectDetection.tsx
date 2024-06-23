import React, { useRef, useEffect, useState } from 'react';
import { View, Dimensions } from 'react-native';
import Canvas, { CanvasRenderingContext2D } from 'react-native-canvas';
import * as tf from '@tensorflow/tfjs';
import '@tensorflow/tfjs-react-native';

const { width, height } = Dimensions.get('window');

const loadModel = async (): Promise<tf.GraphModel> => {
  await tf.ready();
  const model = await tf.loadGraphModel('path_to_model/model.json');
  return model;
};

interface Prediction {
  bbox: [number, number, number, number];
  class: string;
  score: number;
}

const ObjectDetection: React.FC = () => {
  const canvasRef = useRef<Canvas>(null);
  const [model, setModel] = useState<tf.GraphModel | null>(null);

  useEffect(() => {
    const initializeModel = async () => {
      const loadedModel = await loadModel();
      setModel(loadedModel);
    };
    initializeModel();
  }, []);

  const handleCanvas = async (canvas: Canvas | null) => {
    if (canvas) {
      const ctx = canvas.getContext('2d') as CanvasRenderingContext2D;
      const image:typeof Image = new Image();
      image.src = 'path_to_image';
      image.onload = () => {
        canvas.width = width;
        canvas.height = height;
        ctx.drawImage(image, 0, 0, width, height);
        if (model) {
          detectObjects(model, image, ctx);
        }
      };
    }
  };

  const detectObjects = async (model: tf.GraphModel, image: HTMLImageElement, ctx: CanvasRenderingContext2D) => {
    const tensor = tf.browser.fromPixels(image);
    const predictions: any = await model.executeAsync(tensor);
    drawBoundingBoxes(predictions, ctx);
    tf.dispose(tensor);
  };

  const drawBoundingBoxes = (predictions: any, ctx: CanvasRenderingContext2D) => {
    predictions.forEach((prediction: Prediction) => {
      const [x, y, width, height] = prediction.bbox;
      ctx.strokeStyle = 'red';
      ctx.lineWidth = 2;
      ctx.strokeRect(x, y, width, height);
    });
  };

  return <Canvas ref={canvasRef} onCanvasReady={handleCanvas} />;
};

export default ObjectDetection;
