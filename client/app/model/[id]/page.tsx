'use client';

import { useState, useEffect } from 'react';
import { useParams } from 'next/navigation';
import axios from 'axios';
import Link from 'next/link';
import Image from 'next/image';

const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

interface ModelInfo {
  id: string;
  path: string;
  created: string;
  class_mapping: Record<string, string>;
  num_classes: number;
  metrics: Record<string, any>;
  history: Record<string, any>;
  plots: string[];
}

interface Prediction {
  class: string;
  confidence: number;
  top_predictions: Array<{
    class: string;
    confidence: number;
  }>;
}

export default function ModelPage() {
  const params = useParams();
  const modelId = params.id as string;
  
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string>('');
  const [modelInfo, setModelInfo] = useState<ModelInfo | null>(null);
  const [file, setFile] = useState<File | null>(null);
  const [preview, setPreview] = useState<string>('');
  const [prediction, setPrediction] = useState<Prediction | null>(null);
  const [isPredicting, setIsPredicting] = useState(false);
  
  useEffect(() => {
    fetchModelInfo();
  }, [modelId]);
  
  const fetchModelInfo = async () => {
    setIsLoading(true);
    setError('');
    
    try {
      const response = await axios.get(`${API_URL}/api/custom-models/model-info/${modelId}`);
      setModelInfo(response.data);
    } catch (err: any) {
      setError(
        err.response?.data?.message || 
        err.message || 
        'Failed to load model information'
      );
    } finally {
      setIsLoading(false);
    }
  };
  
  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files.length > 0) {
      const selectedFile = e.target.files[0];
      setFile(selectedFile);
      
      // Create preview
      const reader = new FileReader();
      reader.onloadend = () => {
        setPreview(reader.result as string);
      };
      reader.readAsDataURL(selectedFile);
    }
  };
  
  const handlePrediction = async () => {
    if (!file) {
      setError('Please select an image first');
      return;
    }
    
    setIsPredicting(true);
    setError('');
    setPrediction(null);
    
    const formData = new FormData();
    formData.append('file', file);
    formData.append('model_id', modelId);
    
    try {
      const response = await axios.post(
        `${API_URL}/api/custom-models/predict`,
        formData,
        {
          headers: {
            'Content-Type': 'multipart/form-data',
          },
        }
      );
      
      if (response.data.success) {
        setPrediction(response.data.prediction);
      }
    } catch (err: any) {
      setError(
        err.response?.data?.message || 
        err.message || 
        'Failed to make prediction'
      );
    } finally {
      setIsPredicting(false);
    }
  };
  
  if (isLoading) {
    return (
      <div className="min-h-screen p-8">
        <div className="max-w-4xl mx-auto text-center">
          <div className="inline-block animate-spin rounded-full h-8 w-8 border-4 border-blue-500 border-t-transparent"></div>
          <p className="mt-2 text-gray-600">Loading model information...</p>
        </div>
      </div>
    );
  }
  
  if (error) {
    return (
      <div className="min-h-screen p-8">
        <div className="max-w-4xl mx-auto">
          <div className="bg-red-50 text-red-700 p-4 rounded-lg mb-4">
            {error}
          </div>
          <div className="text-center mt-4">
            <Link href="/training" className="text-blue-500 hover:text-blue-700">
              ← Back to Models
            </Link>
          </div>
        </div>
      </div>
    );
  }
  
  if (!modelInfo) {
    return (
      <div className="min-h-screen p-8">
        <div className="max-w-4xl mx-auto text-center">
          <p className="text-gray-600">Model not found</p>
          <div className="mt-4">
            <Link href="/training" className="text-blue-500 hover:text-blue-700">
              ← Back to Models
            </Link>
          </div>
        </div>
      </div>
    );
  }
  
  return (
    <div className="min-h-screen p-8">
      <div className="max-w-4xl mx-auto">
        <div className="mb-6">
          <Link href="/training" className="text-blue-500 hover:text-blue-700">
            ← Back to Models
          </Link>
        </div>
        
        <div className="bg-white shadow-md rounded-lg p-6 mb-6">
          <h1 className="text-2xl font-bold mb-2">Model: {modelInfo.id}</h1>
          <p className="text-gray-600 mb-4">
            This model can recognize {modelInfo.num_classes} different classes.
          </p>
          
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <div>
              <h2 className="text-lg font-semibold mb-2">Model Details</h2>
              <div className="bg-gray-50 p-4 rounded-lg">
                <p><span className="font-medium">Created:</span> {new Date(modelInfo.created).toLocaleString()}</p>
                <p><span className="font-medium">Classes:</span> {modelInfo.num_classes}</p>
                {modelInfo.metrics && modelInfo.metrics.evaluation && (
                  <div className="mt-2">
                    <p className="font-medium">Performance:</p>
                    <pre className="text-xs mt-1 overflow-auto max-h-32 p-2 bg-gray-100 rounded">
                      {modelInfo.metrics.evaluation}
                    </pre>
                  </div>
                )}
              </div>
            </div>
            
            <div>
              <h2 className="text-lg font-semibold mb-2">Available Classes</h2>
              <div className="bg-gray-50 p-4 rounded-lg">
                {modelInfo.class_mapping ? (
                  <div className="grid grid-cols-2 gap-2">
                    {Object.entries(modelInfo.class_mapping).map(([key, value]) => (
                      <div key={key} className="text-sm bg-white p-2 rounded border">
                        {value}
                      </div>
                    ))}
                  </div>
                ) : (
                  <p>No class information available</p>
                )}
              </div>
            </div>
          </div>
          
          {modelInfo.plots && modelInfo.plots.length > 0 && (
            <div className="mt-6">
              <h2 className="text-lg font-semibold mb-2">Training Results</h2>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                {modelInfo.plots.map((plot, index) => (
                  <div key={index} className="border rounded overflow-hidden">
                    <Image
                      src={`${API_URL}/${plot}`}
                      alt={`Plot ${index + 1}`}
                      width={400}
                      height={300}
                      className="w-full h-auto"
                    />
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
        
        <div className="bg-white shadow-md rounded-lg p-6">
          <h2 className="text-xl font-semibold mb-4">Test Your Model</h2>
          
          {error && (
            <div className="bg-red-50 text-red-700 p-4 rounded-lg mb-4">
              {error}
            </div>
          )}
          
          <div className="mb-4">
            <label className="block text-gray-700 text-sm font-bold mb-2">
              Upload an image to test
            </label>
            <input
              type="file"
              accept="image/*"
              onChange={handleFileChange}
              className="block w-full text-sm text-gray-500
                file:mr-4 file:py-2 file:px-4
                file:rounded-md file:border-0
                file:text-sm file:font-semibold
                file:bg-blue-50 file:text-blue-700
                hover:file:bg-blue-100"
            />
          </div>
          
          {preview && (
            <div className="mb-4">
              <p className="text-sm font-medium text-gray-700 mb-2">Preview:</p>
              <div className="w-full max-w-xs mx-auto">
                <img
                  src={preview}
                  alt="Preview"
                  className="max-h-64 max-w-full h-auto rounded-lg shadow-sm mx-auto"
                />
              </div>
            </div>
          )}
          
          <button
            onClick={handlePrediction}
            disabled={!file || isPredicting}
            className={`w-full bg-blue-500 hover:bg-blue-700 text-white font-bold py-2 px-4 rounded focus:outline-none focus:shadow-outline ${
              !file || isPredicting ? 'opacity-50 cursor-not-allowed' : ''
            }`}
          >
            {isPredicting ? 'Processing...' : 'Predict'}
          </button>
          
          {prediction && (
            <div className="mt-6 p-4 bg-gray-50 rounded-lg">
              <h3 className="text-lg font-semibold text-center mb-2">Prediction Result</h3>
              
              <div className="text-center mb-4">
                <div className="inline-block bg-blue-100 text-blue-800 text-xl font-bold px-4 py-2 rounded">
                  {prediction.class}
                </div>
                <p className="mt-1 text-sm text-gray-600">
                  Confidence: {(prediction.confidence * 100).toFixed(2)}%
                </p>
              </div>
              
              {prediction.top_predictions && prediction.top_predictions.length > 1 && (
                <div>
                  <p className="text-sm font-medium text-gray-700 mb-2">Top Predictions:</p>
                  <div className="grid grid-cols-1 gap-2">
                    {prediction.top_predictions.map((pred, index) => (
                      <div key={index} className="flex justify-between items-center p-2 bg-white rounded border">
                        <span>{pred.class}</span>
                        <span className="text-sm text-gray-600">
                          {(pred.confidence * 100).toFixed(2)}%
                        </span>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </div>
          )}
        </div>
      </div>
    </div>
  );
} 