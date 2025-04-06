import { useState, useCallback } from 'react';
import { useDropzone } from 'react-dropzone';
import axios from 'axios';
import { validateImage, ALLOWED_IMAGE_TYPES } from '../utils/validation';

const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

interface RecognitionResult {
  success: boolean;
  text: string;
  confidence?: number;
  regions?: Array<{
    bbox: [number, number, number, number];
    text: string;
  }>;
}

export default function ImageUpload() {
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string>('');
  const [result, setResult] = useState<RecognitionResult | null>(null);
  const [preview, setPreview] = useState<string>('');

  const onDrop = useCallback(async (acceptedFiles: File[]) => {
    const file = acceptedFiles[0];
    if (!file) return;

    // Validate image before processing
    const validation = await validateImage(file);
    if (!validation.isValid) {
      setError(validation.error || 'Invalid image file');
      return;
    }

    setIsLoading(true);
    setError('');
    setResult(null);

    // Create preview URL
    const previewUrl = URL.createObjectURL(file);
    setPreview(previewUrl);

    const formData = new FormData();
    formData.append('file', file);

    try {
      const response = await axios.post<RecognitionResult>(
        `${API_URL}/api/recognize`,
        formData,
        {
          headers: {
            'Content-Type': 'multipart/form-data',
          },
        }
      );

      setResult(response.data);

      // Clean up preview URL when successful
      if (response.data.success) {
        URL.revokeObjectURL(previewUrl);
      }
    } catch (err: any) {
      setError(
        err.response?.data?.message || 
        err.message || 
        'Error processing image. Please try again.'
      );
      URL.revokeObjectURL(previewUrl);
    } finally {
      setIsLoading(false);
    }
  }, []);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      'image/*': ['.png', '.jpg', '.jpeg', '.tiff']
    },
    maxFiles: 1,
    multiple: false,
    maxSize: parseInt(process.env.NEXT_PUBLIC_MAX_FILE_SIZE || '10485760')
  });

  return (
    <div className="w-full max-w-3xl mx-auto">
      <div
        {...getRootProps()}
        className={`border-2 border-dashed rounded-lg p-8 text-center cursor-pointer transition-colors mb-8
          ${isDragActive ? 'border-blue-500 bg-blue-50' : 'border-gray-300 hover:border-gray-400'}`}
      >
        <input {...getInputProps()} />
        <div className="space-y-4">
          {isDragActive ? (
            <p className="text-blue-600">Drop the image here...</p>
          ) : (
            <>
              <p className="text-gray-600">
                Drag and drop an image here, or click to select a file
              </p>
              <p className="text-sm text-gray-500">
                Supported formats: JPEG, PNG, TIFF
              </p>
              <p className="text-sm text-gray-500">
                Maximum file size: {parseInt(process.env.NEXT_PUBLIC_MAX_FILE_SIZE || '10485760') / (1024 * 1024)}MB
              </p>
            </>
          )}
        </div>
      </div>

      {isLoading && (
        <div className="text-center p-4">
          <div className="inline-block animate-spin rounded-full h-8 w-8 border-4 border-blue-500 border-t-transparent"></div>
          <p className="mt-2 text-gray-600">Processing image...</p>
        </div>
      )}

      {error && (
        <div className="bg-red-50 text-red-700 p-4 rounded-lg mb-4">
          {error}
        </div>
      )}

      {preview && !isLoading && (
        <div className="mb-8">
          <h3 className="text-lg font-semibold mb-2">Uploaded Image:</h3>
          <img
            src={preview}
            alt="Uploaded preview"
            className="max-w-full h-auto rounded-lg shadow-md"
          />
        </div>
      )}

      {result && (
        <div className="bg-white shadow-md rounded-lg p-6">
          <h3 className="text-lg font-semibold mb-4">Recognized Text:</h3>
          <div className="bg-gray-50 p-4 rounded-lg whitespace-pre-wrap">
            {result.text}
          </div>
          {result.confidence && (
            <p className="mt-2 text-sm text-gray-600">
              Confidence: {(result.confidence * 100).toFixed(1)}%
            </p>
          )}
          {result.regions && result.regions.length > 0 && (
            <p className="mt-2 text-sm text-gray-600">
              Detected {result.regions.length} text regions
            </p>
          )}
        </div>
      )}
    </div>
  );
}