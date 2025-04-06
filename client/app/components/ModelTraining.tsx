import { useState, useRef, useEffect } from 'react';
import axios from 'axios';
import { useRouter } from 'next/navigation';

const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

interface TrainingJob {
  job_id: string;
  status: string;
  progress: number;
  start_time: string;
  end_time: string | null;
  metrics: any;
  error: string | null;
}

interface TrainedModel {
  id: string;
  path: string;
  created: string;
  metrics: Record<string, number>;
  classes: number;
}

export default function ModelTraining() {
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string>('');
  const [success, setSuccess] = useState<string>('');
  const [file, setFile] = useState<File | null>(null);
  const [imageSize, setImageSize] = useState<string>('28,28');
  const [batchSize, setBatchSize] = useState<number>(32);
  const [epochs, setEpochs] = useState<number>(15);
  const [activeTab, setActiveTab] = useState<'upload' | 'jobs' | 'models'>('upload');
  const [trainingJobs, setTrainingJobs] = useState<TrainingJob[]>([]);
  const [trainedModels, setTrainedModels] = useState<TrainedModel[]>([]);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const router = useRouter();
  
  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files.length > 0) {
      setFile(e.target.files[0]);
    }
  };
  
  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    
    if (!file) {
      setError('Please select a dataset file');
      return;
    }
    
    setIsLoading(true);
    setError('');
    setSuccess('');
    
    const formData = new FormData();
    formData.append('dataset', file);
    formData.append('image_size', imageSize);
    formData.append('batch_size', batchSize.toString());
    formData.append('epochs', epochs.toString());
    
    try {
      const response = await axios.post(`${API_URL}/api/training/start`, formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });
      
      if (response.data.job_id) {
        setSuccess(`Training job started successfully! Job ID: ${response.data.job_id}`);
        setFile(null);
        if (fileInputRef.current) {
          fileInputRef.current.value = '';
        }
        fetchTrainingJobs();
        setActiveTab('jobs');
      }
    } catch (err: any) {
      setError(
        err.response?.data?.message || 
        err.message || 
        'Failed to start training job. Please try again.'
      );
    } finally {
      setIsLoading(false);
    }
  };
  
  const fetchTrainingJobs = async () => {
    try {
      const response = await axios.get(`${API_URL}/api/training/jobs`);
      setTrainingJobs(response.data.jobs || []);
    } catch (err) {
      console.error('Error fetching training jobs:', err);
    }
  };
  
  const fetchTrainedModels = async () => {
    try {
      const response = await axios.get(`${API_URL}/api/training/models`);
      setTrainedModels(response.data.models || []);
    } catch (err) {
      console.error('Error fetching trained models:', err);
    }
  };
  
  useEffect(() => {
    fetchTrainingJobs();
    fetchTrainedModels();
    
    // Poll for updates on active jobs
    const intervalId = setInterval(() => {
      if (activeTab === 'jobs') {
        fetchTrainingJobs();
      }
    }, 5000); // Poll every 5 seconds
    
    return () => clearInterval(intervalId);
  }, [activeTab]);
  
  const formatDate = (dateString: string) => {
    if (!dateString) return 'N/A';
    return new Date(dateString).toLocaleString();
  };
  
  const getStatusColor = (status: string) => {
    switch (status) {
      case 'completed':
        return 'text-green-600';
      case 'failed':
        return 'text-red-600';
      case 'training':
        return 'text-blue-600';
      case 'preparing_data':
        return 'text-yellow-600';
      default:
        return 'text-gray-600';
    }
  };
  
  const handleUseModel = (modelId: string) => {
    router.push(`/model/${modelId}`);
  };
  
  return (
    <div className="w-full max-w-4xl mx-auto">
      <div className="mb-6">
        <div className="border-b border-gray-200">
          <nav className="-mb-px flex">
            <button
              onClick={() => setActiveTab('upload')}
              className={`py-4 px-6 text-center border-b-2 font-medium text-sm ${
                activeTab === 'upload'
                  ? 'border-blue-500 text-blue-600'
                  : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
              }`}
            >
              Upload Dataset
            </button>
            <button
              onClick={() => setActiveTab('jobs')}
              className={`py-4 px-6 text-center border-b-2 font-medium text-sm ${
                activeTab === 'jobs'
                  ? 'border-blue-500 text-blue-600'
                  : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
              }`}
            >
              Training Jobs
            </button>
            <button
              onClick={() => setActiveTab('models')}
              className={`py-4 px-6 text-center border-b-2 font-medium text-sm ${
                activeTab === 'models'
                  ? 'border-blue-500 text-blue-600'
                  : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
              }`}
            >
              Trained Models
            </button>
          </nav>
        </div>
      </div>
      
      {activeTab === 'upload' && (
        <div className="bg-white shadow-md rounded-lg p-6">
          <h2 className="text-2xl font-semibold mb-4">Train a New Model</h2>
          
          <p className="mb-6 text-gray-600">
            Upload a ZIP file containing your dataset organized in folders, where each folder represents a class.
            Each folder should contain multiple images of that class.
          </p>
          
          {error && (
            <div className="bg-red-50 text-red-700 p-4 rounded-lg mb-4">
              {error}
            </div>
          )}
          
          {success && (
            <div className="bg-green-50 text-green-700 p-4 rounded-lg mb-4">
              {success}
            </div>
          )}
          
          <form onSubmit={handleSubmit}>
            <div className="mb-4">
              <label className="block text-gray-700 text-sm font-bold mb-2">
                Dataset (ZIP file)
              </label>
              <input
                type="file"
                ref={fileInputRef}
                accept=".zip"
                onChange={handleFileChange}
                className="block w-full text-sm text-gray-500
                  file:mr-4 file:py-2 file:px-4
                  file:rounded-md file:border-0
                  file:text-sm file:font-semibold
                  file:bg-blue-50 file:text-blue-700
                  hover:file:bg-blue-100"
              />
              <p className="mt-1 text-sm text-gray-500">
                Each folder name will be used as a class label.
              </p>
            </div>
            
            <div className="mb-4">
              <label className="block text-gray-700 text-sm font-bold mb-2">
                Image Size (width,height)
              </label>
              <input
                type="text"
                value={imageSize}
                onChange={(e) => setImageSize(e.target.value)}
                className="shadow appearance-none border rounded w-full py-2 px-3 text-gray-700 leading-tight focus:outline-none focus:shadow-outline"
                placeholder="28,28"
              />
              <p className="mt-1 text-sm text-gray-500">
                The size to resize images to (e.g., 28,28).
              </p>
            </div>
            
            <div className="mb-4">
              <label className="block text-gray-700 text-sm font-bold mb-2">
                Batch Size
              </label>
              <input
                type="number"
                value={batchSize}
                onChange={(e) => setBatchSize(parseInt(e.target.value) || 32)}
                className="shadow appearance-none border rounded w-full py-2 px-3 text-gray-700 leading-tight focus:outline-none focus:shadow-outline"
                min="1"
              />
            </div>
            
            <div className="mb-6">
              <label className="block text-gray-700 text-sm font-bold mb-2">
                Epochs
              </label>
              <input
                type="number"
                value={epochs}
                onChange={(e) => setEpochs(parseInt(e.target.value) || 10)}
                className="shadow appearance-none border rounded w-full py-2 px-3 text-gray-700 leading-tight focus:outline-none focus:shadow-outline"
                min="1"
              />
              <p className="mt-1 text-sm text-gray-500">
                Number of times to train on the entire dataset.
              </p>
            </div>
            
            <button
              type="submit"
              disabled={isLoading}
              className={`w-full bg-blue-500 hover:bg-blue-700 text-white font-bold py-2 px-4 rounded focus:outline-none focus:shadow-outline ${
                isLoading ? 'opacity-50 cursor-not-allowed' : ''
              }`}
            >
              {isLoading ? 'Starting Training...' : 'Start Training'}
            </button>
          </form>
        </div>
      )}
      
      {activeTab === 'jobs' && (
        <div className="bg-white shadow-md rounded-lg p-6">
          <div className="flex justify-between items-center mb-4">
            <h2 className="text-2xl font-semibold">Training Jobs</h2>
            <button
              onClick={fetchTrainingJobs}
              className="text-blue-500 hover:text-blue-700"
            >
              Refresh
            </button>
          </div>
          
          {trainingJobs.length === 0 ? (
            <p className="text-gray-500">No training jobs found.</p>
          ) : (
            <div className="overflow-x-auto">
              <table className="min-w-full bg-white">
                <thead>
                  <tr>
                    <th className="py-2 px-4 border-b border-gray-200 bg-gray-50 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                      Job ID
                    </th>
                    <th className="py-2 px-4 border-b border-gray-200 bg-gray-50 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                      Status
                    </th>
                    <th className="py-2 px-4 border-b border-gray-200 bg-gray-50 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                      Started
                    </th>
                    <th className="py-2 px-4 border-b border-gray-200 bg-gray-50 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                      Finished
                    </th>
                    <th className="py-2 px-4 border-b border-gray-200 bg-gray-50 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                      Progress
                    </th>
                  </tr>
                </thead>
                <tbody>
                  {trainingJobs.map((job) => (
                    <tr key={job.job_id}>
                      <td className="py-2 px-4 border-b border-gray-200 text-sm">
                        {job.job_id}
                      </td>
                      <td className="py-2 px-4 border-b border-gray-200 text-sm">
                        <span className={getStatusColor(job.status)}>
                          {job.status.replace('_', ' ')}
                        </span>
                        {job.error && (
                          <span className="text-red-500 text-xs block">
                            {job.error}
                          </span>
                        )}
                      </td>
                      <td className="py-2 px-4 border-b border-gray-200 text-sm">
                        {formatDate(job.start_time)}
                      </td>
                      <td className="py-2 px-4 border-b border-gray-200 text-sm">
                        {job.end_time ? formatDate(job.end_time) : '-'}
                      </td>
                      <td className="py-2 px-4 border-b border-gray-200 text-sm">
                        <div className="w-full bg-gray-200 rounded-full h-2.5">
                          <div
                            className="bg-blue-600 h-2.5 rounded-full"
                            style={{ width: `${job.progress}%` }}
                          ></div>
                        </div>
                        <span className="text-xs text-gray-500">
                          {job.progress}%
                        </span>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </div>
      )}
      
      {activeTab === 'models' && (
        <div className="bg-white shadow-md rounded-lg p-6">
          <div className="flex justify-between items-center mb-4">
            <h2 className="text-2xl font-semibold">Trained Models</h2>
            <button
              onClick={fetchTrainedModels}
              className="text-blue-500 hover:text-blue-700"
            >
              Refresh
            </button>
          </div>
          
          {trainedModels.length === 0 ? (
            <p className="text-gray-500">No trained models found.</p>
          ) : (
            <div className="grid gap-6 grid-cols-1 md:grid-cols-2">
              {trainedModels.map((model) => (
                <div
                  key={model.id}
                  className="border rounded-lg p-4 hover:shadow-md transition-shadow"
                >
                  <h3 className="font-semibold text-lg">{model.id}</h3>
                  <p className="text-sm text-gray-500">
                    Created: {formatDate(model.created)}
                  </p>
                  <p className="text-sm text-gray-500">
                    Classes: {model.classes}
                  </p>
                  <div className="mt-2">
                    {model.metrics && model.metrics['Test accuracy'] && (
                      <span className="inline-flex items-center px-2.5 py-0.5 rounded-md text-sm font-medium bg-green-100 text-green-800">
                        Accuracy: {(model.metrics['Test accuracy'] * 100).toFixed(2)}%
                      </span>
                    )}
                  </div>
                  <div className="mt-4 flex justify-end">
                    <button
                      onClick={() => handleUseModel(model.id)}
                      className="bg-blue-500 hover:bg-blue-700 text-white text-sm font-bold py-1 px-3 rounded"
                    >
                      Use This Model
                    </button>
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>
      )}
    </div>
  );
} 