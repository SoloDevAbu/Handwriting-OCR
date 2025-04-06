'use client';

import Navbar from '../components/Navbar';
import ModelTraining from '../components/ModelTraining';

export default function TrainingPage() {
  return (
    <>
      <Navbar />
      <main className="min-h-screen p-8">
        <div className="max-w-4xl mx-auto">
          <h1 className="text-3xl font-bold mb-8 text-center">
            Model Training
          </h1>
          
          <p className="text-center mb-8 text-gray-600">
            Train custom models with your own datasets. Upload a dataset, monitor training jobs, and manage your trained models.
          </p>

          <ModelTraining />
        </div>
      </main>
    </>
  );
} 