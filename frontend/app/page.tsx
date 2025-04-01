'use client';

import Navbar from './components/Navbar';
import ImageUpload from './components/ImageUpload';

export default function Home() {
  return (
    <>
      <Navbar />
      <main className="min-h-screen p-8">
        <div className="max-w-4xl mx-auto">
          <h1 className="text-3xl font-bold mb-8 text-center">
            Handwriting Recognition System
          </h1>
          
          <p className="text-center mb-8 text-gray-600">
            Upload an image of handwritten text to convert it into digital text.
            Supports various handwriting styles and multiple image formats.
          </p>

          <ImageUpload />
        </div>
      </main>
    </>
  );
}