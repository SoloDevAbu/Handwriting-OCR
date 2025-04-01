export default function About() {
  return (
    <div className="min-h-screen p-8">
      <div className="max-w-4xl mx-auto">
        <h1 className="text-3xl font-bold mb-8 text-center">
          About the OCR System
        </h1>
        
        <div className="prose lg:prose-xl mx-auto">
          <section className="mb-8">
            <h2 className="text-2xl font-semibold mb-4">Overview</h2>
            <p>
              This Handwriting OCR (Optical Character Recognition) system is designed
              to convert handwritten text from images into digital text. It utilizes
              advanced machine learning techniques and image processing to achieve
              accurate text recognition.
            </p>
          </section>

          <section className="mb-8">
            <h2 className="text-2xl font-semibold mb-4">Features</h2>
            <ul className="list-disc pl-6">
              <li>Support for multiple image formats (JPEG, PNG, TIFF)</li>
              <li>Advanced image preprocessing for better accuracy</li>
              <li>Real-time text recognition</li>
              <li>User-friendly interface</li>
              <li>Support for various handwriting styles</li>
            </ul>
          </section>

          <section className="mb-8">
            <h2 className="text-2xl font-semibold mb-4">How It Works</h2>
            <ol className="list-decimal pl-6">
              <li>Upload your image containing handwritten text</li>
              <li>The system preprocesses the image to enhance quality</li>
              <li>Advanced ML models detect and recognize text</li>
              <li>Results are displayed in an easy-to-read format</li>
            </ol>
          </section>

          <section>
            <h2 className="text-2xl font-semibold mb-4">Technology Stack</h2>
            <ul className="list-disc pl-6">
              <li>Frontend: Next.js with TypeScript</li>
              <li>Backend: FastAPI (Python)</li>
              <li>ML Framework: TensorFlow & PyTesseract</li>
              <li>Image Processing: OpenCV</li>
            </ul>
          </section>
        </div>
      </div>
    </div>
  );
}