export const ALLOWED_IMAGE_TYPES = ['image/jpeg', 'image/png', 'image/tiff'];
const MAX_IMAGE_DIMENSION = 4096; // Maximum width or height in pixels

export async function validateImage(file: File): Promise<{ isValid: boolean; error?: string }> {
  // Check file type
  if (!ALLOWED_IMAGE_TYPES.includes(file.type)) {
    return {
      isValid: false,
      error: `Invalid file type. Allowed types: ${ALLOWED_IMAGE_TYPES.join(', ')}`
    };
  }

  // Check file size
  const maxSize = parseInt(process.env.NEXT_PUBLIC_MAX_FILE_SIZE || '10485760');
  if (file.size > maxSize) {
    return {
      isValid: false,
      error: `File size exceeds ${maxSize / (1024 * 1024)}MB limit`
    };
  }

  // Check image dimensions
  try {
    const dimensions = await getImageDimensions(file);
    if (dimensions.width > MAX_IMAGE_DIMENSION || dimensions.height > MAX_IMAGE_DIMENSION) {
      return {
        isValid: false,
        error: `Image dimensions exceed ${MAX_IMAGE_DIMENSION}x${MAX_IMAGE_DIMENSION} limit`
      };
    }
  } catch (err) {
    return {
      isValid: false,
      error: 'Invalid image file'
    };
  }

  return { isValid: true };
}

function getImageDimensions(file: File): Promise<{ width: number; height: number }> {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.onload = () => {
      URL.revokeObjectURL(img.src); // Clean up
      resolve({ width: img.width, height: img.height });
    };
    img.onerror = () => {
      URL.revokeObjectURL(img.src); // Clean up
      reject(new Error('Failed to load image'));
    };
    img.src = URL.createObjectURL(file);
  });
}