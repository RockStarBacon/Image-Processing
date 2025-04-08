import cv2
import numpy as np
import os

class Numpy:
    def __init__(self, input_dir="./input", output_dir="./output"):
        self.locationpath = os.path.dirname(__file__)
        self.input_dir = os.path.join(self.locationpath, "input")
        self.output_dir = os.path.join(self.locationpath, "output")
        self.images = {}
        os.makedirs(self.output_dir, exist_ok=True)
    
    def load_images(self):
        for filename in os.listdir(self.input_dir):
            if filename.endswith((".jpg", ".png")):
                image_path = os.path.join(self.input_dir, filename)
                image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
                self.images[filename] = image
        print("Images loaded successfully.")
    
    def apply_filters(self):
        self.results = {}
        for filename, image in self.images.items():
            self.results[filename] = {
                "gaussian": self.gaussian_filter(image),
                "sobel": self.sobel_filter(image),
                "median": self.median_filter(image)
            }
        print("Filters applied successfully.")
    
    def check_results(self):
        if not hasattr(self, 'results') or not self.results:
            print("No results available. Apply filters first.")
            return
        for filename, filters in self.results.items():
            print(f"Results for {filename}:")
            for filter_name in filters:
                print(f"  {filter_name} filter applied.")
    
    def save_images(self):
        if not hasattr(self, 'results') or not self.results:
            print("No results to save. Apply filters first.")
            return
        for filename, filters in self.results.items():
            for filter_name, result in filters.items():
                result = np.clip(result, 0, 255).astype(np.uint8)
                result_path = os.path.join(self.output_dir, f"{filter_name}_{filename}")
                cv2.imwrite(result_path, result)
        print("Filtered images saved successfully.")
    
    def gaussian_filter(self, image):
        kernel = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]], dtype=np.float32) / 16
        return cv2.filter2D(image, -1, kernel)
    
    def sobel_filter(self, image):
        Gx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
        Gy = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
        return np.sqrt(Gx**2 + Gy**2)
    
    def median_filter(self, image, kernel_size=3):
        return cv2.medianBlur(image, kernel_size)

if __name__ == "__main__":
    processor = Numpy()
    processor.load_images()
    processor.apply_filters()
    processor.check_results()
    processor.save_images()
