import cv2
import numpy as np
import os
from scipy.signal import convolve2d

class PurePython:
    def __init__(self, input_dir="./input", output_dir="./output"):
        self.locationpath=os.path.dirname(__file__)
        self.input_dir = os.path.join(self.locationpath, "input")
        self.output_dir = os.path.join(self.locationpath, "output")
        self.images = {}
        os.makedirs(self.output_dir, exist_ok=True)
    
    def load_images(self):
        for filename in os.listdir(self.input_dir):
            if filename.endswith(".jpg") or filename.endswith(".png"):
                image_path = os.path.join(self.input_dir, filename)
                image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE).tolist()
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
        if not self.results:
            print("No results available. Apply filters first.")
            return
        for filename, filters in self.results.items():
            print(f"Results for {filename}:")
            for filter_name, result in filters.items():
                print(f"  {filter_name} filter applied.")
    
    def save_images(self):
        if not self.results:
            print("No results to save. Apply filters first.")
            return
        for filename, filters in self.results.items():
            for filter_name, result in filters.items():
                result = np.clip(result, 0, 255)  # Ensure values are within valid range
                result_path = os.path.join(self.output_dir, f"{filter_name}_{filename}")
                cv2.imwrite(result_path, np.array(result, dtype=np.uint8))
        print("Filtered images saved successfully.")
    
    def gaussian_filter(self, image):
        kernel = [[1, 2, 1], [2, 4, 2], [1, 2, 1]]
        kernel = [[elem / 16 for elem in row] for row in kernel]
        return self.convolve(image, kernel)
    
    def sobel_filter(self, image):
        Sx = [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]
        Sy = [[-1, -2, -1], [0, 0, 0], [1, 2, 1]]
        Gx = self.convolve(image, Sx)
        Gy = self.convolve(image, Sy)
        return [[(Gx[i][j]**2 + Gy[i][j]**2)**0.5 for j in range(len(Gx[0]))] for i in range(len(Gx))]
    
    def median_filter(self, image, kernel_size=3):
        offset = kernel_size // 2
        padded_image = self.pad_image(image, offset)
        new_image = [[0] * len(image[0]) for _ in range(len(image))]
        
        for i in range(len(image)):
            for j in range(len(image[0])):
                neighbors = [padded_image[x][y] for x in range(i, i+kernel_size) for y in range(j, j+kernel_size)]
                new_image[i][j] = sorted(neighbors)[len(neighbors) // 2]
        
        return new_image
    
    def convolve(self, image, kernel):
        kernel_size = len(kernel)
        offset = kernel_size // 2
        padded_image = self.pad_image(image, offset)
        new_image = [[0] * len(image[0]) for _ in range(len(image))]
        
        for i in range(len(image)):
            for j in range(len(image[0])):
                new_image[i][j] = sum(padded_image[i+x][j+y] * kernel[x][y] for x in range(kernel_size) for y in range(kernel_size))
        
        return new_image
    
    def pad_image(self, image, padding):
        padded = [[0] * (len(image[0]) + 2 * padding) for _ in range(len(image) + 2 * padding)]
        for i in range(len(image)):
            for j in range(len(image[0])):
                padded[i + padding][j + padding] = image[i][j]
        return padded

if __name__ == "__main__":
    processor = PurePython()
    processor.load_images()
    processor.apply_filters()
    processor.check_results()
    processor.save_images()
