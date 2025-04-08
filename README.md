# 🖼️ Image Filters: Python vs NumPy vs Cython

This project compares the performance of three image processing filters using different implementation strategies: Pure Python, NumPy, and Cython.

## 📌 Filters Implemented
- **Gaussian Filter** (for smoothing)
- **Sobel Filter** (for edge detection)
- **Median Filter** (for noise reduction)

## 📁 Repository Contents
- `gaussian_filter.py` – Gaussian filter using Python
- `sobel_filter.py` – Sobel edge detection using Python
- `median_filter.py` – Median filter using Python
- `cython_implementation/filter_cython.pyx` – Cython version of the filters
- `cython_implementation/setup.py` – Cython build script
- `requirements.txt` – Python dependencies

## 🚀 Setup Instructions

### 1. Clone the repo
```bash
git clone https://github.com/your-username/image-filters-performance.git
cd image-filters-performance
