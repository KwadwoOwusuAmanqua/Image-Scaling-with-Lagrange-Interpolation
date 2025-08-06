import cv2
import numpy as np
import matplotlib.pyplot as plt

def lagrange_interp(x, y, x_new):
    total = 0.0
    n = len(x)
    for i in range(n):
        term = y[i]
        for j in range(n):
            if i != j:
                denom = x[i] - x[j]
                if denom == 0:
                    continue  # skip to avoid division by zero
                term *= (x_new - x[j]) / denom
        total += term
    return total

def interpolate_1d(data, scale):
    n = len(data)
    x_old = np.arange(n)
    x_new = np.linspace(0, n - 1, int(n * scale))

    y_new = []
    for xi in x_new:
        i = int(np.floor(xi))
        idxs = np.arange(i - 1, i + 3)

        # Clamp to valid range and get unique points to avoid duplicates
        idxs = np.clip(idxs, 0, n - 1)
        x_pts = np.unique(x_old[idxs])
        y_pts = data[np.searchsorted(x_old, x_pts)]

        if len(x_pts) < 2:
            y_new.append(data[min(n - 1, max(0, int(round(xi))))])
        else:
            interpolated = lagrange_interp(x_pts, y_pts, xi)
            y_new.append(np.clip(interpolated, 0, 255))

    return np.array(y_new)

def upscale_image_lagrange(img, scale):
    if img is None:
        raise ValueError("Image could not be loaded. Check the file path.")

    # Interpolate each row
    temp = []
    for row in img:
        interp_row = interpolate_1d(row, scale)
        temp.append(interp_row)
    temp = np.array(temp)

    # Interpolate each column
    temp = temp.T
    result = []
    for col in temp:
        interp_col = interpolate_1d(col, scale)
        result.append(interp_col)
    result = np.array(result).T

    return np.uint8(result)

# --- MAIN PROGRAM ---

# Load grayscale image
img_path = 'letter_A.png'
img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
if img is None:
    raise FileNotFoundError(f"Failed to load image at path: {img_path}")

# Set scaling factor
scale_factor = 2

# Apply Lagrange interpolation
scaled_lagrange = upscale_image_lagrange(img, scale_factor)

# OpenCV comparisons
scaled_nearest = cv2.resize(img, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_NEAREST)
scaled_linear = cv2.resize(img, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_LINEAR)

# Display Results
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.title("Lagrange Interpolation")
plt.imshow(scaled_nearest, cmap='gray')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.title("Nearest Neighbor")
plt.imshow(scaled_lagrange, cmap='gray')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.title("Bilinear (OpenCV)")
plt.imshow(scaled_linear, cmap='gray')
plt.axis('off')

plt.tight_layout()
plt.show()
