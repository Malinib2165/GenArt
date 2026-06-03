from flask import Flask, render_template, request, jsonify
import cv2
import numpy as np
import base64
from io import BytesIO
from PIL import Image

app = Flask(__name__)

# Cartoon and sketch effect processor
def apply_cartoon_effect(image, effect_type, num_colors=8, edge_block_size=9, smoothing_iterations=7):
    """
    Applies a cartoon effect to an input image with improved clarity.

    Args:
        image (numpy.ndarray): The input image in BGR format.
        effect_type (str): The type of effect to apply.
        num_colors (int): Number of colors for quantization.
        edge_block_size (int): Block size for adaptive thresholding.
        smoothing_iterations (int): Number of bilateral filter passes.

    Returns:
        numpy.ndarray: The cartoonized image.
    """
    if effect_type == "cartoon":
        # Ensure edge_block_size is odd
        edge_block_size = int(edge_block_size)
        if edge_block_size % 2 == 0:
            edge_block_size += 1

        # Convert to grayscale and smooth
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        smooth = cv2.medianBlur(gray, 5)

        # Detect edges using adaptive thresholding
        edges = cv2.adaptiveThreshold(
            smooth, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, edge_block_size, 2
        )

        # Invert edges for masking
        edges_inv = cv2.bitwise_not(edges)

        # Apply bilateral filter fewer times for less blur
        color = image.copy()
        for _ in range(max(1, int(smoothing_iterations)//2)):
            color = cv2.bilateralFilter(color, 9, 75, 75)

        # Color quantization using k-means clustering
        data = np.float32(color).reshape((-1, 3))
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 0.001)
        k = max(2, int(num_colors))
        _, labels, centers = cv2.kmeans(data, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        centers = np.uint8(centers)
        quantized = centers[labels.flatten()]
        quantized = quantized.reshape(color.shape)

        # Sharpen quantized image to enhance edges
        kernel = np.array([[0, -1, 0],
                           [-1, 5,-1],
                           [0, -1, 0]])
        sharpened = cv2.filter2D(quantized, -1, kernel)

        # Combine sharpened image with edges using bitwise_and for clarity
        edges_colored = cv2.cvtColor(edges_inv, cv2.COLOR_GRAY2BGR)
        cartoon = cv2.bitwise_and(sharpened, edges_colored)

        return cartoon

    elif effect_type == "pencil_gray":
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        smooth = cv2.medianBlur(gray, 5)
        # The adaptive thresholding creates the black outlines on a white background
        edges = cv2.adaptiveThreshold(
            smooth, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 9
        )
        return cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

    elif effect_type == "oil_painting":
        try:
            return cv2.xphoto.oilPainting(image, 7, 1)
        except AttributeError:
            return cv2.stylization(image, sigma_s=40, sigma_r=0.6)
    elif effect_type == "watercolor":
        return cv2.stylization(image, sigma_s=50, sigma_r=0.3)
    elif effect_type == "disney":
        return cv2.stylization(image, sigma_s=60, sigma_r=0.5)

    return image

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/cartoonify', methods=['POST'])
def cartoonify():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files['file']
    effect = request.form.get('effect', 'cartoon')
    
    # Get parameters for the cartoon effect from the frontend
    num_colors = int(request.form.get('num_colors', 8))
    edge_block_size = int(request.form.get('edge_block_size', 9))
    smoothing_iterations = int(request.form.get('smoothing_iterations', 7))

    try:
        image = Image.open(file.stream).convert('RGB')
        image = np.array(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        processed_image = apply_cartoon_effect(
            image,
            effect,
            num_colors=num_colors,
            edge_block_size=edge_block_size,
            smoothing_iterations=smoothing_iterations
        )
        processed_image = cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB)

        _, buffer = cv2.imencode('.png', processed_image)
        encoded_image = base64.b64encode(buffer).decode('utf-8')

        return jsonify({"processed_image": f"data:image/png;base64,{encoded_image}"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
