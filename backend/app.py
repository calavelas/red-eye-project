from flask import Flask, request, jsonify, send_file
from flask_restful import Resource, Api
import imageio
import cv2
import io
from io import BytesIO
import os
import base64
from PIL import Image
import dlib
import numpy as np
import os
import uuid
from flask_cors import CORS
# from IPython.display import Image as IPImage, display

app = Flask(__name__)
api = Api(app)
CORS(app)

template_path = 'backend/static/laser_template.png'
dlib_model_path = "backend/static/shape_predictor_68_face_landmarks.dat"

def crop_to_square(image):
    shift_north_ratio = -0.5  # Adjust this value based on the desired position of the cropped image
    # Get the eye centers
    eye_centers = get_eye_centers(image)

    # Calculate the midpoint between the eye centers
    midpoint = ((eye_centers[0][0] + eye_centers[1][0]) // 2, (eye_centers[0][1] + eye_centers[1][1]) // 2)

    # Shift the midpoint north by 15% of the distance between the eyes
    eye_distance = int(((eye_centers[0][0] - eye_centers[1][0])**2 + (eye_centers[0][1] - eye_centers[1][1])**2)**0.5)
    north_shift = int(shift_north_ratio * eye_distance)
    shifted_midpoint = (midpoint[0], midpoint[1] - north_shift)

    # Calculate the distance between the shifted midpoint and the edges of the image
    height, width = image.shape[:2]
    distance = min(shifted_midpoint[0], width - shifted_midpoint[0], shifted_midpoint[1], height - shifted_midpoint[1])

    # Crop the image to a square around the shifted midpoint with the given distance
    x = shifted_midpoint[0] - distance
    y = shifted_midpoint[1] - distance
    cropped_image = image[y:y+(2*distance), x:x+2*distance]

    #resize cropped image to 500x500
    cropped_image = cv2.resize(cropped_image, (500, 500))

    # # Save the cropped image
    # cv2.imwrite(output_path, cropped_image)
    
    return cropped_image

def get_eye_centers(image):
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(dlib_model_path)
    
    gray_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
    rects = detector(gray_image, 1)
    eye_centers = []

    for rect in rects:
        shape = predictor(gray_image, rect)
        
        # Left eye landmarks
        left_eye_landmarks = []
        for i in range(36, 42):
            x, y = shape.part(i).x, shape.part(i).y
            left_eye_landmarks.append((x, y))
        left_eye_center = np.mean(left_eye_landmarks, axis=0).astype(int)
        eye_centers.append(tuple(left_eye_center))

        # Right eye landmarks
        right_eye_landmarks = []
        for i in range(42, 48):
            x, y = shape.part(i).x, shape.part(i).y
            right_eye_landmarks.append((x, y))
        right_eye_center = np.mean(right_eye_landmarks, axis=0).astype(int)
        eye_centers.append(tuple(right_eye_center))

    return eye_centers

def apply_glowing_red_eye(image, eye_centers, template_path):
    # Load the template image
    template = cv2.imread(template_path, cv2.IMREAD_UNCHANGED)

    # Make a copy of the input image
    processed_image = image.copy()

    # Apply the glowing red eye effect to each eye center separately
    for eye_center in eye_centers:
        x, y = eye_center

        # Resize the template based on the distance between the eye centers
        eye_distance = 75
        template_scale = eye_distance / 160.0
        template_width = int(template_scale * image.shape[1])
        template_height = int(template_width * template.shape[0] / template.shape[1])
        resized_template = cv2.resize(template, (template_width, template_height))

        # Calculate the position of the template based on the eye center coordinates
        template_x = x - template_width // 2
        template_y = y - template_height // 2

        # Blend the template with the image
        alpha = resized_template[..., 3] / 255.0
        processed_image[template_y:template_y+template_height, template_x:template_x+template_width] = (
            alpha[:, :, np.newaxis] * resized_template[..., :3] +
            (1 - alpha[:, :, np.newaxis]) * processed_image[template_y:template_y+template_height, template_x:template_x+template_width]
        )

    return processed_image

def create_glowing_red_eye_static(image):
    # Load the input image and crop it to a square
    image = crop_to_square(image)
    # Get the eye centers
    eye_centers = get_eye_centers(image)

    image = apply_glowing_red_eye(image, eye_centers, template_path)
    return image

def create_glowing_red_eye_animation(image, fps=10, num_frames=10):
    # Load the input image and crop it to a square
    image = crop_to_square(image)
    # Get the eye centers
    eye_centers = get_eye_centers(image)

    # Create a list of blended images for each frame
    frames = []
    for i in range(num_frames):
        # Calculate the blending ratio between the original image and the glowing red eye image
        ratio = (i + 1) / num_frames

        # Blend the original image with the glowing red eye image using the blending ratio
        frame = cv2.addWeighted(image, 1-ratio, apply_glowing_red_eye(image, eye_centers, template_path), ratio, 0)

        # Convert the image to RGB and add it to the list of images
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(Image.fromarray(frame))

    # Reverse the order of the blended images and append them to the existing list of images
    for i in range(num_frames):
        frames.append(frames[num_frames-1-i])

    # return frames
    return frames

from flask import make_response

@app.route('/process_image/', methods=['POST'])
def process_image_endpoint():

    file = request.files['file']
    if file.filename == '':
        return "No selected file", 400
    input_image = io.BytesIO(file.read())

    # Convert the input image to a numpy array and decode it using OpenCV
    np_image = np.frombuffer(input_image.getvalue(), dtype=np.uint8)
    img = cv2.imdecode(np_image, cv2.IMREAD_UNCHANGED)

    # Check if the image is properly decoded
    if img is None:
        return "Error: Image could not be decoded. Please upload a valid JPEG or PNG file.", 400

    # Process the image
    processed_img = create_glowing_red_eye_static(img)
    processed_img = np.array(processed_img)  # Convert the processed image to a numpy array

    # Encode the processed image as a PNG
    output_image = io.BytesIO(cv2.imencode('.png', processed_img)[1].tobytes())
    output_image.seek(0)

    # Create a response with the output image and the Content-Disposition header
    response = make_response(send_file(output_image, mimetype='image/png'))
    response.headers.set('Content-Disposition', 'attachment', filename='output.png')

    return response

@app.route('/process_image_gif/', methods=['POST'])
def process_image_gif_endpoint():

    file = request.files['file']
    if file.filename == '':
        return "No selected file", 400
    input_image = io.BytesIO(file.read())

    np_image = np.frombuffer(input_image.getvalue(), dtype=np.uint8)
    img = cv2.imdecode(np_image, cv2.IMREAD_UNCHANGED)

    processed_images = create_glowing_red_eye_animation(img)

    # Save the images as an in-memory GIF file
    output_image = io.BytesIO()
    processed_images[0].save(output_image, format='GIF', save_all=True, append_images=processed_images[1:], duration=int(800/10), loop=0)
    output_image.seek(0)

    # Create a response with the output image and the Content-Disposition header
    response = make_response(send_file(output_image, mimetype='image/gif'))
    response.headers.set('Content-Disposition', 'attachment', filename='output.gif')

    return response

if __name__ == "__main__":
    app.run(debug=True)






