import streamlit as st
from PIL import Image
import os
import torch
from transformers import AutoImageProcessor, AutoTokenizer, VisionEncoderDecoderModel

# Load models and weights
weights_path = 'weights/best14.pt'
object_detection_model = torch.hub.load('Mexbow/yolov5_model', 'custom', path=weights_path, autoshape=True)
captioning_processor = AutoImageProcessor.from_pretrained("motheecreator/ViT-GPT2-Image-Captioning")
tokenizer = AutoTokenizer.from_pretrained("motheecreator/ViT-GPT2-Image-Captioning")
caption_model = VisionEncoderDecoderModel.from_pretrained("motheecreator/ViT-GPT2-Image-Captioning")

# Streamlit app setup
st.title("Object Detection and Image Captioning")
st.write("Upload an image to detect objects and generate captions.")

# Image upload section
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

def process_image(image):
    # Detect objects in the image
    results = object_detection_model(image)
    
    # Render image with detected boxes
    img_with_boxes = results.render()[0]
    detected_image_path = 'static/detected_image.jpg'
    img_with_boxes = Image.fromarray(img_with_boxes)
    img_with_boxes.save(detected_image_path)
    
    # Get boxes and labels
    boxes = results.xyxy[0][:, :4].cpu().numpy()
    labels = [results.names[int(x)] for x in results.xyxy[0][:, 5].cpu().numpy()]

    # Caption the original image
    original_inputs = captioning_processor(images=image, return_tensors="pt")
    with torch.no_grad():
        caption_ids = caption_model.generate(**original_inputs)
    original_caption = tokenizer.decode(caption_ids[0], skip_special_tokens=True)

    # Crop objects and caption them
    cropped_images = crop_objects(image, boxes)
    captions = []
    for cropped_image in cropped_images:
        inputs = captioning_processor(images=cropped_image, return_tensors="pt")
        with torch.no_grad():
            caption_ids = caption_model.generate(**inputs)
        caption = tokenizer.decode(caption_ids[0], skip_special_tokens=True)
        captions.append(caption)
    
    return {'labels': labels, 'captions': captions, 'detected_image_path': detected_image_path}, original_caption

def crop_objects(image, boxes):
    cropped_images = []
    for box in boxes:
        cropped_image = image.crop((box[0], box[1], box[2], box[3]))
        cropped_images.append(cropped_image)
    return cropped_images

# Display results if an image is uploaded
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    results, original_caption = process_image(image)
    
    # Display original image and detected image
    st.image(image, caption="Uploaded Image", use_column_width=True)
    st.image(results['detected_image_path'], caption="Detected Objects", use_column_width=True)
    
    # Show captions
    st.write("**Original Image Caption:**", original_caption)
    st.write("**Detected Objects and Captions:**")
    for label, caption in zip(results['labels'], results['captions']):
        st.write(f"Object: {label} - Caption: {caption}")
