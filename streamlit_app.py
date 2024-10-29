import streamlit as st
from PIL import Image
import torch
from transformers import AutoImageProcessor, AutoTokenizer, VisionEncoderDecoderModel, YolosForObjectDetection

# Load models and weights
object_detection_processor = AutoImageProcessor.from_pretrained("hustvl/yolos-tiny")
object_detection_model = YolosForObjectDetection.from_pretrained("hustvl/yolos-tiny")
captioning_processor = AutoImageProcessor.from_pretrained("motheecreator/ViT-GPT2-Image-Captioning")
tokenizer = AutoTokenizer.from_pretrained("motheecreator/ViT-GPT2-Image-Captioning")
caption_model = VisionEncoderDecoderModel.from_pretrained("motheecreator/ViT-GPT2-Image-Captioning")

# Streamlit app setup
st.title("Object Detection and Image Captioning")
st.write("Upload an image to detect objects and generate captions.")

# Image upload section
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

def process_image(image):
    # Detect objects in the image using YOLOS
    inputs = object_detection_processor(images=image, return_tensors="pt")
    outputs = object_detection_model(**inputs)
    
    # Extract bounding boxes and label indices
    boxes = outputs.pred_boxes[0].cpu().detach().numpy()
    labels_indices = outputs.logits[0].softmax(-1).argmax(-1).cpu().numpy()
    
    # Filter label indices to include only those present in id2label
    labels = [
        object_detection_model.config.id2label[label_idx] 
        for label_idx in labels_indices 
        if label_idx in object_detection_model.config.id2label
    ]

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
    
    return {'labels': labels, 'captions': captions}, original_caption



def crop_objects(image, boxes):
    cropped_images = []
    for box in boxes:
        # Scale bounding box coordinates to image dimensions
        width, height = image.size
        box = [
            int(box[0] * width),
            int(box[1] * height),
            int(box[2] * width),
            int(box[3] * height),
        ]
        cropped_image = image.crop((box[0], box[1], box[2], box[3]))
        cropped_images.append(cropped_image)
    return cropped_images

# Display results if an image is uploaded
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    results, original_caption = process_image(image)
    
    # Display original image and detected objects
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    # Show captions
    st.write("**Original Image Caption:**", original_caption)
    st.write("**Detected Objects and Captions:**")
    for label, caption in zip(results['labels'], results['captions']):
        st.write(f"Object: {label} - Caption: {caption}")
