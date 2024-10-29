import streamlit as st
from PIL import Image
import torch
from transformers import AutoImageProcessor, AutoTokenizer, VisionEncoderDecoderModel, YolosForObjectDetection
from PIL import ImageDraw

@st.cache_resource
def load_models():
    # Load object detection model and processor
    object_detection_model = YolosForObjectDetection.from_pretrained("hustvl/yolos-tiny")
    object_detection_processor = YolosImageProcessor.from_pretrained("hustvl/yolos-tiny")

    # Load captioning model and processor
    caption_model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
    captioning_processor = AutoProcessor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
    tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")

    return object_detection_model, object_detection_processor, caption_model, captioning_processor, tokenizer

# Load models once and reuse them
object_detection_model, object_detection_processor, caption_model, captioning_processor, tokenizer = load_models()

# Streamlit app setup
st.title("Object Detection and Image Captioning")
st.write("Upload an image to detect objects and generate captions.")

# Image upload section
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

def process_image(image):
    # Detect objects in the image using YOLOS
    inputs = object_detection_processor(images=image, return_tensors="pt")
    outputs = object_detection_model(**inputs)

    # Extract bounding boxes and labels
    boxes = outputs.pred_boxes[0].cpu().detach().numpy()
    labels_indices = outputs.logits[0].softmax(-1).argmax(-1).cpu().numpy()
    
    # Filter labels and map them to id2label
    labels = [
        object_detection_model.config.id2label[label_idx]
        for label_idx in labels_indices
        if label_idx in object_detection_model.config.id2label
    ]

    # Draw bounding boxes on the image
    width, height = image.size
    draw = ImageDraw.Draw(image)
    for box, label in zip(boxes, labels):
        # Convert normalized coordinates to absolute pixel values
        left = int(box[0] * width)
        upper = int(box[1] * height)
        right = int(box[2] * width)
        lower = int(box[3] * height)
        
        # Ensure the coordinates are valid
        if left < right and upper < lower:
            # Draw the bounding box
            draw.rectangle([(left, upper), (right, lower)], outline="red", width=3)
            draw.text((left, upper), label, fill="red")

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
    width, height = image.size
    cropped_images = []
    for box in boxes:
        # Convert normalized coordinates to absolute pixel values
        left = int(box[0] * width)
        upper = int(box[1] * height)
        right = int(box[2] * width)
        lower = int(box[3] * height)
        
        # Ensure coordinates are in the correct order
        if right > left and lower > upper:
            cropped_image = image.crop((left, upper, right, lower))
            cropped_images.append(cropped_image)
        else:
            # Skip invalid boxes
            print(f"Invalid box coordinates: {(left, upper, right, lower)}")

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
