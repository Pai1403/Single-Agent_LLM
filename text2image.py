import requests
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration


def caption_image(image_path):
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large")

    image_path = 'hands-1838658_1280.jpg'  # Changed variable name to image_path
    try:
        raw_image = Image.open(image_path).convert("RGB")  # Open the image using PIL
    except FileNotFoundError:
        print(f"Error: File not found at {image_path}.  Ensure the file exists and the path is correct.")
        exit()

    # conditional image captioning
    text = "a photography of"
    inputs = processor(raw_image, text, return_tensors="pt")

    out = model.generate(**inputs)
    print(processor.decode(out[0], skip_special_tokens=True))

    # unconditional image captioning
    inputs = processor(raw_image, return_tensors="pt")

    out = model.generate(**inputs)
    print(processor.decode(out[0], skip_special_tokens=True))

