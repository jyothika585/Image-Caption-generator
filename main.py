import streamlit as st
from PIL import Image
import torch
from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer
import torchvision.transforms as transforms


@st.cache_resource
def load_ai_model():
    """Load the ViT-GPT2 model and its processors."""
    model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
    image_processor = ViTImageProcessor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
    tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
    return model, image_processor, tokenizer

# Preprocess the uploaded image
def preprocess_image(image):
    """Convert image to tensor and normalize it."""
    if image.mode == 'RGBA':
        image = image.convert('RGB')

    transform = transforms.Compose([
        transforms.Resize((224, 224)),  
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    return transform(image).unsqueeze(0)  

# Generate caption
def generate_caption(image_tensor, model, tokenizer):
    """Generate caption using the model."""
    try:
        with torch.no_grad():
            output_ids = model.generate(
                image_tensor,
                max_length=50,  
                num_beams=5,  
                pad_token_id=tokenizer.pad_token_id
            )  
            return tokenizer.decode(output_ids[0], skip_special_tokens=True)
    except Exception as e:
        st.error(f"Generation error: {str(e)}")
        return None

# Streamlit UI
def main():
    st.set_page_config(page_title="AI Image Caption Generator", page_icon="🖼️", layout="centered")

    st.title("🖼️ AI Image Caption Generator")
    st.subheader("Upload an image and let AI generate a caption for you!")

    
    model, image_processor, tokenizer = load_ai_model()

    
    uploaded_file = st.file_uploader("📤 Upload an image (JPG, JPEG, PNG)", type=["jpg", "jpeg", "png"])

    if uploaded_file:
        image = Image.open(uploaded_file)

        
        col1, col2 = st.columns([3, 2])

        with col1:
            st.image(image, caption="Uploaded Image", use_container_width=True)

        with col2:
            st.markdown("### 🛠️ **AI-Powered Captioning**")
            st.write("Click the button below to generate an AI-powered caption for this image.")

            if st.button("✨ Generate Caption", use_container_width=True):
                with st.spinner("🤖 AI is thinking..."):
                    image_tensor = preprocess_image(image)
                    caption = generate_caption(image_tensor, model, tokenizer)

                    if caption:
                        st.success("📝 **Generated Caption:**")
                        st.markdown(f" {caption}")

if __name__ == "__main__":
    main()
