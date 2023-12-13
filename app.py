import streamlit as st
from PIL import Image
from inference import preprocess_image, Segmentation
import io
import os

def main():
    st.title("Semantic Segmentation with Streamlit")

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png"])

    if uploaded_file is not None:
        # image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        image = Image.open(uploaded_file).convert("RGB")

        save_path = "uploaded_image.jpg"
        image.save(save_path)

        # input_tensor = preprocess_image(save_path)

        segmented_result = Segmentation(save_path)

        # Display original and segmented images
        st.image([image, segmented_result], caption=["Original Image", "Segmented Image"], width=300, use_column_width=True)

        if os.path.exists(save_path):
            os.remove(save_path)


        st.cache_data.clear()

if __name__ == "__main__":
    main()
