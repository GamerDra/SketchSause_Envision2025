import streamlit as st
import torch
import torchvision.transforms as transforms
from PIL import Image
import os
import random
from models.map.map_generator import Generator as MapGenerator
from models.anime.anime_generator import Generator as AnimeGenerator  

image_size = (256, 256)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

st.set_page_config(page_title="Pix2Pix Demo", layout="centered")
st.title("🎨 SketchSause")

@st.cache_resource
def load_map_model():
    model = MapGenerator().to(device)
    model.load_state_dict(torch.load("models/map/Map_generator_weights.pth", map_location=device))  
    model.eval()
    return model

@st.cache_resource
def load_anime_model():
    model = AnimeGenerator().to(device)
    
    checkpoint = torch.load("models/anime/Anime_gen_weights.pth.tar", map_location=device)
    
    if "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        new_key = k.replace("module.", "") if k.startswith("module.") else k
        new_state_dict[new_key] = v

    model.load_state_dict(new_state_dict)
    model.eval()
    return model


map_gen = load_map_model()
anime_gen = load_anime_model()

input_transform = transforms.Compose([
    transforms.Resize(image_size),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
])
output_transform = transforms.Compose([
    transforms.Normalize(mean=[-1]*3, std=[1/0.5]*3),  
    transforms.ToPILImage()
])

tab_anime, tab_map = st.tabs(["🎭 Anime Dataset", "🗺️ Map Dataset"])

with tab_map:
    st.header("Generate Map using Satellite Image")

    demo_folder = "demo_images/map"
    demo_files = [f for f in os.listdir(demo_folder) if f.lower().endswith(("jpg", "jpeg", "png"))]

    uploaded_file = st.file_uploader("Upload image", type=["jpg", "jpeg", "png"], key="map_upload")
    use_random = st.button("🔀 Random Image", key="map_random")

    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
    elif use_random and demo_files:
        image_path = os.path.join(demo_folder, random.choice(demo_files))
        image = Image.open(image_path).convert("RGB")
        st.info("Random demo image selected.")
    elif demo_files:
        image_path = os.path.join(demo_folder, random.choice(demo_files))
        image = Image.open(image_path).convert("RGB")
        st.info("Loaded random demo image.")
    else:
        image = None

    if image:
        st.image(image, caption="Input Image", use_column_width=True)
        width, height = image.size
        if width == 1200:
            st.success("Detected concatenated image (1200px). Using left 600px as input.")
            input_img = image.crop((0, 0, 600, height))
        elif width == 600:
            st.success("Detected single satellite image (600px).")
            input_img = image
        else:
            st.warning("Width must be 600px (input only) or 1200px (concat image).")
            st.stop()

        input_tensor = input_transform(input_img).unsqueeze(0).to(device)

        with torch.no_grad():
            output_tensor = map_gen(input_tensor)[0].cpu()

        output_img = output_transform(output_tensor)

        st.subheader("Generated Output")
        st.image(output_img, caption="Generated Map", use_column_width=True)

with tab_anime:
    st.header("Turn your sketches into anime")

    demo_folder = "demo_images/anime"
    demo_files = [f for f in os.listdir(demo_folder) if f.lower().endswith(("jpg", "jpeg", "png"))]

    uploaded_file = st.file_uploader("Upload image", type=["jpg", "jpeg", "png"], key="anime_upload")

    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
    elif st.button("🔀 Random Image", key="anime_random") and demo_files:
        image_path = os.path.join(demo_folder, random.choice(demo_files))
        image = Image.open(image_path).convert("RGB")
        st.info("Random demo image selected.")
    elif demo_files:
        image_path = os.path.join(demo_folder, random.choice(demo_files))
        image = Image.open(image_path).convert("RGB")
        st.info("Loaded random demo image.")
    else:
        image = None

    if image:
        st.image(image, caption="Input Sketch", use_column_width=True)

        width, height = image.size
        right_half = image.crop((width // 2, 0, width, height))
        right_half = right_half.resize((256, 256), Image.LANCZOS)


        input_tensor = input_transform(right_half).unsqueeze(0).to(device)

        with torch.no_grad():
            output_tensor = anime_gen(input_tensor)[0].cpu()

        output_img = output_transform(output_tensor)

        st.subheader("Generated Anime Output")
        st.image(output_img, caption="Generated Anime Image from Right Half Input", use_column_width=True)
