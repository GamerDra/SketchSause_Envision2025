# 🎨 SketchSause - Pix2Pix Demo

Turn your **satellite images into maps** and **sketches into anime art** using Pix2Pix-style image-to-image translation models.

---

## Demo Features

- **Map Generator**: Convert satellite images into corresponding maps.
- **Anime Generator**: Convert hand-drawn sketches into anime-style colored images.

---

## Installation Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/GamerDra/Pix2Pix_Envision2025.git
cd Pix2Pix_Envision2025/demo
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Download Pretrained Weights

- Download the pretrained model weights from this Google Drive link : https://drive.google.com/drive/folders/1U37T78977ZUBJOwRZ0F6SETx0FEMpgqV?usp=sharing
- Once downloaded, place the files in the following structure:
```
├── demo/
│   ├── app.py
│   └── demo_images/
│       ├── map/
│       └── anime/
├── models/
│   ├── anime/
│   │   ├── anime_generator.py
│   │   └── Anime_gen_weights.pth.tar   <-- Place here
│   └── map/
│       ├── map_generator.py
│       └── Map_generator_weights.pth   <-- Place here
```

### 4. Run the app

```bash
streamlit run app.py
```

