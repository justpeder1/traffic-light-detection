import streamlit as st
import cv2
import torch
from ultralytics import YOLO
import numpy as np
from PIL import Image
import io
import tempfile
import os

# Set page config
st.set_page_config(
    page_title="Yaya ve Trafik Işığı Algılama",
    page_icon="��",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS to match the original template design
st.markdown("""
    <style>
    .main {
        background-color: #f8f9fa;
    }
    .stApp {
        max-width: 100%;
    }
    .main-container {
        background-color: white;
        padding: 2rem;
        border-radius: 10px;
        box-shadow: 0 0 10px rgba(0,0,0,0.1);
        margin: 1rem;
    }
    .uploadfile {
        border: 2px dashed #dee2e6;
        border-radius: 10px;
        padding: 2rem;
        text-align: center;
        margin: 1rem 0;
    }
    .stButton > button {
        width: 100%;
        margin-top: 1rem;
    }
    .confidence-container {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    .stImage {
        max-width: 100%;
        border-radius: 10px;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state for model
if 'model' not in st.session_state:
    # CUDA kontrolü
    if torch.cuda.is_available():
        device = torch.device("cuda")
        st.success("CUDA kullanılabilir. GPU kullanılıyor: " + torch.cuda.get_device_name(0))
    else:
        device = torch.device("cpu")
        st.info("CUDA kullanılamıyor. CPU kullanılıyor.")
    
    try:
        # Model yükleme
        st.session_state.model = YOLO("light.pt")
        st.session_state.model.to(device)
        st.session_state.device = device
    except Exception as e:
        st.error(f"Model yüklenirken hata oluştu: {str(e)}")

# Sidebar navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Sayfa Seçin:", ["Ana Sayfa", "Resim Tespiti", "Video Tespiti", "Gerçek Zamanlı Tespit"])

if page == "Ana Sayfa":
    st.title("Yaya ve Trafik Işığı Algılama Sistemi")
    st.markdown("""
        <div class="main-container">
            <p class="lead">
                Gelişmiş yapay zeka destekli sistem ile yaya ve trafik ışıklarını tespit etme ve analiz etme teknolojisi.
            </p>
        </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
            <div class="feature-card">
                <h3>📷 Resim Tespiti</h3>
                <p>Tek bir resim yükleyerek yaya ve trafik ışığı tespiti yapın.</p>
            </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
            <div class="feature-card">
                <h3>🎥 Video Tespiti</h3>
                <p>Video dosyası ile hareket halinde tespit yapın.</p>
            </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
            <div class="feature-card">
                <h3>🔄 Gerçek Zamanlı Tespit</h3>
                <p>Kamera ile anlık tespit ve analiz yapın.</p>
            </div>
        """, unsafe_allow_html=True)

elif page == "Resim Tespiti":
    st.title("Resim Tespiti")
    
    # Confidence slider
    confidence = st.slider("Güven Eşiği", 0.1, 1.0, 0.3, 0.05)
    
    # File upload
    uploaded_file = st.file_uploader("Resim Yükle", type=['jpg', 'jpeg', 'png'])
    
    # Camera input
    camera_input = st.camera_input("Kamera ile Fotoğraf Çek")
    
    input_image = None
    
    if uploaded_file is not None:
        input_image = Image.open(uploaded_file)
    elif camera_input is not None:
        input_image = Image.open(camera_input)
    
    if input_image is not None:
        # Convert to RGB if necessary
        if input_image.mode != 'RGB':
            input_image = input_image.convert('RGB')
        
        # Show original image
        st.subheader("Orijinal Resim")
        st.image(input_image, use_column_width=True)
        
        # Process button
        if st.button("Tespit Et"):
            with st.spinner("Tespit yapılıyor..."):
                try:
                    # Convert to numpy array
                    image_array = np.array(input_image)
                    
                    # Model prediction
                    results = st.session_state.model.predict(
                        image_array,
                        conf=confidence,
                        imgsz=1280,
                        device=st.session_state.device
                    )
                    
                    # Plot results
                    result_image = results[0].plot()
                    
                    # Show results
                    st.subheader("Tespit Sonucu")
                    st.image(result_image, use_column_width=True)
                    
                except Exception as e:
                    st.error(f"Tespit sırasında bir hata oluştu: {str(e)}")

elif page == "Video Tespiti":
    st.title("Video Tespiti")
    
    # Confidence slider
    confidence = st.slider("Güven Eşiği", 0.1, 1.0, 0.5, 0.05)
    
    # File upload
    uploaded_file = st.file_uploader("Video Yükle", type=['mp4', 'avi', 'mov'])
    
    if uploaded_file is not None:
        # Save uploaded file temporarily
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_file.read())
        
        # Process video button
        if st.button("Video İşle"):
            with st.spinner("Video işleniyor..."):
                try:
                    # Create a temporary file for output
                    output_path = "processed_video.mp4"
                    
                    # Process video frames
                    vid_cap = cv2.VideoCapture(tfile.name)
                    st_frame = st.empty()
                    
                    while vid_cap.isOpened():
                        ret, frame = vid_cap.read()
                        if not ret:
                            break
                        
                        # Convert BGR to RGB
                        frame_rgb = frame[:, :, ::-1]
                        
                        # Model prediction
                        results = st.session_state.model.predict(
                            frame_rgb,
                            conf=confidence,
                            imgsz=1280,
                            device=st.session_state.device
                        )
                        
                        # Plot results
                        result_frame = results[0].plot()
                        
                        # Display frame
                        st_frame.image(result_frame, channels="BGR")
                    
                    vid_cap.release()
                    
                except Exception as e:
                    st.error(f"Video işlenirken bir hata oluştu: {str(e)}")
                finally:
                    # Cleanup
                    os.unlink(tfile.name)

elif page == "Gerçek Zamanlı Tespit":
    st.title("Gerçek Zamanlı Tespit")
    
    # Confidence slider
    confidence = st.slider("Güven Eşiği", 0.1, 1.0, 0.3, 0.05)
    
    # Camera input
    camera_input = st.camera_input("Kamera")
    
    if camera_input is not None:
        try:
            # Convert to PIL Image
            image = Image.open(camera_input)
            
            # Convert to RGB if necessary
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Convert to numpy array
            image_array = np.array(image)
            
            # Model prediction
            results = st.session_state.model.predict(
                image_array,
                conf=confidence,
                imgsz=1280,
                device=st.session_state.device
            )
            
            # Plot results
            result_image = results[0].plot()
            
            # Show results
            st.image(result_image, channels="BGR", use_column_width=True)
            
        except Exception as e:
            st.error(f"Tespit sırasında bir hata oluştu: {str(e)}") 