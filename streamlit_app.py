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
    page_icon="🚦",
    layout="wide"
)

# Custom CSS for mobile-friendly design
st.markdown("""
    <style>
    .stApp {
        max-width: 100%;
        padding: 1rem;
    }
    .main {
        max-width: 1200px;
        margin: 0 auto;
    }
    .uploadfile {
        border: 2px dashed #ccc;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        margin: 20px 0;
    }
    .stSlider {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    .stImage {
        max-width: 100%;
        height: auto;
        border-radius: 10px;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if 'model' not in st.session_state:
    # CUDA kontrolü
    if torch.cuda.is_available():
        device = torch.device("cuda")
        st.success("CUDA kullanılabilir. GPU kullanılıyor: " + torch.cuda.get_device_name(0))
    else:
        device = torch.device("cpu")
        st.info("CUDA kullanılamıyor. CPU kullanılıyor.")
    
    # Model yükleme
    st.session_state.model = YOLO("light.pt")
    st.session_state.model.to(device)

# Sidebar navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Sayfa Seçin:", ["Ana Sayfa", "Resim Tespiti", "Video Tespiti", "Gerçek Zamanlı Tespit"])

if page == "Ana Sayfa":
    st.title("Yaya ve Trafik Işığı Algılama Sistemi")
    st.markdown("""
        Gelişmiş yapay zeka destekli sistem ile yaya ve trafik ışıklarını tespit etme ve analiz etme teknolojisi.
    """)

    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.card(
            title="Resim Tespiti",
            text="Tek bir resim yükleyerek yaya ve trafik ışığı tespiti yapın.",
        )
    
    with col2:
        st.card(
            title="Video Tespiti",
            text="Video dosyası ile hareket halinde tespit yapın.",
        )
    
    with col3:
        st.card(
            title="Gerçek Zamanlı Tespit",
            text="Kamera ile anlık tespit ve analiz yapın.",
        )

    st.markdown("### Proje Hakkında")
    st.write("""
        Bu yaya ve trafik ışığı tespit sistemi, en son teknoloji bilgisayarlı görü ve derin öğrenme tekniklerini 
        kullanarak çeşitli trafik öğelerini tespit eder:
    """)
    st.markdown("""
        - Kırmızı Işık
        - Sarı Işık
        - Yeşil Işık
        - Yaya
    """)
    st.write("""
        Sistem, YOLOv11 nesne tespit modeli kullanarak çalışır ve kapsamlı bir veri seti üzerinde eğitilmiştir. 
        Yüksek doğruluk ve performans ile gerçek zamanlı tespit yapabilir.
    """)

elif page == "Resim Tespiti":
    st.title("Resim Tespiti")
    st.markdown("Resim yükleyerek yaya ve trafik ışığı tespiti yapın.")

    # Confidence slider
    confidence = st.slider("Güven Eşiği", 0.1, 1.0, 0.3, 0.05)

    # File uploader
    uploaded_file = st.file_uploader("Resim Yükle", type=['jpg', 'jpeg', 'png'])

    if uploaded_file is not None:
        # Convert uploaded file to image
        image = Image.open(uploaded_file)
        image_array = np.array(image)

        # Show original image
        st.subheader("Orijinal Resim")
        st.image(image, use_column_width=True)

        # Process image
        if st.button("Tespit Et"):
            with st.spinner("Tespit yapılıyor..."):
                # Model prediction
                results = st.session_state.model.predict(
                    image_array,
                    conf=confidence,
                    imgsz=1280,
                    device=st.session_state.model.device
                )
                
                # Plot results
                result_image = results[0].plot()
                
                # Show result
                st.subheader("Tespit Sonucu")
                st.image(result_image, use_column_width=True)

elif page == "Video Tespiti":
    st.title("Video Tespiti")
    st.markdown("Video yükleyerek yaya ve trafik ışığı tespiti yapın.")

    # Confidence slider
    confidence = st.slider("Güven Eşiği", 0.1, 1.0, 0.5, 0.05)

    # File uploader
    uploaded_file = st.file_uploader("Video Yükle", type=['mp4', 'avi', 'mov', 'webm'])

    if uploaded_file is not None:
        # Save uploaded file temporarily
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_file.read())
        
        # Show video info
        vid = cv2.VideoCapture(tfile.name)
        fps = int(vid.get(cv2.CAP_PROP_FPS))
        frame_count = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = frame_count/fps
        width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        st.write(f"Video Bilgileri:")
        st.write(f"- Süre: {int(duration//60)}:{int(duration%60)} dakika")
        st.write(f"- Boyut: {width}x{height} piksel")
        st.write(f"- FPS: {fps}")

        if st.button("Video İşle"):
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Create output video writer
            output_path = "processed_video.mp4"
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            
            frame_count = 0
            while vid.isOpened():
                ret, frame = vid.read()
                if not ret:
                    break
                
                # Model prediction
                results = st.session_state.model.predict(
                    frame,
                    conf=confidence,
                    imgsz=1280,
                    device=st.session_state.model.device
                )
                
                # Plot results
                result_frame = results[0].plot()
                
                # Write frame
                out.write(result_frame)
                
                # Update progress
                frame_count += 1
                progress = int((frame_count / frame_count) * 100)
                progress_bar.progress(progress)
                status_text.text(f"İşleniyor... {progress}%")
            
            vid.release()
            out.release()
            
            # Show processed video
            st.video(output_path)
            
            # Cleanup
            os.unlink(tfile.name)
            os.unlink(output_path)

elif page == "Gerçek Zamanlı Tespit":
    st.title("Gerçek Zamanlı Tespit")
    st.markdown("Kamera ile gerçek zamanlı tespit yapın.")
    
    # Confidence slider
    confidence = st.slider("Güven Eşiği", 0.1, 1.0, 0.3, 0.05)

    # Camera input
    camera_image = st.camera_input("Kamera")

    if camera_image is not None:
        # Convert image to numpy array
        image = Image.open(camera_image)
        image_array = np.array(image)

        # Process image
        results = st.session_state.model.predict(
            image_array,
            conf=confidence,
            imgsz=1280,
            device=st.session_state.model.device
        )
        
        # Plot results
        result_image = results[0].plot()
        
        # Show result
        st.image(result_image, use_column_width=True) 