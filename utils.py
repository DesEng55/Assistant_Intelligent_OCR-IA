import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io
import base64
from typing import Tuple, Optional, List
import logging
import time
from datetime import datetime

def setup_logging():
    """Configure le système de logging"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('ocr_assistant.log'),
            logging.StreamHandler()
        ]
    )

def load_image(uploaded_file) -> Optional[np.ndarray]:
    """
    Charge une image uploadée
    
    Args:
        uploaded_file: Fichier uploadé via Streamlit
        
    Returns:
        Image numpy array ou None
    """
    try:
        if uploaded_file is not None:
            # Lire l'image
            image = Image.open(uploaded_file)
            
            # Convertir en RGB si nécessaire
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Convertir en numpy array
            image_array = np.array(image)
            
            # Convertir RGB en BGR pour OpenCV
            image_bgr = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
            
            return image_bgr
        
        return None
        
    except Exception as e:
        st.error(f"Erreur lors du chargement de l'image: {str(e)}")
        return None

def resize_image(image: np.ndarray, max_width: int = 800, max_height: int = 600) -> np.ndarray:
    """
    Redimensionne une image en gardant le ratio
    
    Args:
        image: Image numpy array
        max_width: Largeur maximale
        max_height: Hauteur maximale
        
    Returns:
        Image redimensionnée
    """
    height, width = image.shape[:2]
    
    # Calculer le ratio de redimensionnement
    width_ratio = max_width / width
    height_ratio = max_height / height
    ratio = min(width_ratio, height_ratio)
    
    if ratio < 1:
        new_width = int(width * ratio)
        new_height = int(height * ratio)
        resized = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
        return resized
    
    return image

def image_to_base64(image: np.ndarray) -> str:
    """
    Convertit une image numpy en base64
    
    Args:
        image: Image numpy array
        
    Returns:
        String base64
    """
    try:
        # Convertir BGR en RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Convertir en PIL Image
        pil_image = Image.fromarray(image_rgb)
        
        # Convertir en bytes
        buffer = io.BytesIO()
        pil_image.save(buffer, format='PNG')
        
        # Encoder en base64
        img_str = base64.b64encode(buffer.getvalue()).decode()
        
        return f"data:image/png;base64,{img_str}"
        
    except Exception as e:
        logging.error(f"Erreur conversion base64: {str(e)}")
        return ""

def create_download_link(text: str, filename: str) -> str:
    """
    Crée un lien de téléchargement pour du texte
    
    Args:
        text: Contenu à télécharger
        filename: Nom du fichier
        
    Returns:
        HTML du lien de téléchargement
    """
    b64 = base64.b64encode(text.encode()).decode()
    href = f'<a href="data:file/txt;base64,{b64}" download="{filename}">📥 Télécharger {filename}</a>'
    return href

def format_confidence(confidence: float) -> str:
    """
    Formate le niveau de confiance avec couleur
    
    Args:
        confidence: Niveau de confiance (0-1)
        
    Returns:
        HTML formaté avec couleur
    """
    percentage = confidence * 100
    
    if percentage >= 80:
        color = "green"
        icon = "✅"
    elif percentage >= 60:
        color = "orange"
        icon = "⚠️"
    else:
        color = "red"
        icon = "❌"
    
    return f'<span style="color: {color};">{icon} {percentage:.1f}%</span>'

def create_history_entry(text: str, translation: str, source_lang: str, target_lang: str) -> dict:
    """
    Crée une entrée d'historique
    
    Args:
        text: Texte original
        translation: Traduction
        source_lang: Langue source
        target_lang: Langue cible
        
    Returns:
        Dictionnaire d'historique
    """
    return {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'original_text': text,
        'translation': translation,
        'source_lang': source_lang,
        'target_lang': target_lang
    }

def display_history(history: List[dict]):
    """
    Affiche l'historique des traductions
    
    Args:
        history: Liste des entrées d'historique
    """
    if not history:
        st.info("Aucun historique disponible")
        return
    
    st.subheader("📜 Historique des Traductions")
    
    for i, entry in enumerate(reversed(history[-10:])):  # 10 dernières entrées
        with st.expander(f"🕐 {entry['timestamp']} - {entry['source_lang']} → {entry['target_lang']}"):
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Texte original:**")
                st.write(entry['original_text'][:200] + "..." if len(entry['original_text']) > 200 else entry['original_text'])
            
            with col2:
                st.write("**Traduction:**")
                st.write(entry['translation'][:200] + "..." if len(entry['translation']) > 200 else entry['translation'])

def validate_image_file(uploaded_file) -> bool:
    """
    Valide un fichier image uploadé
    
    Args:
        uploaded_file: Fichier uploadé
        
    Returns:
        True si valide, False sinon
    """
    if uploaded_file is None:
        return False
    
    # Vérifier la taille
    if uploaded_file.size > 10 * 1024 * 1024:  # 10MB
        st.error("Le fichier est trop volumineux (maximum 10MB)")
        return False
    
    # Vérifier le format
    allowed_types = ['image/jpeg', 'image/jpg', 'image/png', 'image/bmp', 'image/tiff', 'image/webp']
    if uploaded_file.type not in allowed_types:
        st.error("Format de fichier non supporté. Utilisez JPG, PNG, BMP, TIFF ou WebP.")
        return False
    
    return True

def create_metrics_display(ocr_results: dict, processing_time: float):
    """
    Affiche les métriques de performance
    
    Args:
        ocr_results: Résultats OCR
        processing_time: Temps de traitement
    """
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Confiance OCR", f"{ocr_results.get('confidence', 0)*100:.1f}%")
    
    with col2:
        st.metric("Temps de traitement", f"{processing_time:.2f}s")
    
    with col3:
        text_length = len(ocr_results.get('text', ''))
        st.metric("Caractères extraits", text_length)
    
    with col4:
        boxes_count = len(ocr_results.get('boxes', []))
        st.metric("Zones détectées", boxes_count)

class PerformanceMonitor:
    """Moniteur de performance pour optimisation"""
    
    def __init__(self):
        self.start_time = None
        self.metrics = {}
    
    def start(self, operation: str):
        """Démarre le chronométrage d'une opération"""
        self.start_time = time.time()
        self.current_operation = operation
    
    def end(self):
        """Termine le chronométrage et enregistre"""
        if self.start_time:
            duration = time.time() - self.start_time
            self.metrics[self.current_operation] = duration
            self.start_time = None
            return duration
        return 0
    
    def get_metrics(self) -> dict:
        """Retourne les métriques collectées"""
        return self.metrics.copy()
    
    def display_metrics(self):
        """Affiche les métriques dans Streamlit"""
        if self.metrics:
            st.subheader("📊 Métriques de Performance")
            for operation, duration in self.metrics.items():
                st.write(f"**{operation}**: {duration:.3f}s")

# Initialiser le monitoring global
performance_monitor = PerformanceMonitor()