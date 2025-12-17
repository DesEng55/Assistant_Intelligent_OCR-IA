import cv2
import easyocr
import numpy as np
from PIL import Image
import streamlit as st
from typing import List, Tuple, Dict, Optional
import logging

class OCRPipeline:
    def __init__(self, languages: List[str] = ['fr', 'en'], gpu: bool = True):
        """
        Initialise le pipeline OCR avec EasyOCR et OpenCV
        
        Args:
            languages: Liste des langues supportées
            gpu: Utiliser GPU si disponible
        """
        self.languages = languages
        self.gpu = gpu
        self.reader = self._initialize_reader()
        
    def _validate_languages(self, languages: List[str]) -> List[str]:
        """
        Valide et corrige les combinaisons de langues pour EasyOCR
        
        Args:
            languages: Liste des langues demandées
            
        Returns:
            Liste des langues validées
        """
        # Langues chinoises qui nécessitent l'anglais
        chinese_langs = ['ch_sim', 'ch_tra']
        
        # Si on a du chinois mais pas d'anglais, ajouter l'anglais
        has_chinese = any(lang in chinese_langs for lang in languages)
        has_english = 'en' in languages
        
        if has_chinese and not has_english:
            logging.warning("Chinois détecté sans anglais. Ajout automatique de l'anglais.")
            languages = languages + ['en']
        
        # Retirer les langues non-anglaises si on a du chinois traditionnel
        if 'ch_tra' in languages:
            # Garder seulement ch_tra et en
            validated = ['ch_tra', 'en']
            if 'ch_sim' in languages:
                validated.insert(0, 'ch_sim')
            logging.info(f"Chinois traditionnel détecté. Langues ajustées à: {validated}")
            return validated
        
        return languages
    
    def _initialize_reader(self) -> Optional[easyocr.Reader]:
        """Initialise le lecteur EasyOCR"""
        try:
            # Valider les langues avant initialisation
            validated_langs = self._validate_languages(self.languages)
            
            reader = easyocr.Reader(validated_langs, gpu=self.gpu)
            self.languages = validated_langs  # Mettre à jour avec les langues validées
            return reader
        except Exception as e:
            logging.error(f"Erreur lors de l'initialisation d'EasyOCR: {str(e)}")
            # Fallback sans GPU
            try:
                logging.info("Tentative d'initialisation sans GPU...")
                validated_langs = self._validate_languages(self.languages)
                reader = easyocr.Reader(validated_langs, gpu=False)
                self.languages = validated_langs
                return reader
            except Exception as e2:
                logging.error(f"Erreur critique EasyOCR: {str(e2)}")
                return None
    
    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """
        Préprocessing de l'image pour améliorer la qualité OCR
        
        Args:
            image: Image numpy array
            
        Returns:
            Image préprocessée
        """
        # Conversion en niveaux de gris si nécessaire
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Débruitage
        denoised = cv2.fastNlMeansDenoising(gray)
        
        # Amélioration du contraste
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(denoised)
        
        # Binarisation adaptative
        binary = cv2.adaptiveThreshold(
            enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY, 11, 2
        )
        
        return binary
    
    def extract_text_from_image(self, image: np.ndarray, preprocess: bool = True) -> Dict:
        """
        Extrait le texte d'une image
        
        Args:
            image: Image numpy array
            preprocess: Appliquer le préprocessing
            
        Returns:
            Dictionnaire avec texte extrait et métadonnées
        """
        if self.reader is None:
            return {"text": "", "confidence": 0, "boxes": [], "error": "OCR non initialisé"}
        
        try:
            # Préprocessing si demandé
            processed_image = self.preprocess_image(image) if preprocess else image
            
            # Extraction OCR
            results = self.reader.readtext(processed_image)
            
            # Traitement des résultats
            extracted_text = ""
            boxes = []
            confidences = []
            
            for (bbox, text, confidence) in results:
                if confidence > 0.5:  # Seuil de confiance
                    extracted_text += text + " "
                    boxes.append(bbox)
                    confidences.append(confidence)
            
            avg_confidence = np.mean(confidences) if confidences else 0
            
            return {
                "text": extracted_text.strip(),
                "confidence": float(avg_confidence),
                "boxes": boxes,
                "raw_results": results,
                "error": None
            }
            
        except Exception as e:
            logging.error(f"Erreur extraction OCR: {str(e)}")
            return {"text": "", "confidence": 0, "boxes": [], "error": str(e)}
    
    def draw_boxes(self, image: np.ndarray, boxes: List, texts: List = None) -> np.ndarray:
        """
        Dessine les boîtes de détection sur l'image
        
        Args:
            image: Image originale
            boxes: Liste des boîtes de détection
            texts: Textes correspondants (optionnel)
            
        Returns:
            Image avec boîtes dessinées
        """
        if not boxes:
            return image
            
        result_image = image.copy()
        
        for i, box in enumerate(boxes):
            # Conversion des coordonnées
            pts = np.array(box, dtype=np.int32)
            
            # Dessiner le rectangle
            cv2.polylines(result_image, [pts], True, (0, 255, 0), 2)
            
            # Ajouter le texte si disponible
            if texts and i < len(texts):
                cv2.putText(result_image, texts[i], 
                           (pts[0][0], pts[0][1] - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
        
        return result_image
    
    def process_video_frame(self, frame: np.ndarray) -> Dict:
        """
        Traite une frame vidéo pour extraction OCR temps réel
        
        Args:
            frame: Frame vidéo
            
        Returns:
            Résultats OCR
        """
        # Redimensionner pour performance
        height, width = frame.shape[:2]
        if width > 800:
            scale = 800 / width
            new_width = int(width * scale)
            new_height = int(height * scale)
            frame = cv2.resize(frame, (new_width, new_height))
        
        return self.extract_text_from_image(frame, preprocess=True)
    
    def update_languages(self, languages: List[str]):
        """Met à jour les langues supportées"""
        if languages != self.languages:
            # Valider les langues
            validated_langs = self._validate_languages(languages)
            
            if validated_langs != languages:
                logging.warning(f"Langues ajustées de {languages} à {validated_langs} pour compatibilité EasyOCR")
            
            self.languages = validated_langs
            self.reader = None  # Force réinitialisation
            self.reader = self._initialize_reader()