import streamlit as st
import cv2
import numpy as np
from PIL import Image
import time
import logging
from typing import Dict, List, Optional

# Imports locaux
from ocr_pipeline import OCRPipeline
from qwen_integration import QwenIntegration
from config import Config
from utils import (
    setup_logging, load_image, resize_image, image_to_base64,
    create_download_link, format_confidence, create_history_entry,
    display_history, validate_image_file, create_metrics_display,
    performance_monitor
)

# Configuration de la page
st.set_page_config(
    page_title="Assistant Intelligent OCR + Qwen",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Configuration du logging
setup_logging()

# Initialisation des composants
@st.cache_resource
def initialize_components():
    """Initialise les composants principaux avec cache"""
    config = Config()
    ocr_pipeline = OCRPipeline(languages=config.OCR_LANGUAGES, gpu=config.OCR_GPU)
    qwen_integration = QwenIntegration()
    return config, ocr_pipeline, qwen_integration

def main():
    """Fonction principale de l'application"""
    
    # Titre principal
    st.title("📄 Assistant Intelligent de Traduction OCR + Qwen")
    st.markdown("*Extraction de texte multilingue avec traduction contextuelle et IA*")
    
    # Initialisation des composants
    config, ocr_pipeline, qwen_integration = initialize_components()
    
    # Initialisation du state
    if 'history' not in st.session_state:
        st.session_state.history = []
    if 'last_ocr_result' not in st.session_state:
        st.session_state.last_ocr_result = None
    if 'processing' not in st.session_state:
        st.session_state.processing = False
    if 'show_qa' not in st.session_state:
        st.session_state.show_qa = False
    
    # Sidebar - Configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Vérifier si l'API est configurée
        if not config.HUGGINGFACE_API_KEY and not config.OPENAI_API_KEY:
            st.warning("⚠️ API non configurée")
            with st.expander("📖 Comment configurer l'API", expanded=False):
                st.markdown("""
                **Pour activer les fonctionnalités IA complètes:**
                
                1. Créez un fichier `.env` à la racine du projet
                2. Ajoutez votre clé API:
                   ```
                   # Option 1: Hugging Face (recommandé, gratuit)
                   HUGGINGFACE_API_KEY=hf_votre_cle_ici
                   
                   # Option 2: OpenAI (payant)
                   OPENAI_API_KEY=sk-votre_cle_ici
                   ```
                3. Redémarrez l'application
                
                **Obtenir une clé Hugging Face (GRATUIT):**
                - https://huggingface.co/settings/tokens
                - Créez un nouveau token avec accès "read"
                
                **Sans API:** Les fonctionnalités basiques restent disponibles:
                - ✅ Extraction OCR fonctionne normalement
                - ⚠️ Traduction (via Google Translate)
                - ⚠️ Résumé basique
                - ⚠️ Q&A par mots-clés
                """)
        elif config.HUGGINGFACE_API_KEY:
            st.success("✅ API Hugging Face configurée")
        elif config.OPENAI_API_KEY:
            st.success("✅ API OpenAI configurée")
        
        st.divider()
        
        # Sélection des langues OCR
        st.subheader("🔤 Langues OCR")
        available_ocr_langs = {
            'fr': 'Français', 'en': 'Anglais', 'ch_sim': 'Chinois Simplifié',
            'ch_tra': 'Chinois Traditionnel', 'ar': 'Arabe', 'es': 'Espagnol',
            'de': 'Allemand', 'it': 'Italien', 'pt': 'Portugais',
            'ru': 'Russe', 'ja': 'Japonais', 'ko': 'Coréen'
        }
        
        selected_ocr_langs = st.multiselect(
            "Langues à détecter:",
            options=list(available_ocr_langs.keys()),
            default=['fr', 'en'],
            format_func=lambda x: available_ocr_langs[x]
        )
        
        # Avertissement pour les langues chinoises
        if 'ch_tra' in selected_ocr_langs or 'ch_sim' in selected_ocr_langs:
            st.warning("⚠️ Les langues chinoises nécessitent l'anglais et ne sont compatibles qu'avec lui dans EasyOCR. Les autres langues seront automatiquement retirées.")
        
        # Mise à jour des langues OCR
        if selected_ocr_langs != ocr_pipeline.languages:
            ocr_pipeline.update_languages(selected_ocr_langs)
        
        st.divider()
        
        # Configuration traduction
        st.subheader("🌍 Traduction")
        source_lang = st.selectbox(
            "Langue source:",
            options=list(config.TRANSLATION_LANGUAGES.keys()),
            format_func=lambda x: config.TRANSLATION_LANGUAGES[x],
            index=0
        )
        
        target_lang = st.selectbox(
            "Langue cible:",
            options=list(config.TRANSLATION_LANGUAGES.keys()),
            format_func=lambda x: config.TRANSLATION_LANGUAGES[x],
            index=1
        )
        
        st.divider()
        
        # Options avancées
        st.subheader("🔧 Options Avancées")
        preprocess_image = st.checkbox("Préprocessing image", value=True)
        auto_detect_lang = st.checkbox("Détection automatique de langue", value=True)
        enhance_ocr = st.checkbox("Amélioration OCR avec IA", value=False)
        
        # Seuil de confiance
        confidence_threshold = st.slider(
            "Seuil de confiance OCR:",
            min_value=0.0,
            max_value=1.0,
            value=0.5,
            step=0.1
        )
    
    # Interface principale
    tab1, tab2, tab3, tab4 = st.tabs(["📷 Upload Image", "🎥 Caméra Temps Réel", "📝 Résultats", "📜 Historique"])
    
    with tab1:
        st.header("📷 Upload et Traitement d'Images")
        
        # Upload de fichier
        uploaded_file = st.file_uploader(
            "Choisissez une image:",
            type=['jpg', 'jpeg', 'png', 'bmp', 'tiff', 'webp'],
            help="Formats supportés: JPG, PNG, BMP, TIFF, WebP (max 10MB)"
        )
        
        if uploaded_file and validate_image_file(uploaded_file):
            # Affichage de l'image
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.subheader("Image Originale")
                image = load_image(uploaded_file)
                
                if image is not None:
                    # Redimensionner pour affichage
                    display_image = resize_image(image, 400, 300)
                    display_image_rgb = cv2.cvtColor(display_image, cv2.COLOR_BGR2RGB)
                    st.image(display_image_rgb, use_column_width=True)
                    
                    # Bouton de traitement
                    if st.button("🚀 Extraire le Texte", type="primary", disabled=st.session_state.processing):
                        with st.spinner("Extraction en cours..."):
                            st.session_state.processing = True
                            
                            # Monitoring des performances
                            performance_monitor.start("OCR Extraction")
                            
                            # Extraction OCR
                            ocr_results = ocr_pipeline.extract_text_from_image(
                                image, preprocess=preprocess_image
                            )
                            
                            ocr_time = performance_monitor.end()
                            
                            if ocr_results['error']:
                                st.error(f"Erreur OCR: {ocr_results['error']}")
                            else:
                                st.session_state.last_ocr_result = ocr_results
                                st.session_state.last_image = image
                                
                                # Amélioration OCR si activée
                                if enhance_ocr and ocr_results['confidence'] < 0.8:
                                    performance_monitor.start("OCR Enhancement")
                                    enhanced_text = qwen_integration.enhance_ocr_text(
                                        ocr_results['text'], ocr_results['confidence']
                                    )
                                    performance_monitor.end()
                                    ocr_results['enhanced_text'] = enhanced_text
                                
                                st.success("✅ Extraction terminée!")
                                
                                # Affichage des métriques
                                create_metrics_display(ocr_results, ocr_time)
                            
                            st.session_state.processing = False
            
            with col2:
                st.subheader("Résultats de Détection")
                
                if st.session_state.last_ocr_result:
                    results = st.session_state.last_ocr_result
                    
                    if results['boxes'] and 'last_image' in st.session_state:
                        # Image avec boîtes de détection
                        image_with_boxes = ocr_pipeline.draw_boxes(
                            st.session_state.last_image, 
                            results['boxes']
                        )
                        image_with_boxes_rgb = cv2.cvtColor(image_with_boxes, cv2.COLOR_BGR2RGB)
                        st.image(image_with_boxes_rgb, use_column_width=True)
                        
                        # Informations de détection
                        st.write(f"**Zones détectées:** {len(results['boxes'])}")
                        st.write(f"**Confiance:** {format_confidence(results['confidence'])}", unsafe_allow_html=True)
    
    with tab2:
        st.header("🎥 Traitement Caméra Temps Réel")
        st.info("⚠️ Fonctionnalité en développement - Nécessite streamlit-webrtc pour la capture caméra")
        
        # Placeholder pour la caméra temps réel
        camera_placeholder = st.empty()
        
        col1, col2, col3 = st.columns(3)
        with col1:
            start_camera = st.button("🔹 Démarrer Caméra", disabled=True)
        with col2:
            capture_frame = st.button("📸 Capturer Frame", disabled=True)
        with col3:
            stop_camera = st.button("⏹️ Arrêter", disabled=True)
        
        st.markdown("""
        **Fonctionnalités prévues:**
        - Capture vidéo en temps réel
        - Extraction OCR sur frames sélectionnées
        - Superposition des résultats en direct
        - Enregistrement des captures importantes
        """)
    
    with tab3:
        st.header("📝 Analyse et Traduction")
        
        if st.session_state.last_ocr_result:
            results = st.session_state.last_ocr_result
            extracted_text = results.get('enhanced_text', results.get('text', ''))
            
            if extracted_text:
                # Affichage du texte extrait
                st.subheader("📄 Texte Extrait")
                
                col1, col2 = st.columns([3, 1])
                with col1:
                    text_area = st.text_area(
                        "Texte détecté (éditable):",
                        value=extracted_text,
                        height=150,
                        key="extracted_text"
                    )
                
                with col2:
                    st.write("**Statistiques:**")
                    st.write(f"Caractères: {len(text_area)}")
                    st.write(f"Mots: {len(text_area.split())}")
                    st.write(f"Lignes: {len(text_area.splitlines())}")
                
                # Détection automatique de langue
                if auto_detect_lang and text_area:
                    with st.spinner("Détection de langue..."):
                        detected_lang = qwen_integration.detect_language(text_area)
                        st.info(f"🌍 Langue détectée: {config.TRANSLATION_LANGUAGES.get(detected_lang, 'Inconnue')}")
                        if detected_lang != source_lang:
                            source_lang = detected_lang
                
                st.divider()
                
                # Actions IA
                st.subheader("🤖 Actions Intelligence Artificielle")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    if st.button("🌍 Traduire", type="primary"):
                        if text_area:
                            with st.spinner("Traduction en cours..."):
                                performance_monitor.start("Translation")
                                translation = qwen_integration.translate_text(
                                    text_area, source_lang, target_lang
                                )
                                trans_time = performance_monitor.end()
                                
                                st.subheader("📝 Traduction")
                                st.write(translation)
                                
                                # Ajouter à l'historique
                                history_entry = create_history_entry(
                                    text_area, translation, source_lang, target_lang
                                )
                                st.session_state.history.append(history_entry)
                                
                                # Lien de téléchargement
                                download_link = create_download_link(
                                    f"Original:\n{text_area}\n\nTraduction:\n{translation}",
                                    f"traduction_{int(time.time())}.txt"
                                )
                                st.markdown(download_link, unsafe_allow_html=True)
                                
                                # Métriques
                                st.info(f"⏱️ Temps de traduction: {trans_time:.2f}s")
                
                with col2:
                    if st.button("📋 Résumer"):
                        if text_area:
                            with st.spinner("Génération du résumé..."):
                                performance_monitor.start("Summarization")
                                summary = qwen_integration.summarize_text(text_area)
                                sum_time = performance_monitor.end()
                                
                                st.subheader("📋 Résumé")
                                st.write(summary)
                                
                                # Lien de téléchargement
                                download_link = create_download_link(
                                    f"Texte original:\n{text_area}\n\nRésumé:\n{summary}",
                                    f"resume_{int(time.time())}.txt"
                                )
                                st.markdown(download_link, unsafe_allow_html=True)
                                
                                # Métriques
                                st.info(f"⏱️ Temps de résumé: {sum_time:.2f}s")
                
                with col3:
                    if st.button("❓ Q&A"):
                        st.session_state.show_qa = not st.session_state.show_qa
                
                # Section Q&A séparée
                if st.session_state.show_qa:
                    st.divider()
                    st.subheader("❓ Questions & Réponses")
                    question = st.text_input("Posez une question sur le texte:", key="qa_question")
                    
                    if st.button("Répondre", key="qa_answer_btn") and question:
                        with st.spinner("Génération de la réponse..."):
                            performance_monitor.start("Question Answering")
                            answer = qwen_integration.answer_question(text_area, question)
                            qa_time = performance_monitor.end()
                            
                            st.write(f"**Q:** {question}")
                            st.write(f"**R:** {answer}")
                            st.info(f"⏱️ Temps de réponse: {qa_time:.2f}s")
                
            else:
                st.info("Aucun texte extrait. Veuillez d'abord traiter une image.")
        else:
            st.info("Aucune image traitée. Veuillez d'abord extraire du texte d'une image.")
    
    with tab4:
        st.header("📜 Historique des Traductions")
        
        if st.session_state.history:
            # Options d'historique
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if st.button("🗑️ Vider l'historique"):
                    st.session_state.history = []
                    st.success("Historique vidé!")
                    st.rerun()
            
            with col2:
                if st.button("📥 Exporter l'historique"):
                    history_text = "\n\n".join([
                        f"[{entry['timestamp']}] {entry['source_lang']} → {entry['target_lang']}\n"
                        f"Original: {entry['original_text']}\n"
                        f"Traduction: {entry['translation']}"
                        for entry in st.session_state.history
                    ])
                    
                    download_link = create_download_link(
                        history_text,
                        f"historique_traductions_{int(time.time())}.txt"
                    )
                    st.markdown(download_link, unsafe_allow_html=True)
            
            with col3:
                st.write(f"**Total:** {len(st.session_state.history)} entrées")
            
            st.divider()
            
            # Affichage de l'historique
            display_history(st.session_state.history)
        else:
            st.info("Aucun historique disponible. Effectuez des traductions pour voir l'historique.")
    
    # Footer
    st.divider()
    st.markdown("""
    <div style='text-align: center; color: #666; padding: 20px;'>
        📄 Assistant Intelligent OCR + Qwen | 
        Développé avec Streamlit, EasyOCR, OpenCV et Qwen LLM
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()