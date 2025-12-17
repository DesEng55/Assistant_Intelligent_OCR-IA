import os
from dotenv import load_dotenv

# Charger les variables d'environnement
load_dotenv()

class Config:
    # Configuration OCR - Langues par défaut compatibles
    # Note: ch_tra (Chinois Traditionnel) nécessite 'en' et n'est compatible qu'avec l'anglais
    OCR_LANGUAGES = ['fr', 'en']
    OCR_GPU = True
    
    # Configuration Qwen/LLM
    QWEN_MODEL = "mistralai/Mistral-7B-Instruct-v0.2"  # Changed to a more reliable model
    HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY", "")
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
    
    # Configuration Interface
    SUPPORTED_IMAGE_FORMATS = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp']
    MAX_IMAGE_SIZE = 10 * 1024 * 1024  # 10MB
    
    # Configuration Caméra
    CAMERA_WIDTH = 640
    CAMERA_HEIGHT = 480
    CAMERA_FPS = 30
    
    # Configuration Performance
    CACHE_TTL = 3600  # 1 heure
    MAX_CONCURRENT_REQUESTS = 5
    
    # Langues supportées pour traduction
    TRANSLATION_LANGUAGES = {
        'fr': 'Français',
        'en': 'English',
        'es': 'Español',
        'de': 'Deutsch',
        'it': 'Italiano',
        'pt': 'Português',
        'ru': 'Русский',
        'zh': '中文',
        'ar': 'العربية',
        'ja': '日本語',
        'ko': '한국어'
    }
    
    # Langues OCR disponibles avec leurs restrictions
    OCR_LANGUAGE_INFO = {
        'fr': {'name': 'Français', 'compatible_with': 'all'},
        'en': {'name': 'Anglais', 'compatible_with': 'all'},
        'ch_sim': {'name': 'Chinois Simplifié', 'compatible_with': ['en']},
        'ch_tra': {'name': 'Chinois Traditionnel', 'compatible_with': ['en']},
        'ar': {'name': 'Arabe', 'compatible_with': 'all'},
        'es': {'name': 'Espagnol', 'compatible_with': 'all'},
        'de': {'name': 'Allemand', 'compatible_with': 'all'},
        'it': {'name': 'Italien', 'compatible_with': 'all'},
        'pt': {'name': 'Portugais', 'compatible_with': 'all'},
        'ru': {'name': 'Russe', 'compatible_with': 'all'},
        'ja': {'name': 'Japonais', 'compatible_with': 'all'},
        'ko': {'name': 'Coréen', 'compatible_with': 'all'}
    }
    
    # Prompts pour Qwen
    TRANSLATION_PROMPT = """Vous êtes un assistant de traduction intelligent. Traduisez le texte suivant de manière contextuelle et naturelle.

Texte à traduire: {text}
Langue source: {source_lang}
Langue cible: {target_lang}

Fournissez une traduction précise et naturelle:"""
    
    SUMMARY_PROMPT = """Analysez le texte suivant et fournissez un résumé concis et informatif.

Texte: {text}

Résumé (maximum 3 phrases):"""
    
    QA_PROMPT = """Basé sur le texte extrait suivant, répondez à la question de l'utilisateur de manière précise et contextuelle.

Texte de référence: {text}
Question: {question}

Réponse:"""