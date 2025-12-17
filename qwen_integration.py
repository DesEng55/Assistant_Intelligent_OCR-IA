import streamlit as st
import requests
import json
from typing import Dict, List, Optional
import logging
from config import Config
import openai
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

class QwenIntegration:
    def __init__(self):
        """Initialise l'intégration Qwen LLM"""
        self.config = Config()
        self.model = None
        self.tokenizer = None
        self.client = None
        self.hf_api_key = None
        self.mode = self._initialize_model()
    
    def _initialize_model(self) -> Optional[str]:
        """Initialise le modèle Qwen"""
        try:
            # Priorité 1: Hugging Face API (recommandé)
            if self.config.HUGGINGFACE_API_KEY:
                self.hf_api_key = self.config.HUGGINGFACE_API_KEY
                logging.info("Initialisation avec Hugging Face API")
                return "huggingface_api"
            
            # Priorité 2: OpenAI API compatible
            elif self.config.OPENAI_API_KEY:
                self.client = openai.OpenAI(api_key=self.config.OPENAI_API_KEY)
                logging.info("Initialisation avec OpenAI API")
                return "openai_api"
            
            # Priorité 3: Modèle local (nécessite beaucoup de ressources)
            else:
                logging.info("Tentative de chargement du modèle local...")
                logging.warning("⚠️ Le chargement local nécessite beaucoup de RAM (>16GB)")
                
                self.tokenizer = AutoTokenizer.from_pretrained(
                    self.config.QWEN_MODEL,
                    trust_remote_code=True
                )
                
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.config.QWEN_MODEL,
                    device_map="auto",
                    trust_remote_code=True,
                    torch_dtype=torch.float16
                )
                
                logging.info("Modèle local Qwen chargé avec succès")
                return "local"
            
        except Exception as e:
            logging.error(f"Erreur lors du chargement du modèle: {str(e)}")
            return None
    
    def generate_response(self, prompt: str, max_tokens: int = 500, temperature: float = 0.7) -> str:
        """
        Génère une réponse avec Qwen
        
        Args:
            prompt: Prompt d'entrée
            max_tokens: Nombre maximum de tokens
            temperature: Température de génération
            
        Returns:
            Réponse générée
        """
        try:
            # Méthode 1: Hugging Face Inference API
            if self.hf_api_key:
                # Try using a simpler, faster model first
                api_url = "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.2"
                headers = {
                    "Authorization": f"Bearer {self.hf_api_key}",
                    "Content-Type": "application/json"
                }
                
                # Format prompt for instruction-following
                formatted_prompt = f"[INST] {prompt} [/INST]"
                
                payload = {
                    "inputs": formatted_prompt,
                    "parameters": {
                        "max_new_tokens": max_tokens,
                        "temperature": temperature,
                        "return_full_text": False,
                        "do_sample": True,
                        "top_p": 0.9
                    },
                    "options": {
                        "wait_for_model": True
                    }
                }
                
                response = requests.post(api_url, headers=headers, json=payload, timeout=120)
                
                if response.status_code == 200:
                    result = response.json()
                    if isinstance(result, list) and len(result) > 0:
                        generated = result[0].get("generated_text", "")
                        return generated.strip()
                    elif isinstance(result, dict):
                        return result.get("generated_text", str(result)).strip()
                    return str(result).strip()
                    
                elif response.status_code == 503:
                    # Model is loading - wait and retry
                    logging.info("Model is loading, waiting 20 seconds...")
                    import time
                    time.sleep(20)
                    
                    # Retry once
                    response = requests.post(api_url, headers=headers, json=payload, timeout=120)
                    if response.status_code == 200:
                        result = response.json()
                        if isinstance(result, list) and len(result) > 0:
                            return result[0].get("generated_text", "").strip()
                    
                    logging.warning("Model still loading, falling back to basic mode")
                    return None
                    
                else:
                    logging.error(f"Erreur API HuggingFace: {response.status_code} - {response.text}")
                    return None
            
            # Méthode 2: OpenAI API
            elif self.client:
                response = self.client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=max_tokens,
                    temperature=temperature
                )
                return response.choices[0].message.content
            
            # Méthode 3: Modèle local
            elif self.model and self.tokenizer:
                inputs = self.tokenizer(prompt, return_tensors="pt")
                
                with torch.no_grad():
                    outputs = self.model.generate(
                        inputs.input_ids,
                        max_new_tokens=max_tokens,
                        temperature=temperature,
                        do_sample=True,
                        pad_token_id=self.tokenizer.eos_token_id
                    )
                
                response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                return response[len(prompt):].strip()
            
            else:
                return None
                
        except requests.exceptions.Timeout:
            logging.warning("Request timeout - model may be loading or busy")
            return None
        except Exception as e:
            logging.error(f"Erreur génération: {str(e)}")
            import traceback
            traceback.print_exc()
            return None
    
    def translate_text(self, text: str, source_lang: str, target_lang: str) -> str:
        """
        Traduit un texte de manière contextuelle
        
        Args:
            text: Texte à traduire
            source_lang: Langue source
            target_lang: Langue cible
            
        Returns:
            Texte traduit
        """
        if not text.strip():
            return ""
        
        # Essayer avec le modèle AI
        prompt = self.config.TRANSLATION_PROMPT.format(
            text=text,
            source_lang=source_lang,
            target_lang=target_lang
        )
        
        result = self.generate_response(prompt, max_tokens=300, temperature=0.3)
        
        if result is None:
            # Fallback: utiliser une traduction basique avec deep_translator
            try:
                from deep_translator import GoogleTranslator
                translator = GoogleTranslator(source='auto', target=target_lang)
                return translator.translate(text)
            except:
                return f"⚠️ Traduction non disponible. Veuillez configurer une clé API OpenAI dans le fichier .env\n\nTexte original: {text}"
        
        return result
    
    def summarize_text(self, text: str) -> str:
        """
        Génère un résumé du texte
        
        Args:
            text: Texte à résumer
            
        Returns:
            Résumé
        """
        if not text.strip():
            return ""
        
        prompt = self.config.SUMMARY_PROMPT.format(text=text)
        result = self.generate_response(prompt, max_tokens=200, temperature=0.5)
        
        if result is None:
            # Fallback: créer un résumé basique
            sentences = text.split('.')
            summary = '. '.join(sentences[:3])
            return f"⚠️ Résumé automatique (basique):\n\n{summary}...\n\n💡 Pour un meilleur résumé, configurez une clé API OpenAI dans le fichier .env"
        
        return result
    
    def answer_question(self, text: str, question: str) -> str:
        """
        Répond à une question basée sur le texte extrait
        
        Args:
            text: Texte de référence
            question: Question de l'utilisateur
            
        Returns:
            Réponse
        """
        if not text.strip() or not question.strip():
            return ""
        
        prompt = self.config.QA_PROMPT.format(text=text, question=question)
        result = self.generate_response(prompt, max_tokens=300, temperature=0.4)
        
        if result is None:
            # Fallback: recherche simple de mots-clés
            question_lower = question.lower()
            text_lower = text.lower()
            
            # Trouver la phrase la plus pertinente
            sentences = text.split('.')
            relevant_sentences = []
            
            for sentence in sentences:
                # Compter les mots de la question présents dans la phrase
                words = question_lower.split()
                matches = sum(1 for word in words if len(word) > 3 and word in sentence.lower())
                if matches > 0:
                    relevant_sentences.append((sentence, matches))
            
            if relevant_sentences:
                # Trier par pertinence
                relevant_sentences.sort(key=lambda x: x[1], reverse=True)
                best_sentence = relevant_sentences[0][0]
                return f"⚠️ Réponse basique (recherche de mots-clés):\n\n{best_sentence}\n\n💡 Pour des réponses plus intelligentes, configurez une clé API OpenAI dans le fichier .env"
            else:
                return "⚠️ Aucune réponse trouvée dans le texte.\n\n💡 Pour des réponses plus intelligentes, configurez une clé API OpenAI dans le fichier .env"
        
        return result
    
    def detect_language(self, text: str) -> str:
        """
        Détecte la langue du texte
        
        Args:
            text: Texte à analyser
            
        Returns:
            Code de langue détecté
        """
        if not text.strip():
            return "unknown"
        
        # Essayer avec le modèle AI d'abord
        prompt = f"""
        Détectez la langue du texte suivant et répondez uniquement par le code de langue (fr, en, es, de, it, pt, ru, zh, ar, ja, ko):
        
        Texte: {text[:200]}...
        
        Langue:
        """
        
        response = self.generate_response(prompt, max_tokens=10, temperature=0.1)
        
        if response is None:
            # Fallback: détection basique par mots-clés
            try:
                from langdetect import detect
                detected = detect(text)
                # Mapper les codes ISO vers nos codes
                lang_map = {
                    'fr': 'fr', 'en': 'en', 'es': 'es', 'de': 'de', 
                    'it': 'it', 'pt': 'pt', 'ru': 'ru', 'zh-cn': 'zh',
                    'zh-tw': 'zh', 'ar': 'ar', 'ja': 'ja', 'ko': 'ko'
                }
                return lang_map.get(detected, 'fr')
            except:
                # Détection ultra-basique par caractères
                if any('\u4e00' <= c <= '\u9fff' for c in text):
                    return 'zh'
                elif any('\u3040' <= c <= '\u309f' or '\u30a0' <= c <= '\u30ff' for c in text):
                    return 'ja'
                elif any('\uac00' <= c <= '\ud7af' for c in text):
                    return 'ko'
                elif any('\u0600' <= c <= '\u06ff' for c in text):
                    return 'ar'
                elif any('\u0400' <= c <= '\u04ff' for c in text):
                    return 'ru'
                else:
                    return 'fr'  # Défaut
        
        # Nettoyer la réponse
        detected_lang = response.strip().lower()
        
        # Vérifier si c'est un code de langue valide
        valid_langs = list(self.config.TRANSLATION_LANGUAGES.keys())
        if detected_lang in valid_langs:
            return detected_lang
        
        # Essayer de mapper des réponses communes
        lang_mapping = {
            'french': 'fr', 'français': 'fr',
            'english': 'en', 'anglais': 'en',
            'spanish': 'es', 'espagnol': 'es',
            'german': 'de', 'allemand': 'de',
            'italian': 'it', 'italien': 'it',
            'portuguese': 'pt', 'portugais': 'pt',
            'russian': 'ru', 'russe': 'ru',
            'chinese': 'zh', 'chinois': 'zh',
            'arabic': 'ar', 'arabe': 'ar',
            'japanese': 'ja', 'japonais': 'ja',
            'korean': 'ko', 'coréen': 'ko'
        }
        
        for key, value in lang_mapping.items():
            if key in detected_lang:
                return value
        
        return "fr"  # Défaut français
    
    def enhance_ocr_text(self, text: str, confidence: float) -> str:
        """
        Améliore le texte OCR en corrigeant les erreurs potentielles
        
        Args:
            text: Texte OCR brut
            confidence: Niveau de confiance OCR
            
        Returns:
            Texte amélioré
        """
        if not text.strip() or confidence > 0.9:
            return text
        
        prompt = f"""
        Le texte suivant a été extrait par OCR avec un niveau de confiance de {confidence:.2f}. 
        Corrigez les erreurs potentielles tout en préservant le sens original:
        
        Texte OCR: {text}
        
        Texte corrigé:
        """
        
        return self.generate_response(prompt, max_tokens=len(text) + 100, temperature=0.2)