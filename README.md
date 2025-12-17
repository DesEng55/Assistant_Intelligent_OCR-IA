# 📄 Assistant Intelligent OCR + IA

Application de reconnaissance de texte multilingue avec traduction et analyse par IA.

## ✨ Fonctionnalités

- 🔍 **OCR Multilingue** - 12+ langues (Français, Anglais, Espagnol, Chinois, Arabe, etc.)
- 🌍 **Traduction Intelligente** - Powered by Mistral-7B AI
- 📋 **Résumés Automatiques** - Génération de résumés concis
- ❓ **Q&A** - Questions-réponses sur le texte extrait
- 📊 **Historique** - Sauvegarde et export des traductions

---

## 🚀 Installation Rapide

### 1. Installer les dépendances
```bash
pip install -r requirements.txt
```

### 2. Configurer l'API (Optionnel - Recommandé)

**Obtenez une clé Hugging Face GRATUITE :**
1. Créez un compte sur https://huggingface.co/join
2. Allez sur https://huggingface.co/settings/tokens
3. Créez un nouveau token (type: **Read**)
4. Copiez la clé (commence par `hf_`)

**Créez un fichier `.env` :**
```env
HUGGINGFACE_API_KEY=hf_votre_cle_ici
```

### 3. Lancer l'application
```bash
streamlit run app.py
```

Ouvrez http://localhost:8501 dans votre navigateur.

---

## 📖 Utilisation

1. **Sélectionnez les langues** OCR dans la barre latérale
2. **Uploadez une image** (JPG, PNG, max 10MB)
3. **Cliquez "Extraire le Texte"**
4. Utilisez les boutons **Traduire**, **Résumer** ou **Q&A**

---

## 🐛 Dépannage

### ❌ "API key not found"
- Vérifiez que `.env` existe dans le dossier racine
- Vérifiez l'orthographe : `HUGGINGFACE_API_KEY=hf_...`
- Redémarrez l'application

### ⏰ "Model is loading"
- Normal à la première utilisation (30-60 secondes)
- Réessayez après 1 minute
- Les requêtes suivantes seront rapides

### 🔍 Diagnostic
```bash
python diagnostic.py
```

### ❌ "Chinese_tra is only compatible with English"
- Sélectionnez **uniquement** Anglais + Chinois Traditionnel

---

## ⚙️ Configuration

### Langues OCR par défaut
Éditez `config.py`:
```python
OCR_LANGUAGES = ['fr', 'en']
```

### GPU/CPU
```python
OCR_GPU = True  # False pour CPU uniquement
```

---

## 📁 Structure du Projet

```
assistant-ocr/
├── app.py                  # Application Streamlit
├── config.py              # Configuration
├── ocr_pipeline.py        # Pipeline OCR
├── qwen_integration.py    # Intégration IA
├── utils.py               # Utilitaires
├── diagnostic.py          # Script de diagnostic
├── requirements.txt       # Dépendances
├── .env                   # Clés API (à créer)
└── README.md             # Ce fichier
```

---

## 🛠️ Technologies

- **Streamlit** - Interface web
- **EasyOCR** - Reconnaissance de caractères
- **OpenCV** - Traitement d'images
- **Mistral-7B** - Modèle d'IA
- **Hugging Face API** - Inférence cloud

---
