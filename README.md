# 🧠 Visual Question Answering (VQA) avec BLIP sur VizWiz

Ce projet montre comment fine-tuner le modèle pré-entraîné **BLIP-VQA-base** sur le jeu de données **VizWiz-VQA**, afin de permettre à un système de répondre à des questions en langage naturel posées sur des images prises dans des conditions réelles (par des personnes malvoyantes).

---

## 🖼️ Schéma d’architecture

![BLIP-VQA pipeline](./A_diagram_titled_"BLIP_Fine-Tuning_for_VQA_on_VizW.png)

---

## 🧰 Stack utilisée

| Composant        | Rôle |
|------------------|------|
| `BLIP-VQA-base`  | Modèle multimodal vision + texte (ViT + BERT) |
| `VizWiz`         | Dataset VQA d’images réelles avec annotations humaines |
| `PyTorch`        | Framework de deep learning |
| `Transformers`   | Chargement du modèle BLIP via 🤗 Hugging Face |
| `Streamlit`      | Interface web pour tester en ligne |
| `Google Colab`   | Environnement d’entraînement (12 Go de VRAM) |

---

## 🗂️ Pipeline de traitement

1. **Extraction** : chargement manuel des images et des JSON contenant les questions / réponses.
2. **Prétraitement à la volée** : `BlipProcessor` transforme les images (224×224 RGB) et tokenize les questions.
3. **Fine-tuning adaptatif** :
   - le **Vision Encoder (ViT)** est gelé au début,
   - seuls l’encodeur texte, la fusion et la tête générative sont entraînés,
   - si la perte stagne, le ViT est progressivement dégelé.
4. **Évaluation** : prédiction par génération token-par-token + calcul d’accuracy sur les réponses humaines.
5. **Déploiement** : interface interactive avec Streamlit.

---

## ▶️ Démo Streamlit (locale)

```bash
streamlit run app.py
