from PIL import Image
import torch
from transformers import BlipProcessor
from transformers import BlipForQuestionAnswering
import streamlit as st
def predict_vqa(image_path: str,
                question: str,
                model: BlipForQuestionAnswering,
                processor: BlipProcessor,
                device: torch.device) -> str:
    """
    Prédit la réponse à une question sur une image avec un modèle BLIP-VQA.

    Args:
        image_path (str): chemin vers le fichier image.
        question   (str): texte de la question.
        model      (BlipForQuestionAnswering): modèle finetuné.
        processor  (BlipProcessor): processor associé.
        device     (torch.device): 'cuda' ou 'cpu'.

    Returns:
        str: réponse générée par le modèle.
    """
    # Chargement et prétraitement
    image = Image.open(image_path).convert("RGB")
    inputs = processor(image,
                       question,
                       return_tensors="pt").to(device)

    # Génération
    with torch.no_grad():
        generated_ids = model.generate(**inputs,
                                       max_length=16,
                                       num_beams=5,
                                       early_stopping=True)
    # Décodage
    answer = processor.batch_decode(generated_ids,
                                    skip_special_tokens=True)[0]
    return answer
processor = BlipProcessor.from_pretrained("Salesforce/blip-vqa-base")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model=torch.load("vqa_model.pt",weights_only=False)
# -------------------------------
# 2. Interface utilisateur
# -------------------------------
st.title("Visual Question Answering - BLIP")

image_file = st.file_uploader("📷 Téléverse une image", type=["png", "jpg", "jpeg"])
question = st.text_input("❓ Pose ta question sur l'image")

# -------------------------------
# 3. Prédiction
# -------------------------------
if image_file and question:
    image = Image.open(image_file).convert("RGB")
    st.image(image, caption="Image chargée", use_column_width=True)

    with st.spinner("💬 Génération de la réponse..."):
        inputs = processor(image, question, return_tensors="pt").to(device)

        with torch.no_grad():
            generated_ids = model.generate(**inputs, max_length=16)
            answer = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

        st.success(f"🧠 Réponse : **{answer}**")