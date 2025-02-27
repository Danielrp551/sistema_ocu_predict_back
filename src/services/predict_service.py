import os, io
import torch
from PIL import Image
from torchvision import transforms
from ..extensions import db
from ..models.eye_disease_model import EyeDiseaseClassifier, id2label
from ..services.aws_service import upload_file_to_s3
from ..models.history_model import PredictionHistory

# Carga modelo
#model_path = "src/modelosDeep/modelo_resnet18_v3_patience_15.pth"  # Ajusta la ruta si difiere
#num_classes = 4
#modelo_inferencia = EyeDiseaseClassifier(num_classes=num_classes)
#modelo_inferencia.load_state_dict(torch.load(model_path, map_location="cpu"))
#modelo_inferencia.eval()

preprocesamiento = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

def load_model(model_path, num_classes=4):
    modelo = EyeDiseaseClassifier(num_classes=num_classes)
    modelo.load_state_dict(torch.load(model_path, map_location="cpu"))
    modelo.eval()
    return modelo

modelo_inferencia = None

def initialize_model():
    global modelo_inferencia
    model_path = "src/modelosDeep/modelo_resnet18_patience_5_drpt_02.pth"
    modelo_inferencia = load_model(model_path)

def predict_disease_service(file, user_id):
    global modelo_inferencia
    if modelo_inferencia is None:
        raise Exception("El modelo no ha sido inicializado. Asegúrese de descargarlo primero.")
    try:
        original_filename = file.filename
        file_bytes = file.read()

        # 1. Subir a S3
        try:
            s3_key = upload_file_to_s3(file_bytes, original_filename)
        except Exception as e:
            print("Error al subir a S3:", e)
            raise Exception("Error al subir la imagen a almacenamiento externo.")

        # 2. Procesar la imagen en memoria (para el modelo)
        try:
            image = Image.open(io.BytesIO(file_bytes)).convert('RGB')
            image_tensor = preprocesamiento(image).unsqueeze(0)
        except Exception as e:
            print("Error al procesar la imagen:", e)
            raise Exception("Error al procesar la imagen.")

        # 3. Hacer inferencia
        with torch.no_grad():
            output = modelo_inferencia(image_tensor)
            pred_class = torch.argmax(output, dim=1).item()
            probabilities = torch.softmax(output, dim=1).cpu().numpy()[0].tolist()

        class_name = id2label[pred_class]

        # 4. Guardar en historial
        history_record = PredictionHistory(
            user_id=user_id,
            image_filename=original_filename,
            image_s3_key=s3_key,
            predicted_class_name=class_name,
            probabilities=str(probabilities),
            doctor_feedback=None
        )
        db.session.add(history_record)
        db.session.commit()

        return {
            "predicted_class_index": pred_class,
            "predicted_class_name": class_name,
            "probabilities": probabilities,
            "s3_key": s3_key,
            "history_id": history_record.id
        }
    except Exception as ex:
        print("Error en predict_disease_service:", ex)
        raise ex 
