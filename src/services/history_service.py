from ..models.history_model import PredictionHistory
from ..services.aws_service import get_file_as_base64
from ..extensions import db
from sqlalchemy import desc, asc
import base64
import os

def get_user_history_service(user_id, queryObject):
    # Parsear parametros
    image_filename = queryObject.get("image_filename", None)
    predited_class_name = queryObject.get("predicted_class_name", None)
    doctor_feedback = queryObject.get("doctor_feedback", None)
    id = queryObject.get("id", None) 
    #
    sort_by = queryObject.get("SortBy", None)
    is_desc_str = queryObject.get("IsDescending", "false").lower()
    is_desc = (is_desc_str == "true")    
    page_number = int(queryObject.get("PageNumber", 1))
    page_size = int(queryObject.get("PageSize", 10))

    # Construir query base
    query = PredictionHistory.query.filter_by(user_id=user_id)

    if image_filename:
        query = query.filter(PredictionHistory.image_filename.ilike(f"%{image_filename}%"))
    if predited_class_name:
        query = query.filter(PredictionHistory.predicted_class_name.ilike(f"%{predited_class_name}%"))
    if doctor_feedback:
        if doctor_feedback == "Sin feedback":
            query = query.filter(PredictionHistory.doctor_feedback == None)
        else:
            query = query.filter(PredictionHistory.doctor_feedback == doctor_feedback)
    if id:
        query = query.filter(PredictionHistory.id == id)
    # Ordenar
    if sort_by == "created_at":
        if is_desc:
            query = query.order_by(desc(PredictionHistory.created_at))
        else:
            query = query.order_by(asc(PredictionHistory.created_at))
    elif sort_by == "predicted_class_name":
        if is_desc:
            query = query.order_by(desc(PredictionHistory.predicted_class_name))
        else:
            query = query.order_by(asc(PredictionHistory.predicted_class_name))
    elif sort_by == "doctor_feedback":
        if is_desc:
            query = query.order_by(desc(PredictionHistory.doctor_feedback))
        else:
            query = query.order_by(asc(PredictionHistory.doctor_feedback))
    elif sort_by == "image_filename":
        if is_desc:
            query = query.order_by(desc(PredictionHistory.image_filename))
        else:
            query = query.order_by(asc(PredictionHistory.image_filename))
    elif sort_by == "id":
        if is_desc:
            query = query.order_by(desc(PredictionHistory.id))
        else:
            query = query.order_by(asc(PredictionHistory.id))
    else:
        # Si no hay sort by, por defecto la fecha de creacion descendente
        query = query.order_by(desc(PredictionHistory.created_at))

    # Paginacion
    if page_number < 1:
        page_number = 1
    if page_size < 1:
        page_size = 1

    pagination = query.paginate(page=page_number, per_page=page_size, error_out=False)

    records = pagination.items  # registros de la página actual
    total = pagination.total    # total de registros

    # Convertir a diccionario para retornar en JSON
    history_data = []
    for r in records:
        history_data.append({
            "id": r.id,
            "image_filename": r.image_filename,
            "predicted_class_name": r.predicted_class_name,
            "probabilities": r.probabilities,
            "doctor_feedback": r.doctor_feedback,
            "created_at": str(r.created_at)
        })
    return {
        "items": history_data,
        "total": total,
        "pageNumber": pagination.page,
        "pageSize": pagination.per_page,
        "hasNext": pagination.has_next,
        "hasPrev": pagination.has_prev
    }

# Podrías tener otra función para que el médico actualice el feedback
def update_doctor_feedback_service(history_id, feedback):
    record = PredictionHistory.query.get(history_id)
    if not record:
        raise Exception("Registro no encontrado")
    record.doctor_feedback = feedback
    db.session.commit()


def get_history_by_id_with_image_service(user_id, history_id):
    """
    Retorna un dict con la info del registro (id, feedback, etc.)
    e incluye la imagen en base64 (image_base64) si existe el archivo.
    """
    record = PredictionHistory.query.filter_by(user_id=user_id, id=history_id).first()
    if not record:
        raise Exception("No se encontró la solicitud o no tienes acceso a ella.")

    # Armamos el dict con la info principal
    record_dict = {
        "id": record.id,
        "user_id": record.user_id,
        "image_filename": record.image_filename,
        "predicted_class_name": record.predicted_class_name,
        "probabilities": record.probabilities,
        "doctor_feedback": record.doctor_feedback,
        "created_at": record.created_at.isoformat() if record.created_at else None,
        "image_s3_key": record.image_s3_key,
        "image_s3_url": None,
        "image_base64": None
    }

    if record.image_s3_key:
        try:
            record_dict["image_base64"] = get_file_as_base64(record.image_s3_key, mime_type="image/jpeg")
        except Exception as e:
            print("Error en get_history_by_id_with_image_service:", e)
            record_dict["image_base64"] = None

    return record_dict