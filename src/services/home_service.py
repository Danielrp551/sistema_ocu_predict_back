from ..models.history_model import PredictionHistory
from ..extensions import db
from sqlalchemy import func
import datetime

def get_home_stats_service(user_id, date_range):
    """
    Retorna un diccionario con métricas para el dashboard:
      - totalRequests: cuántas solicitudes se hicieron en ese rango (en general)
      - userRequests: cuántas solicitudes hizo el user en ese rango
      - correctPct: porcentaje de feedback "correcto" en ese rango
      - otras métricas que necesites
    """

    now = datetime.datetime.now()
    
    if date_range == "hoy":
        # Filtrar desde hoy a las 00:00
        start_date = now.replace(hour=0, minute=0, second=0, microsecond=0)
    elif date_range == "semana":
        # Ultimos 7 dias
        start_date = now - datetime.timedelta(days=7)
    elif date_range == "mes":
        # Ultimos 30 dias (simple approximation)
        start_date = now - datetime.timedelta(days=30)
    else:
        # fallback: hoy
        start_date = now.replace(hour=0, minute=0, second=0, microsecond=0)

    # Query base: filtrar por created_at >= start_date
    total_requests = (db.session.query(func.count(PredictionHistory.id))
                                .filter(PredictionHistory.created_at >= start_date)
                                .scalar())
    
    # Cuántas hizo el user
    user_requests = (db.session.query(func.count(PredictionHistory.id))
                               .filter(PredictionHistory.created_at >= start_date)
                               .filter(PredictionHistory.user_id == user_id)
                               .scalar())

    # Porcentaje de correctas => count(doctor_feedback="correcto") / count(total)
    # Filtramos solo las que tienen feedback
    total_with_feedback = (db.session.query(func.count(PredictionHistory.id))
                                      .filter(PredictionHistory.created_at >= start_date)
                                      .filter(PredictionHistory.doctor_feedback.isnot(None))
                                      .scalar())
    
    correct_count = (db.session.query(func.count(PredictionHistory.id))
                                .filter(PredictionHistory.created_at >= start_date)
                                .filter(PredictionHistory.doctor_feedback == "correcto")
                                .scalar())

    if total_with_feedback > 0:
        correct_pct = round((correct_count / total_with_feedback) * 100, 2)
    else:
        correct_pct = 0.0

    # Podrías armar data para un gráfico, p.ej. "chartData" ...
    # Ejemplo simple: feedback correcto vs incorrecto
    # ...
    incorrect_count = total_with_feedback - correct_count
    chart_data = {
        "labels": ["Correcto", "Incorrecto"],
        "datasets": [
            {
                "data": [correct_count, incorrect_count],
                "backgroundColor": ["#4caf50", "#f44336"]
            }
        ]
    }

    # Retornamos todo en un diccionario
    return {
        "totalRequests": total_requests,
        "userRequests": user_requests,
        "correctPct": correct_pct,
        "chartData": chart_data
    }
