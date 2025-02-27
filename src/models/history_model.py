from ..extensions import db
import datetime

class PredictionHistory(db.Model):
    __tablename__ = "prediction_history"

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=False)
    image_filename = db.Column(db.String(200), nullable=False)
    image_s3_key = db.Column(db.String(300), nullable=True)
    predicted_class_name = db.Column(db.String(100), nullable=False)
    probabilities = db.Column(db.String(500), nullable=False)  
    doctor_feedback = db.Column(db.String(50), nullable=True)  
    created_at = db.Column(db.DateTime, default=datetime.datetime.utcnow)
