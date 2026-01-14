from flask import Flask, session
from .config import Config
from .extensions import db
from app.models.user import User
from app.models.patient import Patient
from app.models.doctor import Doctor
from app.models.specialization import Specialization
from app.models.service import Service
from app.models.availability import Availability
from app.models.appointment import Appointment
from app.models.notification import Notification
from datetime import datetime

def create_app():
    app = Flask(__name__)
    app.config.from_object(Config)

    db.init_app(app)

    with app.app_context():
        db.create_all()

    from .controllers.auth_controller import auth_bp
    from .controllers.patient_controller import patient_bp
    from .controllers.doctor_controller import doctor_bp

    app.register_blueprint(auth_bp)
    app.register_blueprint(patient_bp)
    app.register_blueprint(doctor_bp)

    @app.context_processor
    def inject_unseen_notifications():
        user_id = session.get("user_id")
        if not user_id:
            return {"unseen_notifications": 0}

        now = datetime.utcnow()
        unseen_count = Notification.query.filter(
            Notification.user_id == user_id,
            Notification.is_read == False,
            (Notification.send_time == None) | (Notification.send_time <= now)
        ).count()

        return {"unseen_notifications": unseen_count}

    return app
