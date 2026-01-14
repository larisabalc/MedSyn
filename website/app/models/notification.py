from app.extensions import db
from datetime import datetime

class Notification(db.Model):
    __tablename__ = "notification"

    notification_id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("users.user_id"), nullable=False)
    title = db.Column(db.String(150), nullable=False)
    message = db.Column(db.Text, nullable=False)
    is_read = db.Column(db.Boolean, default=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    send_time = db.Column(db.DateTime, nullable=True)
    appointment_id = db.Column(db.Integer, db.ForeignKey("appointment.appointment_id"), nullable=True)

    user = db.relationship("User", back_populates="notifications")
    appointment = db.relationship("Appointment", back_populates="notifications")

    def __repr__(self):
        return f"<Notification {self.notification_id} to User {self.user_id}: {self.title}>"
