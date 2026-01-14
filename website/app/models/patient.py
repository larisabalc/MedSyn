from app.extensions import db

class Patient(db.Model):
    __tablename__ = "patient"

    patient_id = db.Column(db.Integer, db.ForeignKey("users.user_id"), primary_key=True)
    insurance_no = db.Column(db.String(100), nullable=True)
    emergency_contact = db.Column(db.String(100), nullable=True)

    user = db.relationship("User", back_populates="patient_profile", uselist=False)
    appointments = db.relationship("Appointment", back_populates="patient")

    def __repr__(self):
        return f"<Patient {self.patient_id}, Insurance: {self.insurance_no}>"
