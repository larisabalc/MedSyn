from app.extensions import db

class Doctor(db.Model):
    __tablename__ = "doctor"

    doctor_id = db.Column(db.Integer, db.ForeignKey("users.user_id"), primary_key=True)
    specialization_id = db.Column(db.Integer, db.ForeignKey("specialization.specialization_id"), nullable=False)
    licence_no = db.Column(db.String(100), nullable=True)

    user = db.relationship("User", back_populates="doctor_profile", uselist=False)
    specialization = db.relationship("Specialization", back_populates="doctors", uselist=False)

    availabilities = db.relationship("Availability", back_populates="doctor")
    appointments = db.relationship("Appointment", back_populates="doctor")

    def __repr__(self):
        return f"<Doctor {self.doctor_id}, License: {self.licence_no}>"
