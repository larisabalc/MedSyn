from app.extensions import db
from app.models.appointment_status import AppointmentStatus

class Appointment(db.Model):
    __tablename__ = "appointment"

    appointment_id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    patient_id = db.Column(db.Integer, db.ForeignKey("patient.patient_id"), nullable=False)
    doctor_id = db.Column(db.Integer, db.ForeignKey("doctor.doctor_id"), nullable=False)
    availability_id = db.Column(db.Integer, db.ForeignKey("availability.availability_id"), nullable=False)
    service_id = db.Column(db.Integer, db.ForeignKey("service.service_id"), nullable=False)
    status = db.Column(db.String(20), nullable=False, default=AppointmentStatus.BOOKED.value)

    notes = db.Column(db.String(500), nullable=True)

    patient = db.relationship("Patient", back_populates="appointments")
    doctor = db.relationship("Doctor", back_populates="appointments")
    availability = db.relationship("Availability", back_populates="appointments")
    service = db.relationship("Service", back_populates="appointments")

    notifications = db.relationship("Notification", back_populates="appointment", cascade="all, delete-orphan")

    def __repr__(self):
        return f"<Appointment {self.appointment_id}>"
