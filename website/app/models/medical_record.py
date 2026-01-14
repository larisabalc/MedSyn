from app.extensions import db

class MedicalRecord(db.Model):
    __tablename__ = "medical_record"

    record_id = db.Column(db.Integer, primary_key=True)
    doctor_id = db.Column(db.Integer, db.ForeignKey("doctor.doctor_id"), nullable=False)
    patient_id = db.Column(db.Integer, db.ForeignKey("patient.patient_id"), nullable=False)
    symptoms = db.Column(db.String(500))
    diagnosis = db.Column(db.String(500))
    treatment = db.Column(db.String(500))
    cholesterol_lvl = db.Column(db.Integer)
    blood_pressure_lvl = db.Column(db.Integer)
    is_generated = db.Column(db.Boolean, default=False)

    doctor = db.relationship("Doctor", backref="medical_records")
    patient = db.relationship("Patient", backref="medical_records")
