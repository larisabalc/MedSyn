from datetime import datetime, date, timedelta
import os
import csv

from flask import Blueprint, render_template, request, redirect, url_for, session, jsonify, flash

from app.extensions import db

from app.models.user import User
from app.models.doctor import Doctor
from app.models.specialization import Specialization
from app.models.service import Service
from app.models.availability import Availability
from app.models.appointment import Appointment
from app.models.notification import Notification
from app.models.medical_record import MedicalRecord

from ocr_service.ocr_engine import OCREngine
from ocr_service.medical_extractor import MedicalInfoExtractor

from diagnosis_engine.diagnosis_service import DiagnosisService
from diagnosis_engine.models.context_diagnosis_classifier import ContextDiagnosisClassifier
from diagnosis_engine.models.no_context_diagnosis_classifier import NoContextDiagnosisClassifier

patient_bp = Blueprint("patient", __name__, url_prefix="/patient")

API_KEY = os.getenv("MISTRAL_API_KEY")
UPLOAD_FOLDER = r"website/app/static/uploads"
ALLOWED_EXTENSIONS = {"txt", "pdf", "docx", "png"}
DIAGNOSIS_CSV_PATH = "data/raw/Doctor_Versus_Disease.csv"

@patient_bp.route("/dashboard")
def dashboard():
    user_id = session.get("user_id")
    if not user_id:
        return redirect(url_for("auth.login"))

    user = User.query.get(user_id)

    if user.role.name != "PATIENT":
        return "Unauthorized", 403

    appointments_count = Appointment.query.filter_by(patient_id=user.patient_profile.patient_id).count()

    return render_template(
        "patient/dashboard.html",
        user=user,
        appointments_count=appointments_count
    )

# region AI_Diagnosis
def get_suggested_doctors(diagnosis_result: str):
    """Return a list of Doctor objects based on AI diagnosis using CSV mapping"""
    if not diagnosis_result:
        return []

    diagnosis_result_lower = diagnosis_result.lower()
    matched_specializations = set()

    with open(DIAGNOSIS_CSV_PATH, newline="", encoding="utf-8") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            diag_name = row["diagnosis"].strip().lower()
            spec_name = row["specialization"].strip()
            if diag_name in diagnosis_result_lower:
                matched_specializations.add(spec_name)

    if not matched_specializations:
        return []

    doctors = Doctor.query.join(User).filter(
        Doctor.specialization.has(
            Specialization.specialization_name.in_(matched_specializations)
        )
    ).all()

    return doctors

def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

def categorize_blood_pressure(bp_text):
    """Return 'low', 'normal', 'high' based on systolic/diastolic format '120/80'"""
    try:
        systolic, diastolic = map(int, bp_text.split("/"))
        if systolic < 90 or diastolic < 60:
            return "low"
        elif systolic <= 120 and diastolic <= 80:
            return "normal"
        else:
            return "high"
    except Exception:
        return "unknown"

def categorize_cholesterol(chol_text):
    """Return 'low', 'normal', or 'high' based on numeric mg/dL"""
    try:
        chol = int(chol_text)
        if chol < 200:
            return "normal"
        elif 200 <= chol <= 239:
            return "high"
        else: 
            return "high"
    except Exception:
        return "unknown"

@patient_bp.route("/ai_diagnosis", methods=["GET", "POST"])
def ai_diagnosis():
    user_id = session.get("user_id")
    if not user_id:
        return redirect(url_for("auth.login"))

    user = User.query.get(user_id)
    diagnosis_result = None
    recommended_doctors = []

    if request.method == "POST":
        model_type = request.form.get("model_type")
        symptoms_text = request.form.get("symptoms", "").strip()
        blood_pressure = request.form.get("blood_pressure", "").strip()
        cholesterol = request.form.get("cholesterol", "").strip()

        uploaded_files = request.files.getlist("files")
        uploaded_file_paths = []
        invalid_files = []

        for file in uploaded_files:
            if file:
                if allowed_file(file.filename):
                    filename = f"{datetime.utcnow().strftime('%Y%m%d%H%M%S')}_{file.filename}"
                    filepath = os.path.join(UPLOAD_FOLDER, filename)
                    file.save(filepath)
                    uploaded_file_paths.append(filepath)
                else:
                    invalid_files.append(file.filename)

        if invalid_files:
            flash(f"The following files have invalid extensions: {', '.join(invalid_files)}. Allowed: txt, pdf, docx", "danger")
            return render_template(
                "patient/ai_diagnosis.html",
                user=user,
                diagnosis_result=None,
                recommended_doctors=[]
            )
        
        if model_type == "without_context" and (blood_pressure or cholesterol):
            flash("Blood pressure and cholesterol values are ignored when using 'Without Context' model.", "warning")
            return render_template(
                    "patient/ai_diagnosis.html",
                    user=user,
                    diagnosis_result=None,
                    recommended_doctors=[]
                )

        if model_type == "with_context":
            if not blood_pressure or not cholesterol:
                flash("Blood pressure and cholesterol are required when using 'With Context' and uploading files.", "danger")
                return render_template(
                    "patient/ai_diagnosis.html",
                    user=user,
                    diagnosis_result=None,
                    recommended_doctors=[]
                )

        patient_input = ""
        if model_type == "without_context":
            patient_input = symptoms_text

        elif model_type == "with_context":
            today = date.today()
            user_age = today.year - user.birth_date.year - ((today.month, today.day) < (user.birth_date.month, user.birth_date.day))
            bp_category = categorize_blood_pressure(blood_pressure) if blood_pressure else "unknown"
            chol_category = categorize_cholesterol(cholesterol) if cholesterol else "unknown"

            patient_input = f"The patient is a {user_age}-year-old {user.gender}. "
            if bp_category != "unknown":
                patient_input += f"Blood pressure is {bp_category}. "
            if chol_category != "unknown":
                patient_input += f"Cholesterol level is {chol_category}. "
            if symptoms_text:
                patient_input += f"Reported symptoms: {symptoms_text}. "

        extracted_from_files = []
        if model_type == "with_context" and uploaded_file_paths:
            ocr = OCREngine(API_KEY)
            extractor = MedicalInfoExtractor(API_KEY)
            for filepath in uploaded_file_paths:
                try:
                    text = ocr.extract_text(filepath)
                    info = extractor.extract(text)
                    if "symptoms" in info and info["symptoms"]:
                        extracted_from_files.extend(info["symptoms"])
                except Exception as e:
                    flash(f"Failed to process file {os.path.basename(filepath)}: {str(e)}", "danger")

            if extracted_from_files:
                patient_input += f" Extracted symptoms from files: {', '.join(extracted_from_files)}. "

        if model_type == "without_context":
            has_symptoms = bool(symptoms_text)
        else:
            has_symptoms = bool(symptoms_text) or bool(extracted_from_files)

        if not has_symptoms:
            flash("Please provide at least one symptom (text or valid file).", "warning")
            return render_template(
                "patient/ai_diagnosis.html",
                user=user,
                diagnosis_result=None,
                recommended_doctors=[]
            )

        try:
            if model_type == "with_context":
                context_strategy = ContextDiagnosisClassifier()
                context_strategy.load_model("diagnosis_engine/trained_models/context")
                service = DiagnosisService(strategy=context_strategy)
            else:
                no_context_strategy = NoContextDiagnosisClassifier()
                no_context_strategy.load_model("diagnosis_engine/trained_models/no_context")
                service = DiagnosisService(strategy=no_context_strategy)

            diagnosis_result = service.predict(patient_input)
            recommended_doctors = get_suggested_doctors(diagnosis_result)

            medical_record = MedicalRecord(
                doctor_id=None,  
                patient_id=user.user_id,
                symptoms=symptoms_text,
                diagnosis=diagnosis_result,
                treatment=None,
                cholesterol_lvl=cholesterol if chol_category != "unknown" else None,
                blood_pressure_lvl=blood_pressure if bp_category != "unknown" else None,
                is_generated=True
            )
            db.session.add(medical_record)
            db.session.commit()

            notification = Notification(
                user_id=user.user_id,
                title="New Medical Record Created",
                message=f"A new AI-generated medical record has been created for you.",
                is_read=False,
                send_time=datetime.utcnow() 
            )
            db.session.add(notification)
            db.session.commit()

            flash("AI diagnosis generated and medical record created successfully.", "success")

        except Exception as e:
            flash(f"AI prediction failed: {str(e)}", "danger")

    return render_template(
        "patient/ai_diagnosis.html",
        user=user,
        diagnosis_result=diagnosis_result,
        recommended_doctors=recommended_doctors
    )
# endregion

# region MedicalRecords
@patient_bp.route("/medical-records")
def medical_records():
    user_id = session.get("user_id")
    if not user_id:
        return redirect(url_for("auth.login"))

    patient = User.query.get(user_id).patient_profile
    records = MedicalRecord.query.filter_by(patient_id=patient.patient_id).order_by(MedicalRecord.record_id.desc()).all()

    return render_template("patient/medical_records.html", records=records)
# endregion

# region Notifications
@patient_bp.route("/notifications/mark_seen/<int:notif_id>", methods=["POST"])
def mark_notification_seen(notif_id):
    user_id = session.get("user_id")
    if not user_id:
        return redirect(url_for("auth.login"))

    notif = Notification.query.get_or_404(notif_id)
    if notif.user_id != user_id:
        return "Unauthorized", 403

    notif.is_read = True
    db.session.commit()

    unseen_count = Notification.query.filter_by(user_id=user_id, is_read=False).count()
    session['unseen_notifications'] = unseen_count

    return redirect(url_for("patient.notifications"))

@patient_bp.route("/notifications")
def notifications():
    user_id = session.get("user_id")
    if not user_id:
        return redirect(url_for("auth.login"))

    now = datetime.utcnow()
    notifications = Notification.query.filter(
        Notification.user_id == user_id,
        (Notification.send_time == None) | (Notification.send_time <= now)
    ).order_by(Notification.created_at.desc()).all()

    unread_count = Notification.query.filter(
        Notification.user_id == user_id,
        Notification.is_read == False,
        (Notification.send_time == None) | (Notification.send_time <= now)
    ).count()

    session['unseen_notifications'] = unread_count

    return render_template("patient/notifications.html", notifications=notifications)
# endregion

# region Doctors
@patient_bp.route("/doctors/<int:doctor_id>")
def doctor_profile(doctor_id):
    user_id = session.get("user_id")
    if not user_id:
        return redirect(url_for("auth.login"))

    from app.models.doctor import Doctor

    doctor = Doctor.query.get_or_404(doctor_id)

    return render_template(
        "patient/doctor_profile.html",
        doctor=doctor
    )

@patient_bp.route("/doctors")
def doctors():
    user_id = session.get("user_id")
    if not user_id:
        return redirect(url_for("auth.login"))

    from app.models.doctor import Doctor

    doctors = Doctor.query.all()

    return render_template(
        "patient/doctors.html",
        doctors=doctors
    )
# endregion

# region Profile
@patient_bp.route("/profile/edit", methods=["GET", "POST"])
def edit_profile():
    user_id = session.get("user_id")
    if not user_id:
        return redirect(url_for("auth.login"))

    user = User.query.get(user_id)

    if not user or not user.patient_profile:
        return redirect(url_for("auth.login"))

    if request.method == "POST":
        user.username = request.form.get("username").strip()
        user.gender = request.form.get("gender")
        
        birth_date = request.form.get("birth_date")
        if birth_date:
            user.birth_date = datetime.strptime(birth_date, "%Y-%m-%d").date()

        patient = user.patient_profile
        patient.insurance_no = request.form.get("insurance_no")
        patient.emergency_contact = request.form.get("emergency_contact")

        db.session.commit()

        flash("Profile updated successfully!", "success")
        return redirect(url_for("patient.profile"))

    return render_template(
        "patient/edit_profile.html",
        user=user
    )

@patient_bp.route("/profile")
def profile():
    user_id = session.get("user_id")
    if not user_id:
        return redirect(url_for("auth.login"))

    user = User.query.get(user_id)
    return render_template("patient/profile.html", user=user)
# endregion

# region Appointments
@patient_bp.route("/appointments/cancel/<int:appointment_id>", methods=["POST"])
def cancel_appointment(appointment_id):
    user_id = session.get("user_id")
    if not user_id:
        return redirect(url_for("auth.login"))

    user = User.query.get(user_id)
    if user.role.name != "PATIENT":
        return "Unauthorized", 403

    appointment = Appointment.query.get(appointment_id)
    if not appointment or appointment.patient.user.user_id != user_id:
        return "Appointment not found", 404

    availability = appointment.availability
    availability.availability_status = False
    appointment.status = "Canceled"

    Notification.query.filter_by(appointment_id=appointment.appointment_id).delete(synchronize_session=False)
    db.session.commit()

    doctor_user = appointment.doctor.user
    notif_message = (
        f"The appointment with patient {appointment.patient.user.username} "
        f"on {appointment.availability.availability_date.strftime('%Y-%m-%d')} "
        f"from {appointment.availability.start_time.strftime('%H:%M')} "
        f"to {appointment.availability.end_time.strftime('%H:%M')} has been canceled."
    )
    doctor_notification = Notification(
        user_id=doctor_user.user_id,
        appointment_id=appointment.appointment_id,
        title="Appointment Canceled",
        message=notif_message,
        send_time=datetime.utcnow()
    )

    db.session.add(doctor_notification)
    db.session.commit()

    return redirect(url_for("patient.appointments"))

@patient_bp.route("/appointments")
def appointments():
    user_id = session.get("user_id")
    if not user_id:
        return redirect(url_for("auth.login"))

    user = User.query.get(user_id)
    patient = user.patient_profile

    appointments_list = Appointment.query.filter_by(patient_id=patient.patient_id).all()

    return render_template(
        "patient/appointments.html",
        user=user,
        appointments=appointments_list,
        current_time=datetime.utcnow()
    )

@patient_bp.route("/appointments/new", methods=["GET"])
def new_appointment():
    user_id = session.get("user_id")
    if not user_id:
        return redirect(url_for("auth.login"))

    user = User.query.get(user_id)
    specializations = Specialization.query.filter(Specialization.specialization_name != "Unassigned").all()

    return render_template(
        "patient/new_appointment.html",
        user=user,
        specializations=specializations,
        date = date
    )

@patient_bp.route("/appointments/services/<int:spec_id>")
def get_services(spec_id):
    services = Service.query.filter_by(specialization_id=spec_id).all()
    data = [{"id": s.service_id, "name": s.service_name} for s in services]
    return jsonify(data)

@patient_bp.route("/appointments/doctors/<int:spec_id>")
def get_doctors(spec_id):
    doctors = Doctor.query.filter_by(specialization_id=spec_id).all()
    data = [{"id": d.doctor_id, "name": d.user.username} for d in doctors]
    return jsonify(data)

@patient_bp.route("/appointments/availabilities/<int:doctor_id>")
def get_availabilities(doctor_id):
    date_str = request.args.get("date")
    now = datetime.utcnow()

    query = Availability.query.filter_by(doctor_id=doctor_id, availability_status=True)

    if date_str:
        selected_date = datetime.strptime(date_str, "%Y-%m-%d").date()
        query = query.filter(Availability.availability_date == selected_date)
        if selected_date == now.date():
            query = query.filter(Availability.start_time > now.time())
    else:
        query = query.filter(
            (Availability.availability_date > now.date()) |
            ((Availability.availability_date == now.date()) & (Availability.start_time > now.time()))
        )

    availabilities = query.order_by(Availability.availability_date, Availability.start_time).all()

    for a in availabilities:
        if a.availability_date < now.date() or (a.availability_date == now.date() and a.start_time <= now.time()):
            a.availability_status = False
            db.session.add(a)
    db.session.commit()

    availabilities = [a for a in availabilities if a.availability_status]

    data = [
        {
            "id": a.availability_id,
            "date": a.availability_date.strftime("%Y-%m-%d"),
            "start": a.start_time.strftime("%H:%M"),
            "end": a.end_time.strftime("%H:%M")
        }
        for a in availabilities
    ]

    return jsonify(data)

@patient_bp.route("/appointments/new", methods=["POST"])
def create_appointment():
    user_id = session.get("user_id")
    if not user_id:
        return redirect(url_for("auth.login"))

    patient = User.query.get(user_id).patient_profile

    service_id = request.form.get("service")
    doctor_id = request.form.get("doctor")
    availability_id = request.form.get("availability")
    notes = request.form.get("notes", "")

    if not availability_id:
        flash("Please select an available time slot before booking.", "warning")
        user = User.query.get(session.get("user_id"))
        specializations = Specialization.query.filter(Specialization.specialization_name != "Unassigned").all()
        return render_template(
            "patient/new_appointment.html",
            user=user,
            specializations=specializations,
            date=date
        )
    
    if not service_id or not doctor_id:
        flash("Please select a service and a doctor.", "warning")
        user = User.query.get(session.get("user_id"))
        specializations = Specialization.query.filter(Specialization.specialization_name != "Unassigned").all()
        return render_template(
            "patient/new_appointment.html",
            user=user,
            specializations=specializations,
            date=date
        )

    appointment = Appointment(
        patient_id=patient.patient_id,
        doctor_id=doctor_id,
        service_id=service_id,
        availability_id=availability_id,
        notes=notes
    )

    availability = Availability.query.get(availability_id)
    availability.availability_status = False

    db.session.add(appointment)
    db.session.commit()

    appointment_time = datetime.combine(
        appointment.availability.availability_date,
        appointment.availability.start_time
    )
    reminder_time = appointment_time - timedelta(hours=1)

    doctor_user_id = appointment.doctor.user.user_id

    doctor_msg = (
        f"New appointment booked by {patient.user.username} on "
        f"{appointment.availability.availability_date.strftime('%Y-%m-%d')} "
        f"from {appointment.availability.start_time.strftime('%H:%M')} "
        f"to {appointment.availability.end_time.strftime('%H:%M')}."
    )
    doctor_notification = Notification(
        user_id=doctor_user_id,
        appointment_id=appointment.appointment_id,
        title="New Appointment",
        message=doctor_msg,
        send_time=datetime.utcnow() 
    )

    patient_notification = Notification(
        user_id=patient.user.user_id,
        appointment_id=appointment.appointment_id,
        title="Appointment Reminder",
        message=(
            f"Your appointment with Dr. {appointment.doctor.user.username} "
            f"on {appointment.availability.availability_date.strftime('%Y-%m-%d')} "
            f"from {appointment.availability.start_time.strftime('%H:%M')} "
            f"to {appointment.availability.end_time.strftime('%H:%M')}."
        ),
        send_time=reminder_time
    )

    doctor_reminder = Notification(
        user_id=doctor_user_id,
        appointment_id=appointment.appointment_id,
        title="Appointment Reminder",
        message=(
            f"Upcoming appointment with {patient.user.username} on "
            f"{appointment.availability.availability_date.strftime('%Y-%m-%d')} "
            f"from {appointment.availability.start_time.strftime('%H:%M')} "
            f"to {appointment.availability.end_time.strftime('%H:%M')}."
        ),
        send_time=reminder_time
    )

    db.session.add_all([doctor_notification, patient_notification, doctor_reminder])
    db.session.commit()

    return redirect(url_for("patient.dashboard"))
# endregion