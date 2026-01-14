
from diagnosis_engine.models.no_context_diagnosis_classifier import NoContextDiagnosisClassifier

model = NoContextDiagnosisClassifier()

#model.load_local_dataset()
model.prepare_dataset()

model.train(num_train_epochs=1)

model.evaluate(False)

model.save_model()

sample_input = (
    "The patient is a 28-year-old female. "
    "The patient has high cholesterol. "
    "Reported symptoms include itchy, red, inflamed skin, rash."
)

prediction = model.generate_disese_name(sample_input)
print("Predicted diagnosis:", prediction)
