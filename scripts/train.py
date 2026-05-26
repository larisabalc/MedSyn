
from diagnosis_engine.models.no_context_diagnosis_classifier import NoContextDiagnosisClassifier

model = NoContextDiagnosisClassifier()
model.load_model("C:/Users/larisabalc/Desktop/MedSyn/diagnosis_engine/trained_models/no_context")

#model.load_local_dataset()
# model.prepare_dataset()

# model.train(num_train_epochs=1)

# model.evaluate(False)

# model.save_model()

sample_input = (
    "Hoarseness, Vocal Changes, Vocal Fatigue"
)

prediction = model.generate_disease_name(sample_input)
print("Predicted diagnosis:", prediction)
