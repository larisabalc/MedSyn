from synthetic_data.patient_profile_factory import PatientProfileFactory
from synthetic_data.profile_mapper import ProfileMapper
from synthetic_data.dataset_builder import DatasetBuilder
import csv

mapper = ProfileMapper("data/raw/Disease_symptom_and_patient_profile_dataset.csv")
factory = PatientProfileFactory(n_versions=5)

builder = DatasetBuilder(mapper, factory)
final_dataset = builder.build(n_synthetic_versions=5)

final_dataset.to_csv("data/synthetic/final_training_dataset.csv", index=False, quoting=csv.QUOTE_ALL)