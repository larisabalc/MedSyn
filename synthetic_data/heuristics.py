import random
import json
from synthetic_data.utils import contains_any_keyword

with open("synthetic_data/heuristics_config.json") as f:
    CONFIG = json.load(f)

def assign_gender(disease_name, symptoms=None):
    text = str(disease_name) + " " + str(symptoms or "")
    g_conf = CONFIG["gender"]
    if contains_any_keyword(text, g_conf["female"] + g_conf["mature_female"]):
        return "Female"
    if contains_any_keyword(text, g_conf["male"] + g_conf["mature_male"]):
        return "Male"
    return random.choice(["Male", "Female"])

def assign_age(disease_name, symptoms=None):
    text = str(disease_name) + " " + str(symptoms or "")
    a_conf = CONFIG["age"]

    if contains_any_keyword(text, CONFIG["gender"]["female"]):
        return random.randint(*a_conf["female"])
    if contains_any_keyword(text, CONFIG["gender"]["mature_female"]):
        return random.randint(*a_conf["mature_female"])
    if contains_any_keyword(text, CONFIG["gender"]["male"]):
        return random.randint(*a_conf["male"])
    if contains_any_keyword(text, CONFIG["gender"]["mature_male"]):
        return random.randint(*a_conf["mature_male"])
    if contains_any_keyword(text, CONFIG.get("elderly", [])):
        return random.randint(*a_conf.get("elderly", [60, 90]))
    return random.randint(*a_conf["default"])

def assign_blood_pressure(disease_name, symptoms=None):
    text = str(disease_name) + " " + str(symptoms or "")
    bp_conf = CONFIG["blood_pressure"]
    if contains_any_keyword(text, bp_conf["high"]):
        return "High"
    if contains_any_keyword(text, bp_conf["low"]):
        return "Low"
    return random.choices(["Normal", "High", "Low"], weights=[0.7, 0.2, 0.1], k=1)[0]

def assign_cholesterol(disease_name, symptoms=None):
    text = str(disease_name) + " " + str(symptoms or "")
    chol_conf = CONFIG["cholesterol"]
    if contains_any_keyword(text, chol_conf["high"]):
        return "High"
    return random.choices(["Normal", "High"], weights=[0.8, 0.2], k=1)[0]
