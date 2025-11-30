from pathlib import Path
import pickle

models_folder = Path(__file__).parent.parent / "gradient_boost"

model_files = {
    "CR_heavy": models_folder / "cruise_model_heavy.pkl",
    "CR_light": models_folder / "cruise_model_light.pkl",
    # "LVL_A320": models_folder / "cruise_model.pkl", # insert after lvl added to training of cruise model
    "CL_heavy": models_folder / "climb_model_heavy.pkl",
    "CL_light": models_folder / "climb_model_light.pkl",
    "DE_heavy": models_folder / "descent_model_heavy.pkl",
    "DE_light": models_folder / "descent_model_light.pkl",
    "UNKNOWN": models_folder / "all_phase_model.pkl",
}

# model_files = {
#     "CR_": models_folder / "cruise_model.pkl",
#     # "LVL_A320": models_folder / "cruise_model.pkl", # insert after lvl added to training of cruise model
#     "CL_": models_folder / "climb_model.pkl",
#     "DE_": models_folder / "descent_model.pkl",
#     "UNKNOWN": models_folder / "all_phase_model.pkl",
# }

models = {}
for name, file in model_files.items():
    with open(model_files[name], "rb") as f:
        models[name] = pickle.load(f)

def get_fuel_consumption(model_name, X):
    p = models[model_name].predict(X)[0]
    return p