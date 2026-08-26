from pathlib import Path
import os

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
_env_root = os.getenv("SUSTAINTWIN_ROOT")
if _env_root:
    PROJECT_ROOT = Path(_env_root).resolve()
elif (PACKAGE_ROOT / "data").exists():
    PROJECT_ROOT = PACKAGE_ROOT.resolve()
elif (PACKAGE_ROOT.parent / "data").exists():
    # Convenient when the V2 package is extracted as a subfolder inside the main project.
    PROJECT_ROOT = PACKAGE_ROOT.parent.resolve()
else:
    PROJECT_ROOT = PACKAGE_ROOT.resolve()

EXPERIMENT_DIR = PROJECT_ROOT / "data" / "Experiment results"
REAL_DATA_FILE = PROJECT_ROOT / "data" / "New Fan Data.csv"
OUTPUT_ROOT = PROJECT_ROOT / "outputs"

WP1_DIR = OUTPUT_ROOT / "wp1_experiments"
WP2_DIR = OUTPUT_ROOT / "wp2_real_validation"
WP3_DIR = OUTPUT_ROOT / "wp3_ai"
EVIDENCE_DIR = OUTPUT_ROOT / "manuscript_evidence"

# Build 3 experimental protocol (seconds)
BASELINE_END_S = 60.0
FAULT_END_S = 540.0
EXPERIMENT_END_S = 600.0
STEADY_FAULT_START_S = 120.0
STEADY_FAULT_END_S = 480.0

# Configurable industrial historian screening. These values are research filters,
# not universal fan thresholds. Preserve them in the manuscript methods if used.
REAL_NORMAL_MIN_RPM = float(os.getenv("SUSTAINTWIN_REAL_MIN_RPM", "600"))
REAL_NORMAL_MIN_POWER_KW = float(os.getenv("SUSTAINTWIN_REAL_MIN_POWER_KW", "500"))

COMMON_SYNTH = {
    "rpm": "actual_rpm",
    "power": "power_w",
    "vibration": "vibration_g",
    "temperature": "temperature_c",
    "current": "current_a",
}
COMMON_REAL = {
    "rpm": "Speed (RPM)",
    "power": "Motor Power ( kW)",
    "vibration": "Fan Vibration (mm/s)",
    "temperature": "Bag Filter Outlet Temp.",
    "current": "Current",
}

# Leakage-safe diagnostic feature sets.
ML_OBSERVABLE_FEATURES = [
    "temperature_c",
    "actual_rpm",
    "current_a",
    "vibration_g",
    "power_w",
    "cooling_rate_c_per_s",
]
ML_PHYSICS_FEATURES = ML_OBSERVABLE_FEATURES + [
    "cooling_efficiency",
    "thermal_resistance_c_per_w",
]

# Evaluation settings
RANDOM_STATE = 42
N_GROUP_FOLDS = 3
FAULT_CLASSES = ["F0", "F1", "F2", "F3", "F4"]
FAULT_ONLY_CLASSES = ["F1", "F2", "F3", "F4"]
SEVERITIES = [0.3, 0.6, 0.9]
