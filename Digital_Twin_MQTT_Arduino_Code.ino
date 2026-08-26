/*
  SustainTwin V2
  Layer 1 — Build 2

  Physics-aware industrial cooling-process emulator
  with bidirectional MQTT operator control.

  Board:
  Arduino UNO R4 WiFi

  Build 2 retains all working Build 1 features:
  - Thermal dynamics
  - Mechanical speed dynamics
  - Electrical current and power dynamics
  - Healthy vibration behaviour
  - OFF, START-UP, NORMAL and HIGH_LOAD modes
  - Cumulative energy
  - Structured Version 2 MQTT payload
  - Legacy fields for the existing Node-RED dashboard
  - Non-blocking Wi-Fi and MQTT reconnection

  Build 2 adds:
  - MQTT command subscription
  - Automatic and manual control modes
  - Remote speed-command control
  - Remote heat-load control
  - Remote mechanical-load control
  - Command validation and status reporting

  Build 2 deliberately excludes:
  - Fault injection
  - Degradation
  - Sensor noise and drift
  - AI modifications
*/

#include <WiFiS3.h>
#include <PubSubClient.h>
#include <ArduinoJson.h>

// ============================================================
// WIFI AND MQTT CONFIGURATION
// ============================================================

const char* WIFI_SSID = "Decent";
const char* WIFI_PASSWORD = "88888888";

const char* MQTT_HOST = "172.20.10.3";
const int MQTT_PORT = 1883;

// ============================================================
// ASSET AND MQTT CONFIGURATION
// ============================================================

const char* ASSET_ID = "FAN_DT_001";

const char* MODEL_VERSION =
  "process-model-2.0.0-build2";

const char* SCHEMA_VERSION = "2.0";

const char* MQTT_CLIENT_ID =
  "UNO_R4_SustainTwin_V2";

const char* TOPIC_DATA =
  "factory/fan/data";

const char* TOPIC_COMMAND =
  "factory/fan/command";

// ============================================================
// UPDATE AND RECONNECTION INTERVALS
// ============================================================

const unsigned long PHYSICS_INTERVAL_MS = 100;
const unsigned long PUBLISH_INTERVAL_MS = 1000;

const unsigned long WIFI_RETRY_INTERVAL_MS = 10000;
const unsigned long MQTT_RETRY_INTERVAL_MS = 5000;

unsigned long lastPhysicsTime = 0;
unsigned long lastPublishTime = 0;

unsigned long lastWifiAttempt = 0;
unsigned long lastMQTTAttempt = 0;

unsigned long sequenceId = 0;

// ============================================================
// CONTROL MODE
// ============================================================

enum ControlMode : uint8_t {
  CONTROL_AUTOMATIC = 0,
  CONTROL_MANUAL = 1
};

ControlMode controlMode = CONTROL_AUTOMATIC;

const char* getControlModeLabel(uint8_t mode) {

  switch (mode) {

    case CONTROL_AUTOMATIC:
      return "AUTOMATIC";

    case CONTROL_MANUAL:
      return "MANUAL";

    default:
      return "UNKNOWN";
  }
}

// ============================================================
// MANUAL CONTROL COMMANDS
// ============================================================

float manualSpeedCommandPct = 0.0;
float manualProcessHeatLoadW = 70.0;
float manualMechanicalLoad = 0.05;

unsigned long lastCommandTimeMs = 0;
unsigned long acceptedCommandCount = 0;
unsigned long rejectedCommandCount = 0;

// ============================================================
// OPERATING MODES
// ============================================================

enum OperatingMode : uint8_t {
  MODE_OFF = 0,
  MODE_STARTUP = 1,
  MODE_NORMAL = 2,
  MODE_HIGH_LOAD = 3
};

const char* getModeLabel(uint8_t mode) {

  switch (mode) {

    case MODE_OFF:
      return "OFF";

    case MODE_STARTUP:
      return "START-UP";

    case MODE_NORMAL:
      return "NORMAL";

    case MODE_HIGH_LOAD:
      return "HIGH_LOAD";

    default:
      return "UNKNOWN";
  }
}

// ============================================================
// PROCESS INPUTS
// ============================================================

struct ProcessInputs {

  float speedCommandPct;
  float processHeatLoadW;
  float ambientTemperatureC;
  float mechanicalLoad;
  float humidityPct;
  float dustLevel;
  float supplyVoltageV;
};

ProcessInputs inputs = {

  0.0,
  70.0,
  25.0,
  0.05,
  50.0,
  0.0,
  230.0
};

// ============================================================
// LATENT TRUE PROCESS STATE
// ============================================================

struct AssetState {

  float temperatureC;
  float speedRpm;
  float currentA;
  float vibrationG;

  float powerW;
  float cumulativeEnergyWh;

  float coolingEfficiency;
  float coolingRateCPerS;
  float thermalResistanceCPerW;

  OperatingMode operatingMode;
};

AssetState state = {

  35.0,
  0.0,
  0.0,
  0.010,

  0.0,
  0.0,

  1.0,
  0.0,
  0.18,

  MODE_OFF
};

// ============================================================
// CONTROLLED MODEL PARAMETERS
// ============================================================

namespace Model {

  // Mechanical subsystem
  const float RATED_SPEED_RPM = 3000.0;
  const float SPEED_TIME_CONSTANT_S = 1.8;

  // Electrical subsystem
  const float IDLE_CURRENT_A = 0.18;

  const float CURRENT_SPEED_COEFFICIENT =
    1.05;

  const float CURRENT_LOAD_COEFFICIENT =
    0.55;

  const float POWER_FACTOR = 0.88;

  // Thermal subsystem
  const float THERMAL_CAPACITY_J_PER_C =
    650.0;

  const float PASSIVE_THERMAL_RESISTANCE_C_PER_W =
    0.18;

  const float ACTIVE_COOLING_COEFFICIENT =
    8.0;

  // Healthy vibration model
  const float BASE_VIBRATION_G = 0.010;

  const float SPEED_VIBRATION_COEFFICIENT =
    0.020;

  const float LOAD_VIBRATION_COEFFICIENT =
    0.008;

  const float ACCELERATION_VIBRATION_COEFFICIENT =
    0.002;

  // Operating limits
  const float MINIMUM_TEMPERATURE_C = -20.0;
  const float MAXIMUM_TEMPERATURE_C = 120.0;

  // Mode thresholds
  const float HIGH_LOAD_ENTER_W = 210.0;
  const float HIGH_LOAD_EXIT_W = 185.0;

  // Manual-command limits
  const float MIN_SPEED_COMMAND_PCT = 0.0;
  const float MAX_SPEED_COMMAND_PCT = 100.0;

  const float MIN_PROCESS_HEAT_LOAD_W = 0.0;
  const float MAX_PROCESS_HEAT_LOAD_W = 500.0;

  const float MIN_MECHANICAL_LOAD = 0.0;
  const float MAX_MECHANICAL_LOAD = 1.0;
}

// ============================================================
// MQTT AND WIFI CLIENTS
// ============================================================

WiFiClient wifiClient;
PubSubClient mqttClient(wifiClient);

// ============================================================
// VALUE VALIDATION
// ============================================================

bool isFiniteValue(float value) {

  return !isnan(value) && !isinf(value);
}

bool isWithinRange(
  float value,
  float minimumValue,
  float maximumValue
) {

  return (
    isFiniteValue(value) &&
    value >= minimumValue &&
    value <= maximumValue
  );
}

// ============================================================
// MQTT COMMAND CALLBACK
// ============================================================

void mqttCallback(
  char* topic,
  byte* payload,
  unsigned int length
) {

  if (strcmp(topic, TOPIC_COMMAND) != 0) {
    return;
  }

  StaticJsonDocument<512> commandDocument;

  DeserializationError error =
    deserializeJson(
      commandDocument,
      payload,
      length
    );

  if (error) {

    rejectedCommandCount++;

    Serial.print(
      "[COMMAND] Invalid JSON: "
    );

    Serial.println(
      error.c_str()
    );

    return;
  }

  bool commandAccepted = true;
  bool commandContainsUpdate = false;

  ControlMode requestedControlMode =
    controlMode;

  float requestedSpeedCommandPct =
    manualSpeedCommandPct;

  float requestedProcessHeatLoadW =
    manualProcessHeatLoadW;

  float requestedMechanicalLoad =
    manualMechanicalLoad;

  // ----------------------------------------------------------
  // CONTROL MODE
  // ----------------------------------------------------------

  if (
    commandDocument.containsKey(
      "automatic"
    )
  ) {

    commandContainsUpdate = true;

    bool automaticEnabled =
      commandDocument["automatic"].as<bool>();

    requestedControlMode =
      automaticEnabled
        ? CONTROL_AUTOMATIC
        : CONTROL_MANUAL;
  }

  if (
    commandDocument.containsKey(
      "control_mode"
    )
  ) {

    commandContainsUpdate = true;

    const char* requestedMode =
      commandDocument["control_mode"];

    if (requestedMode == nullptr) {

      commandAccepted = false;

    } else if (
      strcmp(requestedMode, "AUTOMATIC") == 0 ||
      strcmp(requestedMode, "automatic") == 0
    ) {

      requestedControlMode =
        CONTROL_AUTOMATIC;

    } else if (
      strcmp(requestedMode, "MANUAL") == 0 ||
      strcmp(requestedMode, "manual") == 0
    ) {

      requestedControlMode =
        CONTROL_MANUAL;

    } else {

      commandAccepted = false;

      Serial.println(
        "[COMMAND] Invalid control_mode"
      );
    }
  }

  // ----------------------------------------------------------
  // SPEED COMMAND
  // ----------------------------------------------------------

  if (
    commandDocument.containsKey(
      "speed_command_pct"
    )
  ) {

    commandContainsUpdate = true;

    float receivedValue =
      commandDocument[
        "speed_command_pct"
      ].as<float>();

    if (
      isWithinRange(
        receivedValue,
        Model::MIN_SPEED_COMMAND_PCT,
        Model::MAX_SPEED_COMMAND_PCT
      )
    ) {

      requestedSpeedCommandPct =
        receivedValue;

    } else {

      commandAccepted = false;

      Serial.println(
        "[COMMAND] Invalid speed_command_pct"
      );
    }
  }

  // ----------------------------------------------------------
  // PROCESS HEAT LOAD
  // ----------------------------------------------------------

  if (
    commandDocument.containsKey(
      "process_heat_load_w"
    )
  ) {

    commandContainsUpdate = true;

    float receivedValue =
      commandDocument[
        "process_heat_load_w"
      ].as<float>();

    if (
      isWithinRange(
        receivedValue,
        Model::MIN_PROCESS_HEAT_LOAD_W,
        Model::MAX_PROCESS_HEAT_LOAD_W
      )
    ) {

      requestedProcessHeatLoadW =
        receivedValue;

    } else {

      commandAccepted = false;

      Serial.println(
        "[COMMAND] Invalid process_heat_load_w"
      );
    }
  }

  // ----------------------------------------------------------
  // MECHANICAL LOAD
  // ----------------------------------------------------------

  if (
    commandDocument.containsKey(
      "mechanical_load"
    )
  ) {

    commandContainsUpdate = true;

    float receivedValue =
      commandDocument[
        "mechanical_load"
      ].as<float>();

    if (
      isWithinRange(
        receivedValue,
        Model::MIN_MECHANICAL_LOAD,
        Model::MAX_MECHANICAL_LOAD
      )
    ) {

      requestedMechanicalLoad =
        receivedValue;

    } else {

      commandAccepted = false;

      Serial.println(
        "[COMMAND] Invalid mechanical_load"
      );
    }
  }

  // ----------------------------------------------------------
  // APPLY COMMAND ATOMICALLY
  // ----------------------------------------------------------

  if (
    commandAccepted &&
    commandContainsUpdate
  ) {

    /*
      When switching from automatic to manual without
      providing manual values, preserve the current
      automatic inputs to avoid an unexpected shutdown.
    */

    if (
      controlMode == CONTROL_AUTOMATIC &&
      requestedControlMode == CONTROL_MANUAL
    ) {

      if (
        !commandDocument.containsKey(
          "speed_command_pct"
        )
      ) {

        requestedSpeedCommandPct =
          inputs.speedCommandPct;
      }

      if (
        !commandDocument.containsKey(
          "process_heat_load_w"
        )
      ) {

        requestedProcessHeatLoadW =
          inputs.processHeatLoadW;
      }

      if (
        !commandDocument.containsKey(
          "mechanical_load"
        )
      ) {

        requestedMechanicalLoad =
          inputs.mechanicalLoad;
      }
    }

    controlMode =
      requestedControlMode;

    manualSpeedCommandPct =
      requestedSpeedCommandPct;

    manualProcessHeatLoadW =
      requestedProcessHeatLoadW;

    manualMechanicalLoad =
      requestedMechanicalLoad;

    lastCommandTimeMs = millis();
    acceptedCommandCount++;

    Serial.println(
      "[COMMAND] Command accepted"
    );

    Serial.print(
      "[COMMAND] Control mode: "
    );

    Serial.println(
      getControlModeLabel(controlMode)
    );

    Serial.print(
      "[COMMAND] Speed command: "
    );

    Serial.print(
      manualSpeedCommandPct,
      2
    );

    Serial.println("%");

    Serial.print(
      "[COMMAND] Process heat load: "
    );

    Serial.print(
      manualProcessHeatLoadW,
      2
    );

    Serial.println(" W");

    Serial.print(
      "[COMMAND] Mechanical load: "
    );

    Serial.println(
      manualMechanicalLoad,
      3
    );

  } else {

    rejectedCommandCount++;

    if (!commandContainsUpdate) {

      Serial.println(
        "[COMMAND] No recognised fields"
      );

    } else {

      Serial.println(
        "[COMMAND] Command rejected"
      );
    }
  }
}

// ============================================================
// WIFI CONNECTION
// ============================================================

void connectWiFi() {

  if (WiFi.status() == WL_CONNECTED) {
    return;
  }

  Serial.println(
    "[WiFi] Attempting connection..."
  );

  WiFi.disconnect();
  delay(500);

  WiFi.begin(
    WIFI_SSID,
    WIFI_PASSWORD
  );

  unsigned long connectionStartTime =
    millis();

  while (
    WiFi.status() != WL_CONNECTED &&
    millis() - connectionStartTime < 10000
  ) {

    delay(250);
  }

  if (WiFi.status() == WL_CONNECTED) {

    Serial.println(
      "[WiFi] Connected"
    );

    unsigned long ipWaitStart =
      millis();

    while (
      WiFi.localIP() ==
        IPAddress(0, 0, 0, 0) &&
      millis() - ipWaitStart < 10000
    ) {

      delay(250);
    }

    Serial.print(
      "[WiFi] IP: "
    );

    Serial.println(
      WiFi.localIP()
    );

  } else {

    Serial.println(
      "[WiFi] Connection failed"
    );
  }
}

// ============================================================
// MQTT CONNECTION
// ============================================================

void connectMQTT() {

  if (WiFi.status() != WL_CONNECTED) {
    return;
  }

  if (
    WiFi.localIP() ==
    IPAddress(0, 0, 0, 0)
  ) {

    Serial.println(
      "[MQTT] No valid WiFi IP yet"
    );

    return;
  }

  if (mqttClient.connected()) {
    return;
  }

  Serial.print(
    "[MQTT] Connecting..."
  );

  if (
    mqttClient.connect(
      MQTT_CLIENT_ID
    )
  ) {

    Serial.println(
      "Connected"
    );

    bool subscribed =
      mqttClient.subscribe(
        TOPIC_COMMAND
      );

    if (subscribed) {

      Serial.print(
        "[MQTT] Subscribed to: "
      );

      Serial.println(
        TOPIC_COMMAND
      );

    } else {

      Serial.println(
        "[MQTT] Command subscription failed"
      );
    }

  } else {

    Serial.print(
      "Failed, state = "
    );

    Serial.println(
      mqttClient.state()
    );
  }
}

// ============================================================
// CONTROLLED AUTOMATIC EXPERIMENT PROFILE
// ============================================================

void updateExperimentProfile(
  unsigned long nowMs
) {

  float elapsedSeconds =
    nowMs / 1000.0;

  /*
    Deterministic automatic sequence:

    0–15 seconds:
      OFF

    15–60 seconds:
      START-UP followed by NORMAL

    60–120 seconds:
      HIGH_LOAD

    After 120 seconds:
      NORMAL
  */

  inputs.ambientTemperatureC =
    25.0 +
    0.8 *
    sin(elapsedSeconds / 45.0);

  inputs.humidityPct = 50.0;
  inputs.dustLevel = 0.0;
  inputs.supplyVoltageV = 230.0;

  if (elapsedSeconds < 15.0) {

    inputs.speedCommandPct = 0.0;
    inputs.processHeatLoadW = 70.0;
    inputs.mechanicalLoad = 0.05;

  } else if (elapsedSeconds < 60.0) {

    inputs.speedCommandPct = 65.0;
    inputs.processHeatLoadW = 145.0;
    inputs.mechanicalLoad = 0.30;

  } else if (elapsedSeconds < 120.0) {

    inputs.speedCommandPct = 90.0;
    inputs.processHeatLoadW = 245.0;
    inputs.mechanicalLoad = 0.62;

  } else {

    inputs.speedCommandPct = 65.0;
    inputs.processHeatLoadW = 145.0;
    inputs.mechanicalLoad = 0.30;
  }
}

// ============================================================
// MANUAL CONTROL PROFILE
// ============================================================

void updateManualControl() {

  inputs.speedCommandPct =
    manualSpeedCommandPct;

  inputs.processHeatLoadW =
    manualProcessHeatLoadW;

  inputs.mechanicalLoad =
    manualMechanicalLoad;

  /*
    Environmental inputs remain healthy and
    controlled in Build 2.
  */

  inputs.ambientTemperatureC = 25.0;
  inputs.humidityPct = 50.0;
  inputs.dustLevel = 0.0;
  inputs.supplyVoltageV = 230.0;
}

// ============================================================
// OPERATING-MODE STATE MACHINE
// ============================================================

void updateOperatingMode() {

  if (inputs.speedCommandPct <= 0.1) {

    state.operatingMode = MODE_OFF;
    return;
  }

  float targetSpeedRpm =
    Model::RATED_SPEED_RPM *
    inputs.speedCommandPct /
    100.0;

  if (
    state.speedRpm <
    0.90 * targetSpeedRpm
  ) {

    state.operatingMode =
      MODE_STARTUP;

    return;
  }

  if (
    state.operatingMode ==
    MODE_HIGH_LOAD
  ) {

    if (
      inputs.processHeatLoadW <
      Model::HIGH_LOAD_EXIT_W
    ) {

      state.operatingMode =
        MODE_NORMAL;
    }

  } else if (
    inputs.processHeatLoadW >=
    Model::HIGH_LOAD_ENTER_W
  ) {

    state.operatingMode =
      MODE_HIGH_LOAD;

  } else {

    state.operatingMode =
      MODE_NORMAL;
  }
}

// ============================================================
// MECHANICAL SUBSYSTEM
// ============================================================

void updateMechanicalModel(float dt) {

  float targetSpeedRpm =
    Model::RATED_SPEED_RPM *
    inputs.speedCommandPct /
    100.0;

  float previousSpeedRpm =
    state.speedRpm;

  float speedDerivative =
    (
      targetSpeedRpm -
      state.speedRpm
    ) /
    Model::SPEED_TIME_CONSTANT_S;

  state.speedRpm +=
    speedDerivative * dt;

  state.speedRpm =
    constrain(
      state.speedRpm,
      0.0,
      Model::RATED_SPEED_RPM
    );

  float normalisedSpeed =
    state.speedRpm /
    Model::RATED_SPEED_RPM;

  float normalisedAcceleration =
    abs(
      state.speedRpm -
      previousSpeedRpm
    ) /
    max(
      1.0f,
      Model::RATED_SPEED_RPM * dt
    );

  state.vibrationG =
    Model::BASE_VIBRATION_G +
    Model::SPEED_VIBRATION_COEFFICIENT *
    normalisedSpeed +
    Model::LOAD_VIBRATION_COEFFICIENT *
    inputs.mechanicalLoad +
    Model::ACCELERATION_VIBRATION_COEFFICIENT *
    normalisedAcceleration;

  if (
    state.operatingMode == MODE_OFF &&
    state.speedRpm < 5.0
  ) {

    state.speedRpm = 0.0;

    state.vibrationG =
      Model::BASE_VIBRATION_G;
  }
}

// ============================================================
// ELECTRICAL SUBSYSTEM
// ============================================================

void updateElectricalModel(float dt) {

  float normalisedSpeed =
    state.speedRpm /
    Model::RATED_SPEED_RPM;

  if (
    state.operatingMode == MODE_OFF &&
    state.speedRpm <= 1.0
  ) {

    state.currentA = 0.0;

  } else {

    state.currentA =
      Model::IDLE_CURRENT_A +
      Model::CURRENT_SPEED_COEFFICIENT *
      normalisedSpeed +
      Model::CURRENT_LOAD_COEFFICIENT *
      inputs.mechanicalLoad;

    /*
      Start-up current is temporarily higher
      than steady-state current.
    */

    if (
      state.operatingMode ==
      MODE_STARTUP
    ) {

      state.currentA *= 1.15;
    }
  }

  state.powerW =
    inputs.supplyVoltageV *
    state.currentA *
    Model::POWER_FACTOR;

  state.cumulativeEnergyWh +=
    state.powerW *
    dt /
    3600.0;
}

// ============================================================
// THERMAL SUBSYSTEM
// ============================================================

void updateThermalModel(float dt) {

  float temperatureDifference =
    max(
      0.0f,
      state.temperatureC -
      inputs.ambientTemperatureC
    );

  float normalisedSpeed =
    state.speedRpm /
    Model::RATED_SPEED_RPM;

  float passiveHeatRemovalW =
    temperatureDifference /
    Model::PASSIVE_THERMAL_RESISTANCE_C_PER_W;

  float activeCoolingW =
    Model::ACTIVE_COOLING_COEFFICIENT *
    state.coolingEfficiency *
    normalisedSpeed *
    temperatureDifference;

  float previousTemperatureC =
    state.temperatureC;

  float temperatureDerivative =
    (
      inputs.processHeatLoadW -
      passiveHeatRemovalW -
      activeCoolingW
    ) /
    Model::THERMAL_CAPACITY_J_PER_C;

  state.temperatureC +=
    temperatureDerivative * dt;

  state.temperatureC =
    constrain(
      state.temperatureC,
      Model::MINIMUM_TEMPERATURE_C,
      Model::MAXIMUM_TEMPERATURE_C
    );

  /*
    Positive cooling rate means the temperature
    is reducing.
  */

  state.coolingRateCPerS =
    (
      previousTemperatureC -
      state.temperatureC
    ) /
    max(dt, 0.001f);
}

// ============================================================
// COMPLETE HEALTHY PROCESS UPDATE
// ============================================================

void updateHealthyProcess(float dt) {

  state.coolingEfficiency = 1.0;

  state.thermalResistanceCPerW =
    Model::PASSIVE_THERMAL_RESISTANCE_C_PER_W;

  updateOperatingMode();
  updateMechanicalModel(dt);
  updateElectricalModel(dt);
  updateThermalModel(dt);
}

// ============================================================
// STRUCTURED MQTT PAYLOAD
// ============================================================

void publishProcessState() {

  if (!mqttClient.connected()) {
    return;
  }

  StaticJsonDocument<1792> document;

  document["schema_version"] =
    SCHEMA_VERSION;

  document["model_version"] =
    MODEL_VERSION;

  document["asset_id"] =
    ASSET_ID;

  document["sequence_id"] =
    sequenceId++;

  document["source_time_ms"] =
    millis();

  document["control_mode"] =
    getControlModeLabel(controlMode);

  // ----------------------------------------------------------
  // OPERATING MODE
  // ----------------------------------------------------------

  JsonObject operatingMode =
    document.createNestedObject(
      "operating_mode"
    );

  operatingMode["code"] =
    static_cast<uint8_t>(
      state.operatingMode
    );

  operatingMode["label"] =
    getModeLabel(
      state.operatingMode
    );

  // ----------------------------------------------------------
  // CONTROL STATUS
  // ----------------------------------------------------------

  JsonObject control =
    document.createNestedObject(
      "control"
    );

  control["mode"] =
    getControlModeLabel(
      controlMode
    );

  control["command_topic"] =
    TOPIC_COMMAND;

  control["last_command_time_ms"] =
    lastCommandTimeMs;

  control["accepted_commands"] =
    acceptedCommandCount;

  control["rejected_commands"] =
    rejectedCommandCount;

  // ----------------------------------------------------------
  // PROCESS INPUTS
  // ----------------------------------------------------------

  JsonObject processInputs =
    document.createNestedObject(
      "inputs"
    );

  processInputs["speed_command_pct"] =
    inputs.speedCommandPct;

  processInputs["process_heat_load_w"] =
    inputs.processHeatLoadW;

  processInputs["ambient_temperature_c"] =
    inputs.ambientTemperatureC;

  processInputs["mechanical_load"] =
    inputs.mechanicalLoad;

  processInputs["humidity_pct"] =
    inputs.humidityPct;

  processInputs["dust_level"] =
    inputs.dustLevel;

  processInputs["supply_voltage_v"] =
    inputs.supplyVoltageV;

  // ----------------------------------------------------------
  // TRUE PROCESS STATE
  // ----------------------------------------------------------

  JsonObject trueState =
    document.createNestedObject(
      "true_state"
    );

  trueState["temperature_c"] =
    state.temperatureC;

  trueState["speed_rpm"] =
    state.speedRpm;

  trueState["current_a"] =
    state.currentA;

  trueState["vibration_g"] =
    state.vibrationG;

  trueState["power_w"] =
    state.powerW;

  trueState["energy_wh"] =
    state.cumulativeEnergyWh;

  trueState["cooling_rate_c_per_s"] =
    state.coolingRateCPerS;

  trueState["cooling_efficiency"] =
    state.coolingEfficiency;

  trueState["thermal_resistance_c_per_w"] =
    state.thermalResistanceCPerW;

  // ----------------------------------------------------------
  // HEALTH
  // ----------------------------------------------------------

  JsonObject health =
    document.createNestedObject(
      "health"
    );

  health["overall"] = 100.0;
  health["status"] = "HEALTHY";

  // ----------------------------------------------------------
  // FAULT
  // ----------------------------------------------------------

  JsonObject fault =
    document.createNestedObject(
      "fault"
    );

  fault["code"] = "F0";
  fault["label"] = "healthy";
  fault["severity"] = 0.0;
  fault["active"] = false;

  // ----------------------------------------------------------
  // LEGACY NODE-RED FIELDS
  // ----------------------------------------------------------

  document["temp"] =
    state.temperatureC;

  document["rpm"] =
    state.speedRpm;

  document["current"] =
    state.currentA;

  document["vibration"] =
    state.vibrationG;

  document["timestamp"] =
    millis();

  document["anomaly"] =
    false;

  // ----------------------------------------------------------
  // SERIALISE AND PUBLISH
  // ----------------------------------------------------------

  char payload[1792];

  size_t payloadLength =
    serializeJson(
      document,
      payload,
      sizeof(payload)
    );

  if (
    payloadLength == 0 ||
    payloadLength >= sizeof(payload)
  ) {

    Serial.println(
      "ERROR: MQTT JSON buffer is too small"
    );

    return;
  }

  bool published =
    mqttClient.publish(
      TOPIC_DATA,
      payload
    );

  if (!published) {

    Serial.print(
      "ERROR: MQTT publication failed. State = "
    );

    Serial.println(
      mqttClient.state()
    );

  } else {

    Serial.println(payload);
  }
}

// ============================================================
// SETUP
// ============================================================

void setup() {

  Serial.begin(115200);
  delay(1500);

  mqttClient.setServer(
    MQTT_HOST,
    MQTT_PORT
  );

  mqttClient.setCallback(
    mqttCallback
  );

  mqttClient.setBufferSize(
    2048
  );

  connectWiFi();
  connectMQTT();

  unsigned long now =
    millis();

  lastPhysicsTime = now;
  lastPublishTime = now;

  lastWifiAttempt = now;
  lastMQTTAttempt = now;

  Serial.println(
    "SustainTwin V2 Layer 1 Build 2 started"
  );

  Serial.print(
    "[CONTROL] Initial mode: "
  );

  Serial.println(
    getControlModeLabel(
      controlMode
    )
  );
}

// ============================================================
// MAIN LOOP
// ============================================================

void loop() {

  unsigned long now =
    millis();

  // ----------------------------------------------------------
  // WIFI RECOVERY
  // ----------------------------------------------------------

  if (
    WiFi.status() != WL_CONNECTED &&
    now - lastWifiAttempt >=
      WIFI_RETRY_INTERVAL_MS
  ) {

    lastWifiAttempt = now;
    connectWiFi();
  }

  // ----------------------------------------------------------
  // MQTT RECOVERY
  // ----------------------------------------------------------

  if (
    WiFi.status() == WL_CONNECTED &&
    !mqttClient.connected() &&
    now - lastMQTTAttempt >=
      MQTT_RETRY_INTERVAL_MS
  ) {

    lastMQTTAttempt = now;
    connectMQTT();
  }

  if (mqttClient.connected()) {
    mqttClient.loop();
  }

  // ----------------------------------------------------------
  // PHYSICS UPDATE
  // ----------------------------------------------------------

  if (
    now - lastPhysicsTime >=
    PHYSICS_INTERVAL_MS
  ) {

    float dt =
      (now - lastPhysicsTime) /
      1000.0;

    /*
      Protect the model from an excessively large
      time step after a long blocking interruption.
    */

    dt = constrain(
      dt,
      0.001f,
      1.0f
    );

    lastPhysicsTime = now;

    if (
      controlMode ==
      CONTROL_AUTOMATIC
    ) {

      updateExperimentProfile(now);

    } else {

      updateManualControl();
    }

    updateHealthyProcess(dt);
  }

  // ----------------------------------------------------------
  // MQTT PUBLICATION
  // ----------------------------------------------------------

  if (
    now - lastPublishTime >=
    PUBLISH_INTERVAL_MS
  ) {

    lastPublishTime = now;
    publishProcessState();
  }
}