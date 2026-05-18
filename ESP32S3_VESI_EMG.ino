/*
 * ESP32-S3 + HX711 (Весы) + AD8232 (EMG датчик)
 * Гибридный скетч для тестирования мышечной силы и активации
 * 
 * ПРИМЕНЕНИЕ: Испытуемый прижимает весы ногой, оператор отслеживает
 * усилие (вес) и одновременную активацию мышцы (EMG сигнал)
 * 
 * WEB ИНТЕРФЕЙС: точка доступа Wi-Fi с реал-тайм данными
 */

#include <Wire.h>
#include <Adafruit_GFX.h>
#include <Adafruit_SSD1306.h>
#include "HX711.h"
#include <math.h>
#include <EEPROM.h>
#include <WiFi.h>
#include <WebServer.h>

// ============= DISPLAY SETTINGS =============
#define SCREEN_WIDTH 128
#define SCREEN_HEIGHT 64
#define OLED_RESET -1

Adafruit_SSD1306 display(SCREEN_WIDTH, SCREEN_HEIGHT, &Wire, OLED_RESET);

// ============= ВEСИ (HX711) - PINOUT ESP32-S3 =============
#define HX711_DOUT 8      // GPIO8
#define HX711_SCK 9       // GPIO9

HX711 scale;

// ============= EMG (AD8232) - PINOUT ESP32-S3 =============
#define EMG_SIGNAL_PIN 7  // GPIO7 ADC input; SIG is wired here on the current build.
// #define EMG_LO_PLUS 5     // GPIO5 - не используется (нет вывода на датчике)
// #define EMG_LO_MINUS 6    // GPIO6 - не используется (нет вывода на датчике)

// ============= SAMPLING SETTINGS =============
const unsigned long WEIGHT_UPDATE_INTERVAL_MS = 100;  // Обновление весов каждые 100мс
const unsigned long EMG_SAMPLE_INTERVAL_MS = 2;       // EMG sampled every 2ms (~500Hz)
const unsigned long DISPLAY_UPDATE_INTERVAL_MS = 100; // Экран обновляется каждые 100мс
const float EMG_SAMPLE_RATE_HZ = 500.0f;

// ============= CONSTANTS =============
const float GRAMS_PER_KG = 1000.0f;
const float GRAVITY = 9.80665f;  // Для перевода в ньютоны
const bool INVERT_POLARITY = true;
const float ADC_REFERENCE_VOLTAGE = 3.3f;
const int ADC_MAX_VALUE = 4095;
const float DEFAULT_ADC_GAIN_UV_PER_COUNT = (ADC_REFERENCE_VOLTAGE * 1000000.0f) / ADC_MAX_VALUE;

// Synthetic EMG mode is intended for demos and hardware-failure fallback only.
// It must be reported as synthetic data, not as measured surface EMG.
bool syntheticEmgEnabled = true;
float syntheticEmgMaxForceN = 420.0f;
const float SYNTH_EMG_REST_COUNTS = 3.0f;
const float SYNTH_EMG_MAX_COUNTS = 880.0f;
const float SYNTH_EMG_ATTACK_ALPHA = 0.13f;
const float SYNTH_EMG_RELEASE_ALPHA = 0.022f;
const float SYNTH_PROFILE_REFERENCE_RMS_COUNTS = 46.0f;
const float SYNTH_RMS_MAX_UV = 450.0f;

enum SyntheticEmgProfile {
  SYNTH_PROFILE_TRAINED = 0,
  SYNTH_PROFILE_UNTRAINED = 1,
  SYNTH_PROFILE_JOINT_PATHOLOGY = 2,
  SYNTH_PROFILE_SPINE_PATHOLOGY = 3
};

int syntheticEmgProfile = SYNTH_PROFILE_TRAINED;

// ============= WEIGHT FILTERING (Adaptive EMA) =============
const float ALPHA_FAST = 0.8f;
const float ALPHA_SLOW = 0.01f;
const float JUMP_THRESHOLD = 10.0f;
float filteredWeight = 0.0f;
float displayedWeight = 0.0f;
float peakWeight = 0.0f;  // ПИКОВОЕ значение веса
unsigned long lastWeightUpdate = 0;
unsigned long lastDisplayUpdate = 0;

// ============= EMG PROCESSING =============
const int EMG_BUFFER_SIZE = 256;
int emgBuffer[EMG_BUFFER_SIZE];
int emgFilteredBuffer[EMG_BUFFER_SIZE];
int emgBufferIndex = 0;
int emgSamplesCollected = 0;
unsigned long lastEMGSample = 0;

// EMG Baseline Calibration (важно для AD8232!)
float emgBaseline = 2048.0f;  // Начальное значение для 12-бит АЦП
bool emgCalibrated = false;
bool emgNeedsRecalibration = false;  // Флаг для запроса перекалибровки
unsigned long emgCalibrationStart = 0;
const unsigned long EMG_CALIBRATION_TIME_MS = 3000;  // 3 сек для калибровки

// EMG Statistics
int emgMin = 4095;
int emgMax = 0;
int emgAvg = 0;
int emgPeak = 0;        // ПИКОВОЕ значение EMG
int emgLastValue = 0;
float emgRMS = 0.0f;    // RMS (Root Mean Square)
int emgFilteredAvg = 0;
int emgFilteredPeak = 0;
int emgFilteredLastValue = 0;
float emgFilteredRMS = 0.0f;
bool emgElectrodeDetached = false;
int emgCurrentWindowMin = 0;
int emgCurrentWindowMax = 0;
int emgCurrentWindowAvg = 0;
int emgFilteredWindowMin = 0;
int emgFilteredWindowMax = 0;
int emgFilteredWindowAvg = 0;
unsigned long emgClipLowCount = 0;
unsigned long emgClipHighCount = 0;
bool emgClippingDetected = false;
float syntheticEmgEnvelopeCounts = 0.0f;
float syntheticEmgFatigue = 0.0f;
float syntheticEmgDrift = 0.0f;

// Программная фильтрация EMG: high-pass + notch 50 Hz
const float EMG_HPF_CUTOFF_HZ = 12.0f;
const float EMG_NOTCH_FREQ_HZ = 50.0f;
const float EMG_NOTCH_R = 0.95f;
const int EMG_RMS_WINDOW_SIZE = 64;         // ~128 мс при 500 Гц
const float EMG_RMS_SMOOTH_ALPHA = 0.35f;   // быстрее реагирует на сокращение
const float EMG_PEAK_DECAY = 0.82f;         // плавное затухание пика на графиках
float emgHighPassState = 0.0f;
float emgHighPassPrevInput = 0.0f;
float emgNotchX1 = 0.0f;
float emgNotchX2 = 0.0f;
float emgNotchY1 = 0.0f;
float emgNotchY2 = 0.0f;

// ============= CALIBRATION =============
float calibrationFactor = 1.0f;
long offsetValue = 0;
bool weightCalibrated = false;  // Флаг калибровки весов
const int EEPROM_CALIBRATION_ADDR = 0;      // 4 байта для calibrationFactor
const int EEPROM_OFFSET_ADDR = 4;           // 4 байта для offsetValue
const int EEPROM_ADC_GAIN_ADDR = 8;         // 4 байта для adcGainUvPerCount
const int EEPROM_ADC_OFFSET_ADDR = 12;      // 4 байта для adcOffsetUv

const int EEPROM_FOOT_LENGTH_ADDR = 16;
const int EEPROM_SYNTH_PROFILE_ADDR = 20;
float adcGainUvPerCount = DEFAULT_ADC_GAIN_UV_PER_COUNT;
float adcOffsetUv = 0.0f;
float footLengthCm = 25.0f;

// ============= WI-FI & WEB SERVER =============
const char* ssid = "VESI-EMG-AP";
const char* password = "vesi1234";  // 8+ символов
WebServer server(80);
bool wifiInitialized = false;
const IPAddress apIP(192, 168, 4, 1);
const IPAddress apGateway(192, 168, 4, 1);
const IPAddress apSubnet(255, 255, 255, 0);

// ============= DATA LOGGING =============
const int MAX_DATA_POINTS = 5000;  // Максимум точек данных
struct DataPoint {
  unsigned long timestamp;
  float weight;
  int emgPeak;
  float emgRMS;
  int emgRaw;
  int emgFilteredPeak;
  float emgFilteredRMS;
  int emgFilteredRaw;
};
DataPoint dataLog[MAX_DATA_POINTS];
int dataLogIndex = 0;

// ============= MODE STATE =============
bool diagnosticMode = false;
bool recordingMode = false;
unsigned long recordingStartTime = 0;

const char* resetReasonToText(esp_reset_reason_t reason) {
  switch (reason) {
    case ESP_RST_UNKNOWN: return "UNKNOWN";
    case ESP_RST_POWERON: return "POWERON";
    case ESP_RST_EXT: return "EXTERNAL";
    case ESP_RST_SW: return "SOFTWARE";
    case ESP_RST_PANIC: return "PANIC";
    case ESP_RST_INT_WDT: return "INT_WDT";
    case ESP_RST_TASK_WDT: return "TASK_WDT";
    case ESP_RST_WDT: return "OTHER_WDT";
    case ESP_RST_DEEPSLEEP: return "DEEPSLEEP";
    case ESP_RST_BROWNOUT: return "BROWNOUT";
    case ESP_RST_SDIO: return "SDIO";
    default: return "UNMAPPED";
  }
}

const char* syntheticProfileKey(int profile) {
  switch (profile) {
    case SYNTH_PROFILE_TRAINED: return "trained";
    case SYNTH_PROFILE_UNTRAINED: return "untrained";
    case SYNTH_PROFILE_JOINT_PATHOLOGY: return "joint";
    case SYNTH_PROFILE_SPINE_PATHOLOGY: return "spine";
    default: return "trained";
  }
}

const char* syntheticProfileLabel(int profile) {
  switch (profile) {
    case SYNTH_PROFILE_TRAINED: return "Norm, trained";
    case SYNTH_PROFILE_UNTRAINED: return "Norm, untrained";
    case SYNTH_PROFILE_JOINT_PATHOLOGY: return "Joint pathology";
    case SYNTH_PROFILE_SPINE_PATHOLOGY: return "Spine pathology";
    default: return "Norm, trained";
  }
}

float syntheticProfileAmplitudeUv(int profile) {
  switch (profile) {
    case SYNTH_PROFILE_TRAINED: return 359.72f;
    case SYNTH_PROFILE_UNTRAINED: return 196.16f;
    case SYNTH_PROFILE_JOINT_PATHOLOGY: return 155.92f;
    case SYNTH_PROFILE_SPINE_PATHOLOGY: return 153.00f;
    default: return 359.72f;
  }
}

int parseSyntheticProfile(const String &value) {
  String key = value;
  key.trim();
  key.toLowerCase();
  if (key == "trained" || key == "0") return SYNTH_PROFILE_TRAINED;
  if (key == "untrained" || key == "1") return SYNTH_PROFILE_UNTRAINED;
  if (key == "joint" || key == "joint_pathology" || key == "2") return SYNTH_PROFILE_JOINT_PATHOLOGY;
  if (key == "spine" || key == "spine_pathology" || key == "3") return SYNTH_PROFILE_SPINE_PATHOLOGY;
  return -1;
}

float syntheticUvPerCount() {
  return syntheticProfileAmplitudeUv(syntheticEmgProfile) / SYNTH_PROFILE_REFERENCE_RMS_COUNTS;
}

float syntheticCountsForUv(float uv) {
  float uvPerCount = syntheticUvPerCount();
  if (uvPerCount <= 0.0f) {
    return 0.0f;
  }
  return uv / uvPerCount;
}

float countsToMicrovolts(float counts) {
  if (syntheticEmgEnabled) {
    float uv = counts * syntheticUvPerCount();
    return (uv > 0.0f) ? uv : 0.0f;
  }
  float uv = counts * adcGainUvPerCount - adcOffsetUv;
  return (uv > 0.0f) ? uv : 0.0f;
}

float weightToNewtons(float weightGrams) {
  return (weightGrams / GRAMS_PER_KG) * GRAVITY;
}

float weightToKgfCm(float weightGrams) {
  return (weightGrams / GRAMS_PER_KG) * footLengthCm;
}

float clampFloat(float value, float minValue, float maxValue) {
  if (value < minValue) {
    return minValue;
  }
  if (value > maxValue) {
    return maxValue;
  }
  return value;
}

float randomUnit() {
  return (float)random(-1000, 1001) / 1000.0f;
}

int generateSyntheticEmgRaw() {
  const float t = millis() * 0.001f;
  const float maxForce = (syntheticEmgMaxForceN > 1.0f) ? syntheticEmgMaxForceN : 1.0f;
  const float forceN = weightToNewtons(displayedWeight);
  const float peakForceN = weightToNewtons(peakWeight);
  const float forceNorm = clampFloat(forceN / maxForce, 0.0f, 1.15f);
  const float force01 = clampFloat(forceNorm, 0.0f, 1.0f);
  const float forceSlope = clampFloat((forceN - peakForceN * 0.88f) / maxForce, -0.25f, 0.25f);

  // Calf plantar-flexion: quiet rest, quicker mid-range recruitment, delayed relaxation.
  const float recruited = powf(force01, 1.38f);
  const float lateDrive = powf(force01, 2.35f);
  const float holdDrive = powf(force01, 0.92f);
  const float activation =
    clampFloat(0.12f * forceNorm + 0.50f * recruited + 0.24f * lateDrive + 0.14f * holdDrive, 0.0f, 1.08f);
  const float contractionBurst = 1.0f + 0.12f * max(0.0f, forceSlope);
  const float releaseLag = 1.0f + 0.10f * max(0.0f, -forceSlope);

  const float fatigueTarget = recordingMode ? activation : 0.0f;
  syntheticEmgFatigue += (fatigueTarget - syntheticEmgFatigue) * 0.0015f;

  const float tremor =
    1.0f +
    0.035f * sinf(2.0f * PI * 6.5f * t) +
    0.016f * sinf(2.0f * PI * 11.0f * t + 1.1f) +
    0.009f * sinf(2.0f * PI * 16.0f * t + 0.35f) +
    0.010f * randomUnit();

  float targetEnvelope =
    SYNTH_EMG_REST_COUNTS +
    SYNTH_EMG_MAX_COUNTS * activation * contractionBurst * releaseLag * tremor * (1.0f + 0.10f * syntheticEmgFatigue);
  targetEnvelope = clampFloat(targetEnvelope, SYNTH_EMG_REST_COUNTS, SYNTH_EMG_MAX_COUNTS * 1.25f);

  const float envelopeAlpha =
    (targetEnvelope > syntheticEmgEnvelopeCounts) ? SYNTH_EMG_ATTACK_ALPHA : SYNTH_EMG_RELEASE_ALPHA;
  syntheticEmgEnvelopeCounts += (targetEnvelope - syntheticEmgEnvelopeCounts) * envelopeAlpha;

  const float carrier =
    0.46f * sinf(2.0f * PI * 83.0f * t) +
    0.32f * sinf(2.0f * PI * 137.0f * t + 0.7f) +
    0.22f * sinf(2.0f * PI * 211.0f * t + 2.0f);
  const float burstNoise = (randomUnit() + 0.55f * randomUnit() + 0.25f * randomUnit()) / 1.80f;

  syntheticEmgDrift = 0.9990f * syntheticEmgDrift + 0.0010f * (randomUnit() * 14.0f);
  const float sample =
    (0.52f * carrier + 0.48f * burstNoise) * syntheticEmgEnvelopeCounts +
    syntheticEmgDrift +
    randomUnit() * (1.4f + 0.008f * syntheticEmgEnvelopeCounts);

  int rawValue = (int)roundf(emgBaseline + sample);
  if (rawValue < 20) {
    rawValue = 20;
  } else if (rawValue > ADC_MAX_VALUE - 20) {
    rawValue = ADC_MAX_VALUE - 20;
  }
  return rawValue;
}

void setup() {
  Serial.begin(115200);
  delay(1000);
  randomSeed((uint32_t)micros());
  esp_reset_reason_t resetReason = esp_reset_reason();
  Serial.print("Reset reason: ");
  Serial.println(resetReasonToText(resetReason));
  Serial.print("Free heap: ");
  Serial.println(ESP.getFreeHeap());
  
  Serial.println("\n╔═══════════════════════════════════════════════╗");
  Serial.println("║   ESP32-S3 ВEСИ + EMG Датчик (Гибрид)        ║");
  Serial.println("║   Тестирование мышечной силы и активации      ║");
  Serial.println("╚═══════════════════════════════════════════════╝\n");

  // Init Display
  Serial.println("[1/4] Инициализация дисплея...");
  Wire.begin(21, 20);  // SDA=GPIO21, SCL=GPIO20 (для ESP32-S3 N16R8)
  if (!display.begin(SSD1306_SWITCHCAPVCC, 0x3C)) {
    Serial.println("❌ ОШИБКА: Дисплей не найден!");
  } else {
    Serial.println("✓ Дисплей инициализирован");
  }
  display.clearDisplay();
  display.setTextColor(SSD1306_WHITE);
  displayStatus("Init...");

  // Init Weight Sensor
  Serial.println("[2/4] Инициализация датчика весов (HX711)...");
  scale.begin(HX711_DOUT, HX711_SCK);
  loadCalibrationFromEEPROM();
  scale.set_scale(calibrationFactor);
  scale.set_offset(offsetValue);  // Восстанавливаем сохранённое смещение
  Serial.println("✓ HX711 инициализирован");
  
  Serial.println("   Тарирование весов...");
  tryTare();
  Serial.println("✓ Тарирование завершено");

  // Init EMG ADC
  Serial.println("[3/4] Инициализация ADC для EMG...");
  analogReadResolution(12);
  analogSetAttenuation(ADC_11db);
  pinMode(EMG_SIGNAL_PIN, INPUT);
  Serial.println("✓ ADC инициализирован");
  Serial.print("✓ ADC gain (uV/count): ");
  Serial.println(adcGainUvPerCount, 3);
  Serial.print("✓ ADC offset (uV): ");
  Serial.println(adcOffsetUv, 1);
  if (syntheticEmgEnabled) {
    Serial.println("NOTE: Synthetic EMG mode is ON. EMG is generated from force and is not measured surface EMG.");
  }
  
  // Start EMG calibration
  Serial.println("\n⚠️  КАЛИБРОВКА EMG: держите мышцу РАССЛАБЛЕННОЙ на 3 сек...");
  emgCalibrationStart = millis();

  Serial.println("\n═══════════════════════════════════════════════");
  Serial.println("Система готова. Нажмите на весы для начала теста.");
  Serial.println("═══════════════════════════════════════════════\n");
  
  // Init Wi-Fi Access Point
  Serial.println("[4/4] Инициализация Wi-Fi AP...");
  initWiFiAP();
  
  displayStatus("sausage");
  delay(1000);
}

void loop() {
  // Обновление весов
  if (scale.is_ready()) {
    unsigned long now = millis();
    
    if (now - lastWeightUpdate >= WEIGHT_UPDATE_INTERVAL_MS) {
      lastWeightUpdate = now;
      updateWeight();
    }
  }

  // Сбор EMG данных (постоянно)
  unsigned long now = millis();
  if (now - lastEMGSample >= EMG_SAMPLE_INTERVAL_MS) {
    lastEMGSample = now;
    collectEMGSample();
  }
  
  // Обработка EMG буфера каждые 100мс
  static unsigned long lastProcessTime = 0;
  if (millis() - lastProcessTime >= 100) {
    lastProcessTime = millis();
    processEMGBuffer();
  }

  // Обновление дисплея
  if (millis() - lastDisplayUpdate >= DISPLAY_UPDATE_INTERVAL_MS) {
    lastDisplayUpdate = millis();
    updateDisplay();
  }

  // Обработка команд через Serial
  if (wifiInitialized) {
    server.handleClient();
  }

  if (Serial.available() > 0) {
    String command = Serial.readStringUntil('\n');
    command.trim();
    handleSerialCommand(command);
  }

  delay(1);
}

// ============= WEIGHT FUNCTIONS =============
void updateWeight() {
  long raw = scale.read();
  
  // Calculate weight
  long zeroed = raw - scale.get_offset();
  // Используем абсолютные значения (работает независимо от полярности)
  float rawWeight = (float)abs(zeroed) / abs(scale.get_scale());

  // Apply Adaptive EMA Filter
  if (filteredWeight == 0.0f && rawWeight > 0.0f) {
    filteredWeight = rawWeight;
  } else {
    float diff = fabs(rawWeight - filteredWeight);
    float alpha = (diff > JUMP_THRESHOLD) ? ALPHA_FAST : ALPHA_SLOW;
    filteredWeight = (alpha * rawWeight) + ((1.0f - alpha) * filteredWeight);
  }

  displayedWeight = filteredWeight;

  // Update peak
  if (displayedWeight > peakWeight) {
    peakWeight = displayedWeight;
  }

  if (recordingMode) {
    Serial.print(millis() - recordingStartTime);
    Serial.print(",");
    Serial.print(displayedWeight, 2);
    Serial.print(",");
    Serial.print(emgPeak);
    Serial.print(",");
    Serial.print(emgRMS, 2);
    Serial.print(",");
    Serial.print(emgLastValue);
    Serial.print(",");
    Serial.print(emgFilteredPeak);
    Serial.print(",");
    Serial.print(emgFilteredRMS, 2);
    Serial.print(",");
    Serial.println(emgFilteredLastValue);
    
    // Сохраняем в буфер для экспорта
    if (dataLogIndex < MAX_DATA_POINTS) {
      dataLog[dataLogIndex].timestamp = millis() - recordingStartTime;
      dataLog[dataLogIndex].weight = displayedWeight;
      dataLog[dataLogIndex].emgPeak = emgPeak;
      dataLog[dataLogIndex].emgRMS = emgRMS;
      dataLog[dataLogIndex].emgRaw = emgLastValue;
      dataLog[dataLogIndex].emgFilteredPeak = emgFilteredPeak;
      dataLog[dataLogIndex].emgFilteredRMS = emgFilteredRMS;
      dataLog[dataLogIndex].emgFilteredRaw = emgFilteredLastValue;
      dataLogIndex++;
    }
  }
}

// ============= EMG FUNCTIONS =============
void collectEMGSample() {
  int rawValue = syntheticEmgEnabled ? generateSyntheticEmgRaw() : analogRead(EMG_SIGNAL_PIN);
  if (!syntheticEmgEnabled) {
    if (rawValue <= 2) {
      emgClipLowCount++;
    } else if (rawValue >= (ADC_MAX_VALUE - 2)) {
      emgClipHighCount++;
    }
  }
  
  // EMG Calibration Phase (первые 3 сек - калибруем baseline при расслабленной мышце)
  // Используем флаг emgNeedsRecalibration для перекалибровки
  if (!emgCalibrated || emgNeedsRecalibration) {
    unsigned long elapsed = millis() - emgCalibrationStart;
    if (elapsed < EMG_CALIBRATION_TIME_MS) {
      // Адаптивная скользящая средняя для baseline
      emgBaseline = emgBaseline * 0.95f + rawValue * 0.05f;
      
      // Диагностика: выводим raw значения для каждого цикла
      if (elapsed % 100 < 2) {  // Каждые 100мс
        Serial.print("RAW: ");
        Serial.print(rawValue);
        Serial.print(" | Baseline: ");
        Serial.print((int)emgBaseline);
        Serial.print(" | Elapsed: ");
        Serial.println(elapsed);
      }
      return;  // Не обрабатываем сигнал во время калибровки
    } else {
      emgCalibrated = true;
      emgNeedsRecalibration = false;  // Сбрасываем флаг перекалибровки
      emgBufferIndex = 0;
      emgSamplesCollected = 0;
      emgMin = 0;
      emgMax = 0;
      emgAvg = 0;
      emgPeak = 0;
      emgLastValue = 0;
      emgRMS = 0.0f;
      emgFilteredAvg = 0;
      emgFilteredPeak = 0;
      emgFilteredLastValue = 0;
      emgFilteredRMS = 0.0f;
      emgCurrentWindowMin = 0;
      emgCurrentWindowMax = 0;
      emgCurrentWindowAvg = 0;
      emgFilteredWindowMin = 0;
      emgFilteredWindowMax = 0;
      emgFilteredWindowAvg = 0;
      emgClipLowCount = 0;
      emgClipHighCount = 0;
      emgClippingDetected = false;
      emgHighPassState = 0.0f;
      emgHighPassPrevInput = 0.0f;
      emgNotchX1 = 0.0f;
      emgNotchX2 = 0.0f;
      emgNotchY1 = 0.0f;
      emgNotchY2 = 0.0f;
      Serial.println("\n✓✓✓ Калибровка EMG завершена!");
      Serial.print("✓ FINAL Базовая линия: ");
      Serial.println((int)emgBaseline);
      Serial.println("Начинаем обработку сигнала...\n");
    }
  }
  
  // Обработка сигнала после калибровки: считаем как отклонение от baseline
  float signalACSigned = (float)rawValue - emgBaseline;
  int signalAC = (int)fabs(signalACSigned);  // Сырой rectified AC

  // High-pass для остаточного дрейфа baseline
  const float dt = 1.0f / EMG_SAMPLE_RATE_HZ;
  const float rc = 1.0f / (2.0f * PI * EMG_HPF_CUTOFF_HZ);
  const float hpAlpha = rc / (rc + dt);
  float highPassOutput = hpAlpha * (emgHighPassState + signalACSigned - emgHighPassPrevInput);
  emgHighPassState = highPassOutput;
  emgHighPassPrevInput = signalACSigned;

  // Notch 50Hz для подавления сетевой помехи
  const float theta = 2.0f * PI * EMG_NOTCH_FREQ_HZ / EMG_SAMPLE_RATE_HZ;
  const float notchCos = cos(theta);
  float notchOutput = highPassOutput
                    - 2.0f * notchCos * emgNotchX1
                    + emgNotchX2
                    + 2.0f * EMG_NOTCH_R * notchCos * emgNotchY1
                    - (EMG_NOTCH_R * EMG_NOTCH_R) * emgNotchY2;
  emgNotchX2 = emgNotchX1;
  emgNotchX1 = highPassOutput;
  emgNotchY2 = emgNotchY1;
  emgNotchY1 = notchOutput;

  int filteredSignal = (int)fabs(notchOutput);
  
  emgBuffer[emgBufferIndex] = signalAC;
  emgFilteredBuffer[emgBufferIndex] = filteredSignal;
  emgBufferIndex = (emgBufferIndex + 1) % EMG_BUFFER_SIZE;
  if (emgSamplesCollected < EMG_BUFFER_SIZE) {
    emgSamplesCollected++;
  }
  emgLastValue = signalAC;
  emgFilteredLastValue = filteredSignal;
  
  // Диагностика первых 10 сэмплов после калибровки
  static int diagCount = 0;
  if (diagCount < 10) {
    Serial.print("POST-CAL RAW: ");
    Serial.print(rawValue);
    Serial.print(" | AC: ");
    Serial.print(signalAC);
    Serial.print(" | FILT: ");
    Serial.print(filteredSignal);
    Serial.print(" | Buffer[");
    Serial.print(emgBufferIndex);
    Serial.print("]=");
    Serial.println(emgBuffer[(emgBufferIndex - 1 + EMG_BUFFER_SIZE) % EMG_BUFFER_SIZE]);
    diagCount++;
  }

  // Check electrode detection (LO signals)
  checkElectrodeStatus();
}

void processEMGBuffer() {
  // Не считаем статистику по пустому или частично невалидному буферу
  const int sampleCount = emgSamplesCollected;
  if (sampleCount <= 0) {
    emgMin = 0;
    emgMax = 0;
    emgAvg = 0;
    emgPeak = 0;
    emgRMS = 0.0f;
    emgFilteredAvg = 0;
    emgFilteredPeak = 0;
    emgFilteredRMS = 0.0f;
    emgCurrentWindowMin = 0;
    emgCurrentWindowMax = 0;
    emgCurrentWindowAvg = 0;
    emgFilteredWindowMin = 0;
    emgFilteredWindowMax = 0;
    emgFilteredWindowAvg = 0;
    emgClippingDetected = false;
    return;
  }

  const int rmsWindowCount = (sampleCount < EMG_RMS_WINDOW_SIZE) ? sampleCount : EMG_RMS_WINDOW_SIZE;
  int sum = 0;
  int filteredSum = 0;
  emgMin = 4095;
  emgMax = 0;
  long sumSquares = 0;
  int filteredMin = 4095;
  int filteredMax = 0;
  long filteredSumSquares = 0;

  for (int i = 0; i < rmsWindowCount; i++) {
    int idx = (emgBufferIndex - rmsWindowCount + i + EMG_BUFFER_SIZE) % EMG_BUFFER_SIZE;
    int val = emgBuffer[idx];
    int filteredVal = emgFilteredBuffer[idx];
    sum += val;
    sumSquares += (long)val * val;
    filteredSum += filteredVal;
    filteredSumSquares += (long)filteredVal * filteredVal;
    
    if (val < emgMin) emgMin = val;
    if (val > emgMax) emgMax = val;
    if (filteredVal < filteredMin) filteredMin = filteredVal;
    if (filteredVal > filteredMax) filteredMax = filteredVal;
  }

  emgAvg = sum / rmsWindowCount;
  emgFilteredAvg = filteredSum / rmsWindowCount;
  emgCurrentWindowMin = emgMin;
  emgCurrentWindowMax = emgMax;
  emgCurrentWindowAvg = emgAvg;
  emgFilteredWindowMin = filteredMin;
  emgFilteredWindowMax = filteredMax;
  emgFilteredWindowAvg = emgFilteredAvg;
  
  // True RMS по последнему короткому окну + легкое сглаживание
  float rawRmsInstant = sqrt((float)sumSquares / rmsWindowCount);
  float filteredRmsInstant = sqrt((float)filteredSumSquares / rmsWindowCount);
  emgRMS = (emgRMS <= 0.0f) ? rawRmsInstant : (EMG_RMS_SMOOTH_ALPHA * rawRmsInstant + (1.0f - EMG_RMS_SMOOTH_ALPHA) * emgRMS);
  emgFilteredRMS = (emgFilteredRMS <= 0.0f) ? filteredRmsInstant : (EMG_RMS_SMOOTH_ALPHA * filteredRmsInstant + (1.0f - EMG_RMS_SMOOTH_ALPHA) * emgFilteredRMS);
  if (syntheticEmgEnabled) {
    const float syntheticRmsMaxCounts = syntheticCountsForUv(SYNTH_RMS_MAX_UV);
    if (emgRMS > syntheticRmsMaxCounts) {
      emgRMS = syntheticRmsMaxCounts;
    }
    if (emgFilteredRMS > syntheticRmsMaxCounts) {
      emgFilteredRMS = syntheticRmsMaxCounts;
    }
  }
  emgClippingDetected = (emgClipLowCount + emgClipHighCount) > 0;

  // Peak с затуханием полезнее для визуализации, чем "вечный максимум"
  emgPeak = (emgPeak < emgMax) ? emgMax : (int)(emgPeak * EMG_PEAK_DECAY);
  emgFilteredPeak = (emgFilteredPeak < filteredMax) ? filteredMax : (int)(emgFilteredPeak * EMG_PEAK_DECAY);
  
  // Diagnostic output - показываем статистику
  static unsigned long lastDiagTime = 0;
  if (millis() - lastDiagTime > 500) {  // Каждые 500мс
    lastDiagTime = millis();
    Serial.print("EMG WIN: Min=");
    Serial.print(emgCurrentWindowMin);
    Serial.print(syntheticEmgEnabled ? " Source=SIM" : " Source=ADC");
    Serial.print(" Max=");
    Serial.print(emgCurrentWindowMax);
    Serial.print(" Avg=");
    Serial.print(emgCurrentWindowAvg);
    Serial.print(" | RMS=");
    Serial.print(rawRmsInstant, 1);
    Serial.print(" | RMSema=");
    Serial.print(emgRMS, 1);
    Serial.print(" (~");
    Serial.print(countsToMicrovolts(emgRMS), 0);
    Serial.print(" uV) | PeakEnv=");
    Serial.print(emgPeak);
    Serial.print(" | FiltRMS=");
    Serial.print(filteredRmsInstant, 1);
    Serial.print(" | FiltRMSema=");
    Serial.print(emgFilteredRMS, 1);
    Serial.print(" (~");
    Serial.print(countsToMicrovolts(emgFilteredRMS), 0);
    Serial.print(" uV) | FiltPeakEnv=");
    Serial.print(emgFilteredPeak);
    Serial.print(" | ClipL=");
    Serial.print(emgClipLowCount);
    Serial.print(" ClipH=");
    Serial.println(emgClipHighCount);
    emgClipLowCount = 0;
    emgClipHighCount = 0;
  }
}

void checkElectrodeStatus() {
  // Детектор электродов не используется (датчик не имеет выводов LO+/LO-)
  emgElectrodeDetached = false;
}

// ============= DISPLAY FUNCTIONS =============
void updateDisplay() {
  display.clearDisplay();
  display.setTextColor(SSD1306_WHITE);
  char line[24];
  float weightN = weightToNewtons(displayedWeight);
  float peakN = weightToNewtons(peakWeight);
  unsigned long elapsed = recordingMode ? (millis() - recordingStartTime) : 0;

  display.setTextSize(1);
  display.setCursor(0, 0);
  display.print(recordingMode ? "REC" : "LIVE");
  display.setCursor(36, 0);
  display.print(weightCalibrated ? "W:OK" : "W:--");
  display.setCursor(78, 0);
  display.print(syntheticEmgEnabled ? "E:SIM" : (emgCalibrated ? "E:OK" : "E:.."));

  if (recordingMode) {
    display.setCursor(108, 0);
    display.print(elapsed / 1000);
    display.print("s");
  } else {
    display.setCursor(102, 0);
    display.print(dataLogIndex);
    display.print("p");
  }

  display.drawLine(0, 9, SCREEN_WIDTH - 1, 9, SSD1306_WHITE);

  display.setTextSize(2);
  dtostrf(weightN, 6, 2, line);
  display.setCursor(0, 13);
  display.print(line);
  display.setTextSize(1);
  display.setCursor(96, 18);
  display.print("N");

  display.setCursor(0, 35);
  display.print("Pk ");
  dtostrf(peakN, 5, 2, line);
  display.print(line);
  display.print("N");

  display.setCursor(0, 45);
  display.print("R ");
  display.print((int)countsToMicrovolts(emgRMS));
  display.print(" F ");
  display.print((int)countsToMicrovolts(emgFilteredRMS));

  display.setCursor(0, 55);
  display.print("Pk ");
  display.print((int)countsToMicrovolts(emgPeak));
  display.print(" F");
  display.print((int)countsToMicrovolts(emgFilteredPeak));
  display.print("u");

  display.display();
}

void displayStatus(const char *msg) {
  display.clearDisplay();
  display.setTextSize(2);
  int16_t x1, y1;
  uint16_t w, h;
  display.getTextBounds(msg, 0, 0, &x1, &y1, &w, &h);
  display.setCursor((SCREEN_WIDTH - w) / 2, (SCREEN_HEIGHT - h) / 2);
  display.println(msg);
  display.display();
}

// ============= CALIBRATION & STORAGE =============
void loadCalibrationFromEEPROM() {
  EEPROM.begin(512);
  
  // Загружаем calibrationFactor
  byte* ptr = (byte*)&calibrationFactor;
  for (int i = 0; i < sizeof(float); i++) {
    ptr[i] = EEPROM.read(EEPROM_CALIBRATION_ADDR + i);
  }
  
  // Загружаем offsetValue
  byte* offsetPtr = (byte*)&offsetValue;
  for (int i = 0; i < sizeof(long); i++) {
    offsetPtr[i] = EEPROM.read(EEPROM_OFFSET_ADDR + i);
  }

  // Загружаем adcGainUvPerCount
  byte* adcGainPtr = (byte*)&adcGainUvPerCount;
  for (int i = 0; i < sizeof(float); i++) {
    adcGainPtr[i] = EEPROM.read(EEPROM_ADC_GAIN_ADDR + i);
  }

  // Загружаем adcOffsetUv
  byte* adcOffsetPtr = (byte*)&adcOffsetUv;
  for (int i = 0; i < sizeof(float); i++) {
    adcOffsetPtr[i] = EEPROM.read(EEPROM_ADC_OFFSET_ADDR + i);
  }

  byte* footLengthPtr = (byte*)&footLengthCm;
  for (int i = 0; i < sizeof(float); i++) {
    footLengthPtr[i] = EEPROM.read(EEPROM_FOOT_LENGTH_ADDR + i);
  }

  syntheticEmgProfile = EEPROM.read(EEPROM_SYNTH_PROFILE_ADDR);
  
  EEPROM.end();
  
  if (isnan(calibrationFactor) || fabs(calibrationFactor) < 1.0f || fabs(calibrationFactor) > 1000.0f) {
    calibrationFactor = 420.0f; // Default value
    offsetValue = 0;
    weightCalibrated = false;
    Serial.println("⚠️  Калибровка загружена неверно, используется значение по умолчанию");
  } else {
    weightCalibrated = true;
  }

  if (isnan(adcGainUvPerCount) || adcGainUvPerCount < 100.0f || adcGainUvPerCount > 5000.0f) {
    adcGainUvPerCount = DEFAULT_ADC_GAIN_UV_PER_COUNT;
    Serial.println("⚠️  ADC gain загружен неверно, используется значение по умолчанию");
  }

  if (isnan(adcOffsetUv) || fabs(adcOffsetUv) > 500000.0f) {
    adcOffsetUv = 0.0f;
    Serial.println("⚠️  ADC offset загружен неверно, используется 0");
  }

  if (isnan(footLengthCm) || footLengthCm < 1.0f || footLengthCm > 100.0f) {
    footLengthCm = 25.0f;
    Serial.println("⚠️  Длина стопы загружена неверно, используется 25.0 см");
  }

  if (syntheticEmgProfile < SYNTH_PROFILE_TRAINED || syntheticEmgProfile > SYNTH_PROFILE_SPINE_PATHOLOGY) {
    syntheticEmgProfile = SYNTH_PROFILE_TRAINED;
    Serial.println("⚠️  Профиль synthetic EMG загружен неверно, используется trained");
  }
  
  Serial.print("✓ Коэффициент калибровки: ");
  Serial.println(calibrationFactor);
  Serial.print("✓ Смещение offset: ");
  Serial.println(offsetValue);
  Serial.print("✓ ADC gain (uV/count): ");
  Serial.println(adcGainUvPerCount, 3);
  Serial.print("✓ ADC offset (uV): ");
  Serial.println(adcOffsetUv, 1);
  Serial.print("✓ Foot length (cm): ");
  Serial.println(footLengthCm, 1);
  Serial.print("✓ Synthetic EMG profile: ");
  Serial.println(syntheticProfileKey(syntheticEmgProfile));
}

void saveCalibrationToEEPROM() {
  EEPROM.begin(512);
  
  // Сохраняем calibrationFactor
  byte* ptr = (byte*)&calibrationFactor;
  for (int i = 0; i < sizeof(float); i++) {
    EEPROM.write(EEPROM_CALIBRATION_ADDR + i, ptr[i]);
  }
  
  // Сохраняем offsetValue
  offsetValue = scale.get_offset();
  byte* offsetPtr = (byte*)&offsetValue;
  for (int i = 0; i < sizeof(long); i++) {
    EEPROM.write(EEPROM_OFFSET_ADDR + i, offsetPtr[i]);
  }

  // Сохраняем adcGainUvPerCount
  byte* adcGainPtr = (byte*)&adcGainUvPerCount;
  for (int i = 0; i < sizeof(float); i++) {
    EEPROM.write(EEPROM_ADC_GAIN_ADDR + i, adcGainPtr[i]);
  }

  // Сохраняем adcOffsetUv
  byte* adcOffsetPtr = (byte*)&adcOffsetUv;
  for (int i = 0; i < sizeof(float); i++) {
    EEPROM.write(EEPROM_ADC_OFFSET_ADDR + i, adcOffsetPtr[i]);
  }

  byte* footLengthPtr = (byte*)&footLengthCm;
  for (int i = 0; i < sizeof(float); i++) {
    EEPROM.write(EEPROM_FOOT_LENGTH_ADDR + i, footLengthPtr[i]);
  }

  EEPROM.write(EEPROM_SYNTH_PROFILE_ADDR, (uint8_t)syntheticEmgProfile);
  
  EEPROM.commit();
  EEPROM.end();
  Serial.println("✓ Калибровка и offset сохранены в EEPROM");
}

void tryTare() {
  if (scale.is_ready()) {
    scale.tare();
    Serial.println("✓ Весы отнулены");
  }
}

void restartEmgAcquisition() {
  emgNeedsRecalibration = true;
  emgCalibrated = false;
  emgCalibrationStart = millis();
  emgBufferIndex = 0;
  emgSamplesCollected = 0;
  emgPeak = 0;
  emgRMS = 0.0f;
  emgFilteredPeak = 0;
  emgFilteredRMS = 0.0f;
  emgClipLowCount = 0;
  emgClipHighCount = 0;
  emgClippingDetected = false;
  syntheticEmgEnvelopeCounts = 0.0f;
  syntheticEmgFatigue = 0.0f;
  syntheticEmgDrift = 0.0f;
}

// ============= SERIAL COMMANDS =============
void handleSerialCommand(String cmd) {
  cmd.toUpperCase();
  
  if (cmd == "TARE") {
    tryTare();
    Serial.println("✓ Весы отнулены");
    
  } else if (cmd == "CALIB") {
    Serial.println("Режим калибровки. Укажите известный вес (в граммах):");
    displayStatus("Calib...");
    
  } else if (cmd.startsWith("CAL ")) {
    float knownMass = cmd.substring(4).toFloat();
    if (knownMass > 0) {
      long reading = scale.read();
      calibrationFactor = (reading - scale.get_offset()) / knownMass;
      scale.set_scale(calibrationFactor);
      saveCalibrationToEEPROM();
      Serial.print("✓ Калибровка: ");
      Serial.println(calibrationFactor);
    }
    
  } else if (cmd == "REC") {
    startRecording();
    
  } else if (cmd == "STOP") {
    stopRecording();
    
  } else if (cmd == "RESET") {
    peakWeight = 0;
    emgPeak = 0;
    Serial.println("✓ Пиковые значения сброшены");
    
  } else if (cmd == "STAT") {
    printStatistics();
    
  } else if (cmd == "DIAG") {
    diagnosticMode = !diagnosticMode;
    Serial.println(diagnosticMode ? "Диагностический режим ВКЛ" : "Диагностический режим ВЫКЛ");
    
  } else if (cmd == "SIMEMG ON") {
    syntheticEmgEnabled = true;
    restartEmgAcquisition();
    Serial.println("Synthetic EMG ON: generated from force sensor, not measured surface EMG.");

  } else if (cmd == "SIMEMG OFF") {
    syntheticEmgEnabled = false;
    restartEmgAcquisition();
    Serial.println("Synthetic EMG OFF: reading ADC input.");

  } else if (cmd.startsWith("SIMMAX ")) {
    float maxForce = cmd.substring(7).toFloat();
    if (maxForce >= 50.0f && maxForce <= 2000.0f) {
      syntheticEmgMaxForceN = maxForce;
      restartEmgAcquisition();
      Serial.print("Synthetic EMG max force set to ");
      Serial.print(syntheticEmgMaxForceN, 1);
      Serial.println(" N");
    } else {
      Serial.println("Use: SIMMAX <50..2000 N>");
    }

  } else if (cmd.startsWith("SIMPROFILE ")) {
    int profile = parseSyntheticProfile(cmd.substring(11));
    if (profile >= 0) {
      syntheticEmgProfile = profile;
      saveCalibrationToEEPROM();
      restartEmgAcquisition();
      Serial.print("Synthetic EMG profile: ");
      Serial.println(syntheticProfileKey(syntheticEmgProfile));
    } else {
      Serial.println("Use: SIMPROFILE trained|untrained|joint|spine");
    }
    
  } else if (cmd == "EMG") {
    Serial.println("=== Статус EMG ===");
    Serial.print("Source: ");
    Serial.println(syntheticEmgEnabled ? "SYNTHETIC_FORCE_MODEL" : "ADC_INPUT");
    Serial.print("Synthetic max force N: ");
    Serial.println(syntheticEmgMaxForceN, 1);
    Serial.print("Synthetic profile: ");
    Serial.print(syntheticProfileLabel(syntheticEmgProfile));
    Serial.print(" (");
    Serial.print(syntheticProfileKey(syntheticEmgProfile));
    Serial.println(")");
    Serial.print("Raw: ");
    Serial.println(emgLastValue);
    Serial.print("Raw uV out: ");
    Serial.println(countsToMicrovolts(emgLastValue), 0);
    Serial.print("Filt: ");
    Serial.println(emgFilteredLastValue);
    Serial.print("Filt uV out: ");
    Serial.println(countsToMicrovolts(emgFilteredLastValue), 0);
    Serial.print("Min-Max: ");
    Serial.print(emgMin);
    Serial.print(" - ");
    Serial.println(emgMax);
    Serial.print("RMS: ");
    Serial.print(emgRMS);
    Serial.print(" (~");
    Serial.print(countsToMicrovolts(emgRMS), 0);
    Serial.println(" uV out)");
    Serial.print("Filtered RMS: ");
    Serial.print(emgFilteredRMS);
    Serial.print(" (~");
    Serial.print(countsToMicrovolts(emgFilteredRMS), 0);
    Serial.println(" uV out)");
    Serial.print("ADC gain (uV/count): ");
    Serial.println(adcGainUvPerCount, 3);
    Serial.print("ADC offset (uV): ");
    Serial.println(adcOffsetUv, 1);
    Serial.print("Electrode: ");
    Serial.println(emgElectrodeDetached ? "DETACHED ⚠️" : "OK ✓");

  } else if (cmd == "ADCSTAT") {
    Serial.println("=== ADC Calibration ===");
    Serial.print("Gain (uV/count): ");
    Serial.println(adcGainUvPerCount, 3);
    Serial.print("Offset (uV): ");
    Serial.println(adcOffsetUv, 1);
    Serial.println("Use: ADCCAL <gain_uV_per_count> <offset_uV>");

  } else if (cmd.startsWith("ADCCAL ")) {
    String values = cmd.substring(7);
    values.trim();
    int sep = values.indexOf(' ');
    if (sep > 0) {
      float newGain = values.substring(0, sep).toFloat();
      float newOffset = values.substring(sep + 1).toFloat();
      if (newGain >= 100.0f && newGain <= 5000.0f && fabs(newOffset) <= 500000.0f) {
        adcGainUvPerCount = newGain;
        adcOffsetUv = newOffset;
        saveCalibrationToEEPROM();
        Serial.println("✓ ADC calibration updated");
      } else {
        Serial.println("⚠️  Invalid ADC calibration values");
      }
    } else {
      Serial.println("⚠️  Use format: ADCCAL <gain_uV_per_count> <offset_uV>");
    }
    
  } else {
    printHelp();
  }
}

void startRecording() {
  if (recordingMode) {
    return;
  }
  dataLogIndex = 0;
  recordingMode = true;
  recordingStartTime = millis();
  if (syntheticEmgEnabled) {
    Serial.println("NOTE: Recording uses Synthetic EMG generated from force, not measured surface EMG.");
  }
  filteredWeight = 0.0f;
  displayedWeight = 0.0f;
  peakWeight = 0;
  emgPeak = 0;
  Serial.println("\n╔════════════════════════════════════╗");
  Serial.println("║       ЗАПИСЬ НАЧАТА                 ║");
  Serial.println("║ time,weight,raw,filt emg           ║");
  Serial.println("╚════════════════════════════════════╝");
}

void stopRecording() {
  if (!recordingMode) {
    return;
  }
  recordingMode = false;
  Serial.println("\n╔════════════════════════════════════╗");
  Serial.println("║       ЗАПИСЬ ЗАВЕРШЕНА              ║");
  Serial.println("╚════════════════════════════════════╝");
  printStatistics();
}

void printStatistics() {
  Serial.println("\n╔════════════════════════════════════╗");
  Serial.println("║        ИТОГОВАЯ СТАТИСТИКА         ║");
  Serial.println("╚════════════════════════════════════╝");
  
  Serial.print("Максимальный вес:    ");
  Serial.print(peakWeight / GRAMS_PER_KG, 3);
  Serial.println(" кг");
  
  Serial.print("Пиковое EMG:         ");
  Serial.print(emgPeak);
  Serial.print(" (~");
  Serial.print(countsToMicrovolts(emgPeak), 0);
  Serial.println(" uV out)");
  
  Serial.print("Диапазон EMG:        ");
  Serial.print(emgMin);
  Serial.print(" - ");
  Serial.println(emgMax);
  
  Serial.print("RMS EMG:             ");
  Serial.print(emgRMS);
  Serial.print(" (~");
  Serial.print(countsToMicrovolts(emgRMS), 0);
  Serial.println(" uV out)");

  Serial.print("RMS EMG filtered:    ");
  Serial.print(emgFilteredRMS);
  Serial.print(" (~");
  Serial.print(countsToMicrovolts(emgFilteredRMS), 0);
  Serial.println(" uV out)");
  
  Serial.println("════════════════════════════════════\n");
}

void printHelp() {
  Serial.println("\n╔════════════════════════════════════╗");
  Serial.println("║         ДОСТУПНЫЕ КОМАНДЫ          ║");
  Serial.println("╠════════════════════════════════════╣");
  Serial.println("║ TARE    - Отнулить весы           ║");
  Serial.println("║ CALIB   - Режим калибровки        ║");
  Serial.println("║ CAL xxx - Калибровка (xxx грамм)  ║");
  Serial.println("║ REC     - Начать запись данных    ║");
  Serial.println("║ STOP    - Остановить запись       ║");
  Serial.println("║ RESET   - Сбросить пики          ║");
  Serial.println("║ STAT    - Показать статистику     ║");
  Serial.println("║ EMG     - Статус EMG датчика      ║");
  Serial.println("║ ADCSTAT - Показать ADC калибровку ║");
  Serial.println("║ ADCCAL  - ADC gain/offset         ║");
  Serial.println("║ DIAG    - Диагностический режим   ║");
  Serial.println("╚════════════════════════════════════╝\n");
}
// ============= WI-FI FUNCTIONS =============
void initWiFiAP() {
  WiFi.persistent(false);
  WiFi.disconnect(true, true);
  delay(100);
  WiFi.mode(WIFI_AP);
  WiFi.setSleep(false);
  WiFi.softAPdisconnect(true);
  if (!WiFi.softAPConfig(apIP, apGateway, apSubnet)) {
    Serial.println("AP IP config failed");
  }
  bool apStarted = WiFi.softAP(ssid, password, 1, false, 4);
  if (!apStarted) {
    Serial.println("\nWi-Fi Access Point start failed");
    wifiInitialized = false;
    return;
  }
  delay(250);
  
  IPAddress IP = WiFi.softAPIP();
  Serial.println("\n✓ Wi-Fi Access Point создана!");
  Serial.print("SSID: ");
  Serial.println(ssid);
  Serial.print("PASS: ");
  Serial.println(password);
  Serial.print("Channel: ");
  Serial.println(WiFi.channel());
  Serial.print("Stations: ");
  Serial.println(WiFi.softAPgetStationNum());
  Serial.print("IP: ");
  Serial.println(IP);
  Serial.print("Адрес: http://");
  Serial.println(IP);
  
  // Setup Web Server
  server.on("/", handleRoot);
  server.on("/index.html", handleRoot);
  server.on("/api/data", handleData);
  server.on("/api/tare", handleTare);
  server.on("/api/cal", handleCal);
  server.on("/api/foot", handleFoot);
  server.on("/api/profile", handleProfile);
  server.on("/api/emg-calib", handleEMGCalib);  // Новый эндпоинт
  server.on("/api/rec", handleRec);
  server.on("/api/stop", handleStop);
  server.on("/api/export", handleExport);
  server.on("/ping", []() {
    server.send(200, "text/plain", "pong");
  });
  server.onNotFound([]() {
    server.send(404, "text/plain", "Not found");
  });
  server.begin();
  
  wifiInitialized = true;
  Serial.println("✓ Веб-сервер запущен");
}

void handleRoot() {
  static const char html[] PROGMEM = R"rawliteral(
<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>VESI + EMG</title>
<style>
body{margin:0;background:#0f141a;color:#eef3f8;font-family:Segoe UI,Arial,sans-serif}
.wrap{max-width:920px;margin:0 auto;padding:18px}
.hero,.panel,.metric{background:#18212b;border:1px solid #273342;border-radius:16px;box-shadow:0 10px 24px rgba(0,0,0,.22)}
.hero{padding:18px 20px;margin-bottom:14px}
.hero h1{margin:0 0 8px;font-size:26px}
.muted{color:#98abbe;font-size:13px}
.status{margin-top:10px;font-weight:700}
.grid{display:grid;grid-template-columns:repeat(3,1fr);gap:12px;margin-bottom:14px}
.metric{padding:14px}
.k{font-size:12px;color:#98abbe;text-transform:uppercase;letter-spacing:.08em}
.v{margin-top:8px;font-size:30px;font-weight:800}
.sub{margin-top:6px;font-size:12px;color:#98abbe}
.panel{padding:16px;margin-bottom:14px}
.row{display:flex;gap:10px;flex-wrap:wrap}
button,input{border:none;border-radius:10px;font-size:14px;padding:11px 14px}
button{cursor:pointer;color:#fff;background:#2d7b62}
.secondary{background:#35629c}
.danger{background:#b84f4f}
.recording{background:#cf3f3f !important;animation:recPulse 1s ease-in-out infinite}
input{background:#10171e;color:#eef3f8;border:1px solid #22303d;min-width:130px}
.chartBox{background:#0d1217;border:1px solid #22303d;border-radius:12px;padding:10px}
svg{display:block;width:100%;height:180px}
.legend{display:flex;gap:16px;flex-wrap:wrap;margin-top:10px;font-size:12px;color:#98abbe}
.sw{display:inline-block;width:18px;height:3px;border-radius:99px;margin-right:6px;vertical-align:middle}
pre{margin:0;white-space:pre-wrap;background:#0d1217;border:1px solid #22303d;border-radius:12px;padding:12px;font-size:12px;color:#c9d4df}
@keyframes recPulse{0%{box-shadow:0 0 0 0 rgba(207,63,63,.45)}70%{box-shadow:0 0 0 10px rgba(207,63,63,0)}100%{box-shadow:0 0 0 0 rgba(207,63,63,0)}}
@media (max-width:760px){.grid{grid-template-columns:1fr}.v{font-size:26px}}
</style>
</head>
<body>
<div class="wrap">
  <section class="hero">
    <h1>VESI + EMG Monitor</h1>
    <div class="muted">Lightweight dashboard build for ESP32 stability</div>
    <div class="status" id="s">loading...</div>
  </section>

  <section class="grid">
    <article class="metric">
      <div class="k">Force</div>
      <div class="v"><span id="f">0</span> N</div>
      <div class="sub">Current force</div>
    </article>
    <article class="metric">
      <div class="k">Peak Force</div>
      <div class="v"><span id="p">0</span> N</div>
      <div class="sub">Session peak</div>
    </article>
    <article class="metric">
      <div class="k">EMG RMS Raw</div>
      <div class="v"><span id="r">0</span> uV</div>
      <div class="sub">Raw RMS output</div>
    </article>
    <article class="metric">
      <div class="k">EMG RMS Filt</div>
      <div class="v"><span id="fr">0</span> uV</div>
      <div class="sub">Filtered RMS output</div>
    </article>
    <article class="metric">
      <div class="k">EMG Peak</div>
      <div class="v"><span id="ep">0</span> uV</div>
      <div class="sub">Peak envelope</div>
    </article>
    <article class="metric">
      <div class="k">Signal</div>
      <div class="v"><span id="clip">OK</span></div>
      <div class="sub">Clip monitor</div>
    </article>
  </section>

  <section class="panel">
    <div class="k" style="margin-bottom:10px">Weight Calibration</div>
    <div class="row" style="margin-bottom:10px">
      <button class="secondary" onclick="tare()">TARE SCALE</button>
      <div class="muted" id="tareState">Zero the platform before calibration.</div>
    </div>
    <div class="row">
      <input id="cw" type="number" value="1000" min="1" placeholder="Known mass g">
      <button class="secondary" onclick="calib()">CALIBRATE WEIGHT</button>
      <div class="muted" id="calState">Place the reference mass, then run calibration.</div>
    </div>
    <div class="row" style="margin-top:10px">
      <input id="fl" type="number" value="25" min="1" step="0.1" placeholder="Foot length cm">
      <button class="secondary" onclick="saveFoot()">SAVE FOOT LENGTH</button>
      <div class="muted" id="footState">Used for torque in the report.</div>
    </div>
  </section>

  <section class="panel">
    <div class="k" style="margin-bottom:10px">Session Control</div>
    <div class="row">
      <button id="recBtn" onclick="startRec()">REC</button>
      <button class="danger" onclick="stopAndDownload()">STOP & DL</button>
      <button class="secondary" onclick="api('/api/emg-calib')">EMG RE-CALIB</button>
      <a href="/ping" style="color:#9fc9ff;align-self:center;text-decoration:none">ping</a>
    </div>
  </section>

  <section class="panel">
    <div class="k" style="margin-bottom:10px">Synthetic EMG Profile</div>
    <div class="row">
      <select id="sp" style="border:none;border-radius:10px;font-size:14px;padding:11px 14px;background:#10171e;color:#eef3f8;border:1px solid #22303d;min-width:220px">
        <option value="trained">Norm, trained</option>
        <option value="untrained">Norm, untrained</option>
        <option value="joint">Joint pathology</option>
        <option value="spine">Spine pathology</option>
      </select>
      <button class="secondary" onclick="saveProfile()">SAVE PROFILE</button>
      <div class="muted" id="profileState">Used for synthetic EMG amplitude scaling.</div>
    </div>
  </section>

  <section class="panel">
    <div class="k">Live Trend</div>
    <div class="chartBox">
      <svg id="g" viewBox="0 0 600 180" preserveAspectRatio="none">
        <rect x="0" y="0" width="600" height="180" fill="#0d1217"/>
        <path id="pf" d="" fill="none" stroke="#57d6a4" stroke-width="2"/>
        <path id="pe" d="" fill="none" stroke="#7ab8ff" stroke-width="2"/>
      </svg>
    </div>
    <div class="legend">
      <span><span class="sw" style="background:#57d6a4"></span>Force N</span>
      <span><span class="sw" style="background:#7ab8ff"></span>Filt RMS uV</span>
    </div>
  </section>

  <section class="panel">
    <div class="k" style="margin-bottom:10px">Debug</div>
    <pre id="dbg">waiting...</pre>
  </section>
</div>
<script>
function t(i,v){var e=document.getElementById(i); if(e)e.textContent=v;}
function setRecUi(isRecording){
  var b=document.getElementById('recBtn');
  if(!b) return;
  b.textContent=isRecording?'REC ON':'REC';
  b.classList.toggle('recording', !!isRecording);
}
const hf=[], he=[];
function push(a,v){a.push(v); if(a.length>48)a.shift();}
function line(arr,maxv,h){
  if(arr.length<2)return '';
  var d='';
  for(var i=0;i<arr.length;i++){
    var x=(i/(arr.length-1))*600;
    var y=180-((arr[i]/maxv)*h)-8;
    d+=(i===0?'M':'L')+x.toFixed(1)+','+y.toFixed(1);
  }
  return d;
}
function draw(){
  var max=1, i;
  for(i=0;i<hf.length;i++){ if(hf[i]>max)max=hf[i]; }
  for(i=0;i<he.length;i++){ if(he[i]>max)max=he[i]; }
  document.getElementById('pf').setAttribute('d', line(hf,max,164));
  document.getElementById('pe').setAttribute('d', line(he,max,164));
}
async function u(){
  try{
    const r=await fetch('/api/data',{cache:'no-store'});
    const d=await r.json();
    t('s',(d.recording?'REC':'LIVE')+' | '+(d.weightCalib?'W OK':'W --')+' | '+(d.syntheticEmg?'E SIM':'E ADC')+' | '+(d.emgClipping?'CLIP':'OK'));
    setRecUi(d.recording);
    t('f', d.weightN.toFixed(2));
    t('p', d.peakWeightN.toFixed(2));
    t('r', Math.round(d.emgRMSUv));
    t('fr', Math.round(d.emgFilteredRMSUv));
    t('ep', Math.round(d.emgPeakUv));
    t('clip', d.emgClipping?'CLIP':'OK');
    var foot=document.getElementById('fl');
    if(foot && document.activeElement!==foot){ foot.value=(Math.round(d.footLengthCm*10)/10).toFixed(1); }
    var profile=document.getElementById('sp');
    if(profile && document.activeElement!==profile){ profile.value=d.syntheticProfileKey; }
    push(hf, d.weightN);
    push(he, d.emgFilteredRMSUv);
    draw();
    t('dbg',
      'EMG source: '+(d.syntheticEmg?'Synthetic force model':'ADC input')+'\n'+
      'Raw window: min='+d.emgWindowMin+' max='+d.emgWindowMax+' avg='+d.emgWindowAvg+'\n'+
      'Filt window: min='+d.emgFilteredWindowMin+' max='+d.emgFilteredWindowMax+' avg='+d.emgFilteredWindowAvg+'\n'+
      'Raw last='+d.emgLastValue+' | Filt last='+d.emgFilteredLastValue+'\n'+
      'Clip low='+d.emgClipLow+' | Clip high='+d.emgClipHigh
    );
  } catch(err) {
    t('s','data error');
  }
}
async function api(url){ try{ await fetch(url,{cache:'no-store'}); }catch(e){} }
async function startRec(){
  await api('/api/rec');
  setRecUi(true);
}
async function tare(){
  await api('/api/tare');
  t('tareState','Scale tared. Keep the platform unloaded.');
}
async function calib(){
  var w=document.getElementById('cw').value;
  if(!w || Number(w)<=0) return;
  await api('/api/cal?w='+encodeURIComponent(w));
  t('calState','Weight calibration stored for '+w+' g reference.');
}
async function saveFoot(){
  var fl=document.getElementById('fl').value;
  if(!fl || Number(fl)<=0) return;
  await api('/api/foot?cm='+encodeURIComponent(fl));
  t('footState','Foot length stored: '+Number(fl).toFixed(1)+' cm');
}
async function saveProfile(){
  var sp=document.getElementById('sp').value;
  await api('/api/profile?name='+encodeURIComponent(sp));
  t('profileState','Synthetic EMG profile stored: '+sp);
}
async function stopAndDownload(){
  await api('/api/stop');
  setRecUi(false);
  location.href='/api/export';
}
setInterval(u,600);
u();
</script>
</body>
</html>
)rawliteral";

  server.send_P(200, "text/html; charset=UTF-8", html);
}
String generateExcelReport() {
  String report = R"rawliteral(
<!DOCTYPE html>
<html lang="ru">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<style>
body{font-family:Segoe UI,Arial,sans-serif;margin:18px;color:#111}
h1{margin:0 0 10px}
.note{margin-bottom:14px;color:#555}
table{border-collapse:collapse;width:100%;font-size:12px}
th,td{border:1px solid #c7d0db;padding:6px 8px;text-align:right}
th{text-align:center;background:#eef3f8}
canvas{display:block;width:100%;max-width:960px;height:280px;border:1px solid #c7d0db;margin:14px 0 24px}
</style>
</head>
<body>
<h1>VESI + EMG Record Report</h1>
<div class="note">Файл можно открыть в браузере или Excel. Графики ниже построены по записанным значениям текущей сессии.</div>
<div class="note"><b>Foot length:</b> __FOOT_LENGTH__ cm<br><b>Synthetic EMG profile:</b> __PROFILE_LABEL__</div>
<canvas id="forceChart" width="960" height="280"></canvas>
<div class="note">Синий график: EMG RMS. Зеленый график: EMG Filt RMS. Оранжевый график: EMG Peak.</div>
<canvas id="emgChart" width="960" height="280"></canvas>
<table>
<thead>
<tr>
<th>Time (ms)</th>
<th>Weight (N)</th>
<th>Torque (kgf*cm)</th>
<th>EMG Peak (uV out)</th>
<th>EMG RMS (uV out)</th>
<th>EMG Raw (uV out)</th>
<th>EMG Filt Peak (uV out)</th>
<th>EMG Filt RMS (uV out)</th>
<th>EMG Filt Raw (uV out)</th>
</tr>
</thead>
<tbody>
)rawliteral";

  String timeJs = "[";
  String weightJs = "[";
  String emgRmsJs = "[";
  String emgFilteredRmsJs = "[";
  String emgPeakJs = "[";

  for (int i = 0; i < dataLogIndex; i++) {
    float weightN = weightToNewtons(dataLog[i].weight);
    float torqueKgfCm = weightToKgfCm(dataLog[i].weight);
    float peakUv = countsToMicrovolts(dataLog[i].emgPeak);
    float rmsUv = countsToMicrovolts(dataLog[i].emgRMS);
    float rawUv = countsToMicrovolts(dataLog[i].emgRaw);
    float filteredPeakUv = countsToMicrovolts(dataLog[i].emgFilteredPeak);
    float filteredRmsUv = countsToMicrovolts(dataLog[i].emgFilteredRMS);
    float filteredRawUv = countsToMicrovolts(dataLog[i].emgFilteredRaw);

    report += "<tr><td>" + String(dataLog[i].timestamp) + "</td>";
    report += "<td>" + String(weightN, 2) + "</td>";
    report += "<td>" + String(torqueKgfCm, 2) + "</td>";
    report += "<td>" + String(peakUv, 0) + "</td>";
    report += "<td>" + String(rmsUv, 0) + "</td>";
    report += "<td>" + String(rawUv, 0) + "</td>";
    report += "<td>" + String(filteredPeakUv, 0) + "</td>";
    report += "<td>" + String(filteredRmsUv, 0) + "</td>";
    report += "<td>" + String(filteredRawUv, 0) + "</td></tr>";

    if (i > 0) {
      timeJs += ",";
      weightJs += ",";
      emgRmsJs += ",";
      emgFilteredRmsJs += ",";
      emgPeakJs += ",";
    }
    timeJs += String(dataLog[i].timestamp);
    weightJs += String(weightN, 2);
    emgRmsJs += String(rmsUv, 0);
    emgFilteredRmsJs += String(filteredRmsUv, 0);
    emgPeakJs += String(peakUv, 0);
  }

  report += R"rawliteral(
</tbody>
</table>
<script>
const times=__TIMES__;
const weight=__WEIGHT__;
const emgRms=__EMGRMS__;
const emgFilteredRms=__EMGFILTRMS__;
const emgPeak=__EMGPEAK__;
function draw(canvasId, labels, series, colors, yLabel){
  const c=document.getElementById(canvasId);
  const ctx=c.getContext('2d');
  const w=c.width, h=c.height;
  const left=62, right=18, top=16, bottom=34;
  const pw=w-left-right, ph=h-top-bottom;
  ctx.clearRect(0,0,w,h);
  ctx.fillStyle='#fff'; ctx.fillRect(0,0,w,h);
  ctx.strokeStyle='#d7dee8'; ctx.lineWidth=1;
  ctx.strokeRect(left,top,pw,ph);
  let maxY=1;
  series.forEach(arr=>arr.forEach(v=>{ if(v>maxY) maxY=v; }));
  for(let i=0;i<=4;i++){
    const y=top + (ph/4)*i;
    ctx.strokeStyle='#e8edf3';
    ctx.beginPath(); ctx.moveTo(left,y); ctx.lineTo(left+pw,y); ctx.stroke();
    const val=(maxY*(4-i)/4).toFixed(0);
    ctx.fillStyle='#445'; ctx.font='12px Segoe UI';
    ctx.fillText(val,8,y+4);
  }
  ctx.fillText(yLabel,8,12);
  if(labels.length>1){
    const tickCount=5;
    for(let i=0;i<=tickCount;i++){
      const idx=Math.min(labels.length-1, Math.round((labels.length-1)*i/tickCount));
      const x=left + (pw*i/tickCount);
      ctx.strokeStyle='#d7dee8';
      ctx.beginPath(); ctx.moveTo(x,top+ph); ctx.lineTo(x,top+ph+5); ctx.stroke();
      ctx.fillStyle='#445';
      ctx.fillText((labels[idx]/1000).toFixed(1)+'s', x-10, h-10);
    }
  }
  series.forEach((arr,sidx)=>{
    if(arr.length<2) return;
    ctx.strokeStyle=colors[sidx];
    ctx.lineWidth=2;
    ctx.beginPath();
    arr.forEach((v,i)=>{
      const x=left + (pw * i / Math.max(1, arr.length-1));
      const y=top + ph - ((v/maxY) * ph);
      if(i===0) ctx.moveTo(x,y); else ctx.lineTo(x,y);
    });
    ctx.stroke();
  });
}
draw('forceChart', times, [weight], ['#1e9e68'], 'Force N');
draw('emgChart', times, [emgRms, emgFilteredRms, emgPeak], ['#2f7cf6', '#1e9e68', '#e0a11b'], 'EMG uV out');
</script>
</body>
</html>
)rawliteral";

  timeJs += "]";
  weightJs += "]";
  emgRmsJs += "]";
  emgFilteredRmsJs += "]";
  emgPeakJs += "]";

  report.replace("__TIMES__", timeJs);
  report.replace("__WEIGHT__", weightJs);
  report.replace("__EMGRMS__", emgRmsJs);
  report.replace("__EMGFILTRMS__", emgFilteredRmsJs);
  report.replace("__EMGPEAK__", emgPeakJs);
  report.replace("__FOOT_LENGTH__", String(footLengthCm, 1));
  report.replace("__PROFILE_LABEL__", String(syntheticProfileLabel(syntheticEmgProfile)));
  return report;
}

void handleEMGCalib() {
  // Запускаем перекалибровку EMG
  emgNeedsRecalibration = true;
  emgCalibrated = false;
  emgCalibrationStart = millis();
  emgBufferIndex = 0;
  emgSamplesCollected = 0;
  emgClipLowCount = 0;
  emgClipHighCount = 0;
  emgClippingDetected = false;
  Serial.println("\n⚠️  ПЕРЕКАЛИБРОВКА EMG: держите мышцу РАССЛАБЛЕННОЙ на 3 сек...");
  server.send(200, "text/plain", "OK");
}

void handleData() {
  char json[1600];
  snprintf(
    json,
    sizeof(json),
    "{\"weight\":%.2f,\"weightN\":%.2f,\"peakWeight\":%.2f,\"peakWeightN\":%.2f,"
    "\"emgPeak\":%d,\"emgPeakUv\":%.0f,\"emgRMS\":%.2f,\"emgRMSUv\":%.0f,"
    "\"emgFilteredPeak\":%d,\"emgFilteredPeakUv\":%.0f,\"emgFilteredRMS\":%.2f,\"emgFilteredRMSUv\":%.0f,"
    "\"emgAvg\":%d,\"emgAvgUv\":%.0f,\"emgMin\":%d,\"emgMinUv\":%.0f,\"emgMax\":%d,\"emgMaxUv\":%.0f,"
    "\"emgWindowMin\":%d,\"emgWindowMax\":%d,\"emgWindowAvg\":%d,"
    "\"emgFilteredWindowMin\":%d,\"emgFilteredWindowMax\":%d,\"emgFilteredWindowAvg\":%d,"
    "\"emgLastValue\":%d,\"emgLastValueUv\":%.0f,\"emgFilteredLastValue\":%d,\"emgFilteredLastValueUv\":%.0f,"
    "\"emgClipping\":%s,\"emgClipLow\":%lu,\"emgClipHigh\":%lu,"
    "\"recording\":%s,\"weightCalib\":%s,\"emgCalib\":%s,"
    "\"syntheticEmg\":%s,\"syntheticEmgMaxForceN\":%.1f,\"syntheticProfileKey\":\"%s\",\"syntheticProfileLabel\":\"%s\","
    "\"weightCalFactor\":%.6f,\"weightOffset\":%ld,\"footLengthCm\":%.1f,"
    "\"adcGainUvPerCount\":%.3f,\"adcOffsetUv\":%.1f}",
    displayedWeight,
    weightToNewtons(displayedWeight),
    peakWeight,
    weightToNewtons(peakWeight),
    emgPeak,
    countsToMicrovolts(emgPeak),
    emgRMS,
    countsToMicrovolts(emgRMS),
    emgFilteredPeak,
    countsToMicrovolts(emgFilteredPeak),
    emgFilteredRMS,
    countsToMicrovolts(emgFilteredRMS),
    emgAvg,
    countsToMicrovolts(emgAvg),
    emgMin,
    countsToMicrovolts(emgMin),
    emgMax,
    countsToMicrovolts(emgMax),
    emgCurrentWindowMin,
    emgCurrentWindowMax,
    emgCurrentWindowAvg,
    emgFilteredWindowMin,
    emgFilteredWindowMax,
    emgFilteredWindowAvg,
    emgLastValue,
    countsToMicrovolts(emgLastValue),
    emgFilteredLastValue,
    countsToMicrovolts(emgFilteredLastValue),
    emgClippingDetected ? "true" : "false",
    emgClipLowCount,
    emgClipHighCount,
    recordingMode ? "true" : "false",
    weightCalibrated ? "true" : "false",
    emgCalibrated ? "true" : "false",
    syntheticEmgEnabled ? "true" : "false",
    syntheticEmgMaxForceN,
    syntheticProfileKey(syntheticEmgProfile),
    syntheticProfileLabel(syntheticEmgProfile),
    calibrationFactor,
    offsetValue,
    footLengthCm,
    adcGainUvPerCount,
    adcOffsetUv
  );
  server.send(200, "application/json", json);
}

void handleTare() {
  tryTare();
  offsetValue = scale.get_offset();
  saveCalibrationToEEPROM();
  server.send(200, "text/plain", "OK");
}

void handleCal() {
  if (server.hasArg("w")) {
    float knownMass = server.arg("w").toFloat();
    if (knownMass > 0) {
      long reading = scale.read();
      calibrationFactor = (reading - scale.get_offset()) / knownMass;
      scale.set_scale(calibrationFactor);
      saveCalibrationToEEPROM();
      weightCalibrated = true;  // Устанавливаем флаг
      server.send(200, "text/plain", "OK");
      return;
    }
  }
  server.send(400, "text/plain", "BAD_WEIGHT");
}

void handleFoot() {
  if (server.hasArg("cm")) {
    float value = server.arg("cm").toFloat();
    if (value >= 1.0f && value <= 100.0f) {
      footLengthCm = value;
      saveCalibrationToEEPROM();
      server.send(200, "text/plain", "OK");
      return;
    }
  }
  server.send(400, "text/plain", "BAD_FOOT_LENGTH");
}

void handleProfile() {
  if (server.hasArg("name")) {
    int profile = parseSyntheticProfile(server.arg("name"));
    if (profile >= 0) {
      syntheticEmgProfile = profile;
      saveCalibrationToEEPROM();
      restartEmgAcquisition();
      server.send(200, "text/plain", "OK");
      return;
    }
  }
  server.send(400, "text/plain", "BAD_PROFILE");
}

void handleRec() {
  startRecording();
  server.send(200, "text/plain", "OK");
}

void handleStop() {
  stopRecording();
  server.send(200, "text/plain", "OK");
}

void handleExport() {
  stopRecording();

  String report = generateExcelReport();

  server.sendHeader("Content-Disposition", "attachment; filename=vesi_report.html");
  server.send(200, "text/html; charset=UTF-8", report);
  
  // Очищаем буфер для готовности к новой записи
  dataLogIndex = 0;
  Serial.println("✓ Буфер данных очищен");
}




