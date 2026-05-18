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
#include <ArduinoJson.h>

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
#define EMG_SIGNAL_PIN 4  // GPIO4 (ADC вход) - там раньше были данные!
// #define EMG_LO_PLUS 5     // GPIO5 - не используется (нет вывода на датчике)
// #define EMG_LO_MINUS 6    // GPIO6 - не используется (нет вывода на датчике)

// ============= SAMPLING SETTINGS =============
const unsigned long WEIGHT_UPDATE_INTERVAL_MS = 100;  // Обновление весов каждые 100мс
const unsigned long EMG_SAMPLE_INTERVAL_MS = 2;       // EMG sampled every 2ms (~500Hz)
const unsigned long DISPLAY_UPDATE_INTERVAL_MS = 100; // Экран обновляется каждые 100мс

// ============= CONSTANTS =============
const float GRAMS_PER_KG = 1000.0f;
const float GRAVITY = 9.80665f;  // Для перевода в ньютоны
const bool INVERT_POLARITY = true;

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
int emgBufferIndex = 0;
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
bool emgElectrodeDetached = false;

// ============= CALIBRATION =============
float calibrationFactor = 1.0f;
long offsetValue = 0;
bool weightCalibrated = false;  // Флаг калибровки весов
const int EEPROM_CALIBRATION_ADDR = 0;      // 4 байта для calibrationFactor
const int EEPROM_OFFSET_ADDR = 4;           // 4 байта для offsetValue

// ============= WI-FI & WEB SERVER =============
const char* ssid = "VESI-EMG-AP";
const char* password = "vesi1234";  // 8+ символов
WebServer server(80);
bool wifiInitialized = false;

// ============= DATA LOGGING =============
const int MAX_DATA_POINTS = 5000;  // Максимум точек данных
struct DataPoint {
  unsigned long timestamp;
  float weight;
  int emgPeak;
  float emgRMS;
  int emgRaw;
};
DataPoint dataLog[MAX_DATA_POINTS];
int dataLogIndex = 0;

// ============= MODE STATE =============
bool diagnosticMode = false;
bool recordingMode = false;
unsigned long recordingStartTime = 0;

void setup() {
  Serial.begin(115200);
  delay(1000);
  
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
  // analogSetAttenuation(ADC_11db);           // Попытка без этого
  pinMode(EMG_SIGNAL_PIN, INPUT);
  Serial.println("✓ ADC инициализирован");
  
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
  
  // Обработка веб-запросов
  if (wifiInitialized) {
    server.handleClient();
  }

  // Обработка команд через Serial
  if (Serial.available() > 0) {
    String command = Serial.readStringUntil('\n');
    command.trim();
    handleSerialCommand(command);
  }
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
    Serial.println(emgLastValue);
    
    // Сохраняем в буфер для экспорта
    if (dataLogIndex < MAX_DATA_POINTS) {
      dataLog[dataLogIndex].timestamp = millis() - recordingStartTime;
      dataLog[dataLogIndex].weight = displayedWeight;
      dataLog[dataLogIndex].emgPeak = emgPeak;
      dataLog[dataLogIndex].emgRMS = emgRMS;
      dataLog[dataLogIndex].emgRaw = emgLastValue;
      dataLogIndex++;
    }
  }
}

// ============= EMG FUNCTIONS =============
void collectEMGSample() {
  int rawValue = analogRead(EMG_SIGNAL_PIN);
  
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
      Serial.println("\n✓✓✓ Калибровка EMG завершена!");
      Serial.print("✓ FINAL Базовая линия: ");
      Serial.println((int)emgBaseline);
      Serial.println("Начинаем обработку сигнала...\n");
    }
  }
  
  // Обработка сигнала после калибровки: считаем как отклонение от baseline
  float signalACFiltered = (float)rawValue - emgBaseline;
  int signalAC = (int)abs(signalACFiltered);  // Берем абсолютное значение отклонения
  
  emgBuffer[emgBufferIndex] = signalAC;
  emgBufferIndex = (emgBufferIndex + 1) % EMG_BUFFER_SIZE;
  emgLastValue = signalAC;
  
  // Диагностика первых 10 сэмплов после калибровки
  static int diagCount = 0;
  if (diagCount < 10) {
    Serial.print("POST-CAL RAW: ");
    Serial.print(rawValue);
    Serial.print(" | AC: ");
    Serial.print(signalAC);
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
  // Calculate statistics for filled buffer
  int sum = 0;
  emgMin = 4095;
  emgMax = 0;
  long sumSquares = 0;

  for (int i = 0; i < EMG_BUFFER_SIZE; i++) {
    int val = emgBuffer[i];
    sum += val;
    
    if (val < emgMin) emgMin = val;
    if (val > emgMax) emgMax = val;
  }

  emgAvg = sum / EMG_BUFFER_SIZE;
  
  // RMS calculation - от среднего значения (правильный способ)
  for (int i = 0; i < EMG_BUFFER_SIZE; i++) {
    int diff = emgBuffer[i] - emgAvg;
    sumSquares += (long)diff * diff;
  }
  emgRMS = sqrt((float)sumSquares / EMG_BUFFER_SIZE);

  // Peak as max value in buffer
  emgPeak = emgMax;
  
  // Diagnostic output - показываем статистику
  static unsigned long lastDiagTime = 0;
  if (millis() - lastDiagTime > 500) {  // Каждые 500мс
    lastDiagTime = millis();
    Serial.print("EMG STATS: Min=");
    Serial.print(emgMin);
    Serial.print(" Max=");
    Serial.print(emgMax);
    Serial.print(" Avg=");
    Serial.print(emgAvg);
    Serial.print(" Peak=");
    Serial.print(emgPeak);
    Serial.print(" RMS=");
    Serial.println(emgRMS, 1);
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

  // Режим записи - показывать по-другому
  if (recordingMode) {
    char weightStr[16];
    
    // Строка 0: Заголовок + время
    display.setTextSize(1);
    display.setCursor(0, 0);
    display.print("REC");
    unsigned long elapsed = millis() - recordingStartTime;
    display.setCursor(100, 0);
    display.print(elapsed / 1000);
    display.println("s");

    // Строка 1-2: Вес в ньютонах (большой)
    display.setTextSize(2);
    display.setCursor(0, 10);
    float weightN = (displayedWeight / 1000.0f) * GRAVITY;  // Перевод в ньютоны
    dtostrf(weightN, 6, 2, weightStr);
    display.print(weightStr);
    display.println("N");

    // Строка 3: Пиковый вес в ньютонах
    display.setTextSize(1);
    display.setCursor(0, 26);
    display.print("Peak: ");
    float peakN = (peakWeight / 1000.0f) * GRAVITY;
    dtostrf(peakN, 5, 2, weightStr);
    display.println(weightStr);

    // Строка 4: EMG Peak и RMS
    display.setCursor(0, 34);
    display.print("EMG: ");
    display.print(emgPeak);
    display.print(" RMS:");
    dtostrf(emgRMS, 5, 0, weightStr);
    display.println(weightStr);
    
    // Строка 5: Avg EMG
    display.setCursor(0, 42);
    display.print("Avg: ");
    display.print(emgAvg);
    display.print(" Max: ");
    display.println(emgMax);
    
    // Строка 6: Статус
    display.setCursor(0, 50);
    display.print("Min:");
    display.print(emgMin);
    display.print(" Last:");
    display.println(emgLastValue);

  } else {
    // Нормальный режим отображения

    // EMG статус (сверху справа)
    display.setTextSize(1);
    display.setCursor(100, 0);
    if (emgElectrodeDetached) {
      display.println("EMG[OFF]");
    } else {
      display.println("EMG[OK]");
    }

    // ========== ВЕС в НЬЮТОНАХ (большие буквы) ==========
    display.setTextSize(3);
    float weightN = (displayedWeight / GRAMS_PER_KG) * GRAVITY;  // Перевод в ньютоны
    char weightStr[12];
    dtostrf(weightN, 6, 2, weightStr);
    
    int16_t x1, y1;
    uint16_t w, h;
    display.getTextBounds(weightStr, 0, 0, &x1, &y1, &w, &h);
    int16_t centerX = (SCREEN_WIDTH - w) / 2;
    display.setCursor(centerX, 12);
    display.println(weightStr);
    display.setTextSize(1);
    display.setCursor(centerX + 20, 30);
    display.println("N");

    // ========== СТАТИСТИКА ==========
    display.setTextSize(1);
    display.setCursor(0, 35);
    display.print("Peak N:");
    float peakN = (peakWeight / GRAMS_PER_KG) * GRAVITY;
    dtostrf(peakN, 5, 2, weightStr);
    display.println(weightStr);

    // EMG линия
    display.setCursor(0, 45);
    display.print("EMG raw:");
    display.print(emgLastValue);
    display.print(" Range:");
    display.println(emgMax - emgMin);

    display.setCursor(0, 54);
    display.print("EMG Peak:");
    display.print(emgPeak);
    display.print(" RMS:");
    dtostrf(emgRMS, 4, 0, weightStr);
    display.print(weightStr);
  }

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
  
  EEPROM.end();
  
  if (isnan(calibrationFactor) || fabs(calibrationFactor) < 1.0f || fabs(calibrationFactor) > 1000.0f) {
    calibrationFactor = 420.0f; // Default value
    Serial.println("⚠️  Калибровка загружена неверно, используется значение по умолчанию");
  }
  
  Serial.print("✓ Коэффициент калибровки: ");
  Serial.println(calibrationFactor);
  Serial.print("✓ Смещение offset: ");
  Serial.println(offsetValue);
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
    
  } else if (cmd == "EMG") {
    Serial.println("=== Статус EMG ===");
    Serial.print("Raw: ");
    Serial.println(emgLastValue);
    Serial.print("Min-Max: ");
    Serial.print(emgMin);
    Serial.print(" - ");
    Serial.println(emgMax);
    Serial.print("RMS: ");
    Serial.println(emgRMS);
    Serial.print("Electrode: ");
    Serial.println(emgElectrodeDetached ? "DETACHED ⚠️" : "OK ✓");
    
  } else {
    printHelp();
  }
}

void startRecording() {
  recordingMode = true;
  recordingStartTime = millis();
  peakWeight = 0;
  emgPeak = 0;
  Serial.println("\n╔════════════════════════════════════╗");
  Serial.println("║       ЗАПИСЬ НАЧАТА                 ║");
  Serial.println("║  time_ms, weight_g, emg_peak       ║");
  Serial.println("╚════════════════════════════════════╝");
}

void stopRecording() {
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
  Serial.println(emgPeak);
  
  Serial.print("Диапазон EMG:        ");
  Serial.print(emgMin);
  Serial.print(" - ");
  Serial.println(emgMax);
  
  Serial.print("RMS EMG:             ");
  Serial.println(emgRMS);
  
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
  Serial.println("║ DIAG    - Диагностический режим   ║");
  Serial.println("╚════════════════════════════════════╝\n");
}
// ============= WI-FI FUNCTIONS =============
void initWiFiAP() {
  WiFi.mode(WIFI_AP);
  WiFi.softAP(ssid, password);
  
  IPAddress IP = WiFi.softAPIP();
  Serial.println("\n✓ Wi-Fi Access Point создана!");
  Serial.print("SSID: ");
  Serial.println(ssid);
  Serial.print("IP: ");
  Serial.println(IP);
  Serial.print("Адрес: http://");
  Serial.println(IP);
  
  // Setup Web Server
  server.on("/", handleRoot);
  server.on("/api/data", handleData);
  server.on("/api/tare", handleTare);
  server.on("/api/cal", handleCal);
  server.on("/api/emg-calib", handleEMGCalib);  // Новый эндпоинт
  server.on("/api/rec", handleRec);
  server.on("/api/stop", handleStop);
  server.on("/api/export", handleExport);
  server.begin();
  
  wifiInitialized = true;
  Serial.println("✓ Веб-сервер запущен");
}

void handleRoot() {
  String html = "<html><head><meta charset='UTF-8'><style>";
  html += "body{font-family:Arial;background:#1a1a1a;color:#fff;padding:20px;margin:0}";
  html += ".container{max-width:600px;margin:0 auto}.status{background:#222;padding:15px;border-radius:8px;margin-bottom:15px;border-left:4px solid #0f0}";
  html += ".status p{margin:8px 0;font-size:16px}.value{color:#0f0;font-weight:bold}";
  html += ".grid{display:grid;grid-template-columns:1fr 1fr;gap:10px;margin:10px 0 15px}.card{background:#222;padding:12px;border-radius:5px;text-align:center}";
  html += ".card-title{font-size:12px;color:#aaa}.card-value{font-size:24px;color:#0f0;font-weight:bold;margin-top:5px}";
  html += ".recording{background:#333;padding:10px;border-radius:5px;text-align:center;color:#f00;display:none}";
  html += ".recording.active{animation:blink 1s infinite}@keyframes blink{0%,50%{opacity:1}51%,100%{opacity:0.5}}";
  html += ".buttons{display:grid;grid-template-columns:1fr 1fr;gap:10px;margin-bottom:15px}";
  html += "button{padding:12px;border:none;border-radius:5px;font-weight:bold;cursor:pointer;font-size:14px}";
  html += ".btn-tare,.btn-cal{background:#0066cc;color:white}.btn-rec{background:#00cc00;color:black}.btn-stop{background:#cc0000;color:white}";
  html += "button:hover{opacity:0.8}button:disabled{opacity:0.5;cursor:not-allowed}";
  html += ".input-group{margin-bottom:10px}.input-group input{width:100%;padding:8px;background:#333;color:#0f0;border:1px solid #0f0;border-radius:3px}";
  html += "</style></head><body><div class='container'>";
  html += "<div id='recording' class='recording'>REC</div>";
  html += "<div class='status'><p>Wt:<span class='value' id='w'>0.00</span>N</p>";
  html += "<p>Peak:<span class='value' id='p'>0.00</span>N</p></div>";
  html += "<div class='status'><p>EMG Peak:<span class='value' id='ep'>0</span></p>";
  html += "<p>RMS:<span class='value' id='er'>0.0</span></p>";
  html += "<p>Avg:<span class='value' id='ea'>0</span></p></div>";
  html += "<div class='grid'><div class='card'><div class='card-title'>Min</div>";
  html += "<div class='card-value' id='emin'>0</div></div>";
  html += "<div class='card'><div class='card-title'>Max</div>";
  html += "<div class='card-value' id='emax'>0</div></div>";
  html += "<div class='card'><div class='card-title'>Last</div>";
  html += "<div class='card-value' id='elv'>0</div></div>";
  html += "<div class='card'><div class='card-title'>Status</div>";
  html += "<div class='card-value' id='st'>OK</div></div></div>";
  html += "<div class='buttons'>";
  html += "<button class='btn-tare' onclick='tare()'>TARE</button>";
  html += "<button class='btn-cal' onclick='calib()'>CALIB</button>";
  html += "<button class='btn-rec' onclick='rec()' id='rb'>START REC</button>";
  html += "<button class='btn-stop' onclick='stop()' id='sb' disabled>STOP & DL</button>";
  html += "</div><div class='buttons'>";
  html += "<button class='btn-cal' onclick='emgCalib()'>EMG RE-CALIB</button>";
  html += "</div><div class='input-group'>";
  html += "<input type='number' id='cw' placeholder='Weight(g)' min='0' max='50000' value='1000'>";
  html += "</div></div><div style='background:#1a1a1a;padding:10px;margin-top:20px;border-top:1px solid #0f0;text-align:center;font-size:12px;color:#aaa'>";
  html += "<p id='status-msg'>INIT...</p></div>";
  html += "<script>";
  html += "const G=9.80665;";
  html += "function upd(){fetch('/api/data').then(r=>r.json()).then(d=>{";
  html += "document.getElementById('w').textContent=((d.weight/1000)*G).toFixed(2);";
  html += "document.getElementById('p').textContent=((d.peakWeight/1000)*G).toFixed(2);";
  html += "document.getElementById('ep').textContent=d.emgPeak;";
  html += "document.getElementById('er').textContent=d.emgRMS.toFixed(1);";
  html += "document.getElementById('ea').textContent=d.emgAvg;";
  html += "document.getElementById('emin').textContent=d.emgMin;";
  html += "document.getElementById('emax').textContent=d.emgMax;";
  html += "document.getElementById('elv').textContent=d.emgLastValue;";
  html += "document.getElementById('st').textContent=d.recording?'REC':'OK';";
  html += "let status='Ready: ';";
  html += "status+=(d.weightCalib?'[W✓]':'[W✗]')+' ';";
  html += "status+=(d.emgCalib?'[E✓]':'[E~]')+' ';";
  html += "status+=(d.recording?'Recording...':'Idle');";
  html += "document.getElementById('status-msg').textContent=status;";
  html += "const rd=document.getElementById('recording');";
  html += "if(d.recording){rd.style.display='block';rd.classList.add('active');}";
  html += "else{rd.style.display='none';rd.classList.remove('active');}";
  html += "});}";
  html += "function tare(){fetch('/api/tare');alert('OK');}";
  html += "function calib(){const w=document.getElementById('cw').value;";
  html += "if(!w||w<=0){alert('Need weight');return;}";
  html += "fetch('/api/cal?w='+w);alert('OK: '+w+'g');}";
  html += "function emgCalib(){fetch('/api/emg-calib');alert('EMG Re-calib started (3s)');}";
  html += "function rec(){fetch('/api/rec');document.getElementById('rb').disabled=true;";
  html += "document.getElementById('sb').disabled=false;}";
  html += "function stop(){window.location.href='/api/export';}";
  html += "setInterval(upd,100);upd();";
  html += "</script></body></html>";
  
  server.send(200, "text/html; charset=UTF-8", html);
}

void handleEMGCalib() {
  // Запускаем перекалибровку EMG
  emgNeedsRecalibration = true;
  emgCalibrated = false;
  emgCalibrationStart = millis();
  Serial.println("\n⚠️  ПЕРЕКАЛИБРОВКА EMG: держите мышцу РАССЛАБЛЕННОЙ на 3 сек...");
  server.send(200, "text/plain", "OK");
}

void handleData() {
  StaticJsonDocument<256> doc;
  
  doc["weight"] = (int)displayedWeight;  // в граммах
  doc["peakWeight"] = (int)peakWeight;
  doc["emgPeak"] = emgPeak;
  doc["emgRMS"] = emgRMS;
  doc["emgAvg"] = emgAvg;
  doc["emgMin"] = emgMin;
  doc["emgMax"] = emgMax;
  doc["emgLastValue"] = emgLastValue;
  doc["recording"] = recordingMode;
  doc["weightCalib"] = weightCalibrated;  // Статус калибровки весов
  doc["emgCalib"] = emgCalibrated;        // Статус калибровки EMG
  
  String json;
  serializeJson(doc, json);
  server.send(200, "application/json", json);
}

void handleTare() {
  tryTare();
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
    }
  }
}

void handleRec() {
  dataLogIndex = 0;  // Очищаем буфер
  recordingMode = true;
  recordingStartTime = millis();
  server.send(200, "text/plain", "OK");
}

void handleStop() {
  recordingMode = false;
  server.send(200, "text/plain", "OK");
}

void handleExport() {
  recordingMode = false;
  
  // Генерируем CSV с конвертацией веса в ньютоны
  String csv = "Time(ms),Weight(N),EMG_Peak,EMG_RMS,EMG_Raw\n";
  for (int i = 0; i < dataLogIndex; i++) {
    csv += String(dataLog[i].timestamp) + ",";
    // Конвертируем вес из граммов в ньютоны: (g/1000) * 9.80665
    float weightN = (dataLog[i].weight / 1000.0f) * GRAVITY;
    csv += String(weightN, 2) + ",";
    csv += String(dataLog[i].emgPeak) + ",";
    csv += String(dataLog[i].emgRMS, 2) + ",";
    csv += String(dataLog[i].emgRaw) + "\n";
  }
  
  server.sendHeader("Content-Disposition", "attachment; filename=data.csv");
  server.send(200, "text/csv", csv);
  
  // Очищаем буфер для готовности к новой записи
  dataLogIndex = 0;
  Serial.println("✓ Буфер данных очищен");
}