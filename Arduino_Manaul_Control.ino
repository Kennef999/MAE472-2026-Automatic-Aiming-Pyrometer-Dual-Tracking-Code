#include <Wire.h>
#include <Adafruit_MotorShield.h>

// Motor shield
Adafruit_MotorShield AFMS = Adafruit_MotorShield();
Adafruit_DCMotor *motor = AFMS.getMotor(1);

// Joystick
const int JOY_Y_PIN = A0;

// Joystick tuning
const int DEADZONE = 60;
const int MAX_SPEED = 255;

// Auto control safety
const unsigned long AUTO_TIMEOUT_MS = 300;
unsigned long lastAutoMs = 0;
int autoCmd = 0;   // -255..255 from Pi

int clampInt(int v, int lo, int hi) {
  if (v < lo) return lo;
  if (v > hi) return hi;
  return v;
}

void stopMotor() {
  motor->setSpeed(0);
  motor->run(RELEASE);
}

// Signed motor control (-255..255)
void driveSigned(int cmd) {
  cmd = constrain(cmd, -255, 255);

  if (cmd == 0) {
    stopMotor();
    return;
  }

  motor->setSpeed(abs(cmd));

  // SIGN CONVENTION:
  // cmd > 0 => FORWARD
  // cmd < 0 => BACKWARD
  // If your actuator moves the wrong way, swap FORWARD/BACKWARD below.
  if (cmd > 0) motor->run(FORWARD);
  else motor->run(BACKWARD);
}

void readAutoSerial() {
  while (Serial.available()) {
    String line = Serial.readStringUntil('\n');
    line.trim();
    if (line.length() == 0) return;

    if (line == "STOP") {
      autoCmd = 0;
      lastAutoMs = millis();
      return;
    }

    if (line.startsWith("DY:")) {
      int v = line.substring(3).toInt(); // expects -255..255
      autoCmd = constrain(v, -255, 255);
      lastAutoMs = millis();
      return;
    }
  }
}

void setup() {
  Serial.begin(115200);

  if (!AFMS.begin()) {
    Serial.println("Could not find Motor Shield. Check I2C wiring/address.");
    while (1) {}
  }

  stopMotor();
  Serial.println("Ready. Joystick overrides. Auto uses DY:<-255..255> when centered.");
}

void loop() {
  // 1) Read any Pi commands
  readAutoSerial();

  // 2) Manual control (joystick override)
  int joy = analogRead(JOY_Y_PIN);
  int delta = joy - 512;

  if (abs(delta) >= DEADZONE) {
    int mag = abs(delta) - DEADZONE;
    mag = clampInt(mag, 0, 512 - DEADZONE);

    int speed = map(mag, 0, 512 - DEADZONE, 0, MAX_SPEED);
    speed = clampInt(speed, 0, MAX_SPEED);

    motor->setSpeed(speed);

    // If joystick direction feels reversed, swap FORWARD/BACKWARD here.
    if (delta > 0) motor->run(FORWARD);
    else motor->run(BACKWARD);

    delay(10);
    return;
  }

  // 3) Auto mode (only when joystick centered)
  if (millis() - lastAutoMs > AUTO_TIMEOUT_MS) {
    stopMotor();
    delay(10);
    return;
  }

  driveSigned(autoCmd);
  delay(10);
}
