/*
 * Snake Detection Hardware Alarm - Arduino Uno Sketch
 * Connect:
 *   - Active Buzzer Positive (+) -> Arduino Digital Pin 8
 *   - Active Buzzer Negative (-) -> Arduino GND
 *   - (Optional) Red LED (+) -> Digital Pin 9 (through 220-330 ohm resistor)
 */

const int BUZZER_PIN = 8;
const int LED_PIN = 9;

void setup() {
  pinMode(BUZZER_PIN, OUTPUT);
  pinMode(LED_PIN, OUTPUT);
  
  digitalWrite(BUZZER_PIN, LOW);
  digitalWrite(LED_PIN, LOW);
  
  Serial.begin(9600); // USB communication with Laptop
}

void loop() {
  if (Serial.available() > 0) {
    char signal = Serial.read();

    if (signal == 'V') {
      // VENOMOUS SNAKE: Fast, high-urgency siren pattern
      for (int i = 0; i < 5; i++) {
        digitalWrite(BUZZER_PIN, HIGH);
        digitalWrite(LED_PIN, HIGH);
        delay(80);
        digitalWrite(BUZZER_PIN, LOW);
        digitalWrite(LED_PIN, LOW);
        delay(80);
      }
    } 
    else if (signal == 'N') {
      // NON-VENOMOUS SNAKE: Slower caution beep
      digitalWrite(BUZZER_PIN, HIGH);
      digitalWrite(LED_PIN, HIGH);
      delay(400);
      digitalWrite(BUZZER_PIN, LOW);
      digitalWrite(LED_PIN, LOW);
    }
  }
}
