#include <ros.h>
#include <ArduinoHardware.h>
#include <std_msgs/Float64.h>
#include <rally_msgs/Pwm.h>

#include "Servo.h"

ros::NodeHandle  nh;

rally_msgs::Pwm raw_pwm_msg;
ros::Publisher raw_pwm_pub("raw_pwm", &raw_pwm_msg);

#define PWM_PUBLISH_RATE 10 //hz

// Assign your channel in pins
#define CHANNEL1_IN_PIN 2
#define CHANNEL2_IN_PIN 7
#define CHANNEL3_IN_PIN 3


// Assign your channel out pins
#define CHANNEL1_OUT_PIN 11
#define CHANNEL2_OUT_PIN 10
#define CHANNEL3_OUT_PIN 9


// Servo objects generate the signals expected by Electronic Speed Controllers and Servos
// We will use the objects to output the signals we read in
// this example code provides a straight pass through of the signal with no custom processing
Servo servoChannel1;
Servo servoChannel2;
Servo servoChannel3;


// These bit flags are set in bUpdateFlagsShared to indicate which
// channels have new signals
#define CHANNEL1_FLAG 1
#define CHANNEL2_FLAG 2
#define CHANNEL3_FLAG 4


// holds the update flags defined above
volatile uint32_t bUpdateFlagsShared;

// shared variables are updated by the ISR and read by loop.
// In loop we immediatley take local copies so that the ISR can keep ownership of the
// shared ones. To access these in loop
// we first turn interrupts off with noInterrupts
// we take a copy to use in loop and the turn interrupts back on
// as quickly as possible, this ensures that we are always able to receive new signals
volatile uint32_t unChannel1InShared;
volatile uint32_t unChannel2InShared;
volatile uint32_t unChannel3InShared;


void setup()
{
  digitalWrite(CHANNEL2_IN_PIN, HIGH);
  pciSetup(CHANNEL2_IN_PIN);
  //Serial.begin(115200);

  //Serial.println("multiChannels");

  // attach servo objects, these will generate the correct
  // pulses for driving Electronic speed controllers, servos or other devices
  // designed to interface directly with RC Receivers
  servoChannel1.attach(CHANNEL1_OUT_PIN);
  servoChannel2.attach(CHANNEL2_OUT_PIN);
  servoChannel3.attach(CHANNEL3_OUT_PIN);

  // attach the interrupts used to read the channels
  attachInterrupt(digitalPinToInterrupt(CHANNEL1_IN_PIN), calcChannel1, CHANGE);
  //attachInterrupt(digitalPinToInterrupt(CHANNEL2_IN_PIN), calcChannel2, CHANGE); //pin_change_ISR
  attachInterrupt(digitalPinToInterrupt(CHANNEL3_IN_PIN), calcChannel3, CHANGE);

  // for loop back test only, lets set each channel to a known value
  //  servoChannel1.writeMicroseconds(1100);
  //  servoChannel2.writeMicroseconds(1200);
  //  servoChannel3.writeMicroseconds(1300);
  nh.initNode();  // intialize ROS node
  nh.getHardware()->setBaud(57600);
  nh.advertise(raw_pwm_pub);  // start the publisher..can be used for debugging.

  while (!nh.connected())
  {
    nh.spinOnce();
  }
  nh.loginfo("LINOBASE CONNECTED");
  delay(1);
}

void loop()
{
  static unsigned long publish_pwm_time = 0;

  // create local variables to hold a local copies of the channel inputs
  // these are declared static so that thier values will be retained
  // between calls to loop.
  static uint32_t unChannel1In;
  static uint32_t unChannel2In;
  static uint32_t unChannel3In;


  // local copy of update flags
  static uint32_t bUpdateFlags;

  // check shared update flags to see if any channels have a new signal
  if (bUpdateFlagsShared)
  {
    noInterrupts(); // turn interrupts off quickly while we take local copies of the shared variables

    // take a local copy of which channels were updated in case we need to use this in the rest of loop
    bUpdateFlags = bUpdateFlagsShared;

    // in the current code, the shared values are always populated
    // so we could copy them without testing the flags
    // however in the future this could change, so lets
    // only copy when the flags tell us we can.

    if (bUpdateFlags & CHANNEL1_FLAG)
    {
      unChannel1In = unChannel1InShared;
    }

    if (bUpdateFlags & CHANNEL2_FLAG)
    {
      unChannel2In = unChannel2InShared;
    }

    if (bUpdateFlags & CHANNEL3_FLAG)
    {
      unChannel3In = unChannel3InShared;
    }

    // clear shared copy of updated flags as we have already taken the updates
    // we still have a local copy if we need to use it in bUpdateFlags
    bUpdateFlagsShared = 0;

    interrupts(); // we have local copies of the inputs, so now we can turn interrupts back on
    // as soon as interrupts are back on, we can no longer use the shared copies, the interrupt
    // service routines own these and could update them at any time. During the update, the
    // shared copies may contain junk. Luckily we have our local copies to work with :-)
  }

  // do any processing from here onwards
  // only use the local values unChannel1, unChannel2, unChannel3, unChannel4, unChannel5, unChannel6, unChannel7, unChannel8
  // variables unChannel1InShared, unChannel2InShared, etc are always owned by the
  // the interrupt routines and should not be used in loop

  if (bUpdateFlags & CHANNEL1_FLAG)
  {
    // remove the // from the line below to implement pass through updates to the servo on this channel -
    servoChannel1.writeMicroseconds(unChannel1In);
    //////Serial.println();
    //Serial.print("CH1: ");
    //Serial.print(unChannel1In);
    //Serial.print(",");
  }

  if (bUpdateFlags & CHANNEL2_FLAG)
  {
    // remove the // from the line below to implement pass through updates to the servo on this channel -
    servoChannel2.writeMicroseconds(unChannel2In);
    //Serial.print("CH2: ");
    //Serial.print(unChannel2In);
    //Serial.print(",");
  }

  if (bUpdateFlags & CHANNEL3_FLAG)
  {
    // remove the // from the line below to implement pass through updates to the servo on this channel -
    servoChannel3.writeMicroseconds(unChannel3In);
    //Serial.print("CH3: ");
    //Serial.print(unChannel3In);
    //Serial.print(",");
  }
  if ((millis() - publish_pwm_time) >= (1000 / PWM_PUBLISH_RATE))
  {
    publishPWM(unChannel1In, unChannel2In, unChannel3In);
    publish_pwm_time = millis();
  }
  bUpdateFlags = 0;

  nh.spinOnce();
}

void calcChannel1()
{
  static uint32_t ulStart;

  if (digitalRead(CHANNEL1_IN_PIN))
  {
    ulStart = micros();
  }
  else
  {
    unChannel1InShared = (uint32_t)(micros() - ulStart);
    bUpdateFlagsShared |= CHANNEL1_FLAG;
  }
}

void calcChannel2()
{
  static uint32_t ulStart;

  if (digitalRead(CHANNEL2_IN_PIN))
  {
    ulStart = micros();
  }
  else
  {
    unChannel2InShared = (uint32_t)(micros() - ulStart);
    bUpdateFlagsShared |= CHANNEL2_FLAG;
  }
}

void calcChannel3()
{
  static uint32_t ulStart;

  if (digitalRead(CHANNEL3_IN_PIN))
  {
    ulStart = micros();
  }
  else
  {
    unChannel3InShared = (uint32_t)(micros() - ulStart);
    bUpdateFlagsShared |= CHANNEL3_FLAG;
  }
}

void pciSetup(byte pin)
{
  *digitalPinToPCMSK(pin) |= bit (digitalPinToPCMSKbit(pin));  // enable pin
  PCIFR  |= bit (digitalPinToPCICRbit(pin)); // clear any outstanding interrupt
  PCICR  |= bit (digitalPinToPCICRbit(pin)); // enable interrupt for the group
}

ISR (PCINT2_vect) // handle pin change interrupt for D0 to D7 here
{
  //Serial.println("Ha");
  calcChannel2();
}

void publishPWM(uint32_t CH1, uint32_t CH2, uint32_t CH3)
{
  //pass accelerometer data to imu object
  raw_pwm_msg.steering = CH1;

  //pass gyroscope data to imu object
  raw_pwm_msg.throttle = CH2;

  //pass accelerometer data to imu object
  raw_pwm_msg.gear_shift = CH3;

  //publish raw_imu_msg object to ROS
  raw_pwm_pub.publish(&raw_pwm_msg);
}


