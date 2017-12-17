#include <ros.h>                // header files sourced from  Step 3
#include <std_msgs/Bool.h>
#include <std_msgs/String.h>
#include <std_msgs/Int32.h>
#include <std_msgs/Empty.h>
//#include <race/drive_values.h>
#include <rally_msgs/Pwm.h>
#include "Servo.h"
ros::NodeHandle  nh;

Servo servoChannel1;
Servo servoChannel2;

// Assign your channel in pins
#define CHANNEL1_IN_PIN 2 // throttle
#define CHANNEL2_IN_PIN 7 // steering

// Assign your channel out pins
#define CHANNEL1_OUT_PIN 11
#define CHANNEL2_OUT_PIN 10

// These bit flags are set in bUpdateFlagsShared to indicate which
// channels have new signals
#define CHANNEL1_FLAG 1
#define CHANNEL2_FLAG 2

// holds the update flags defined above
volatile uint32_t bUpdateFlagsShared;

// shared variables are updated by the ISR and read by loop.
volatile uint32_t unChannel1InShared;
volatile uint32_t unChannel2InShared;

boolean flagStop = false;     // These values were cacluated for the specific Teensy microcontroller using an oscilloscope.
boolean flagTakeover = false;
//int pwm_center_value = 9830;  //  15% duty cycle - corresponds to zero velocity, zero steering
//int pwm_lowerlimit = 6554;    //  10% duty cycle - corresponds to max reverse velocity, extreme left steering
//int pwm_upperlimit = 13108;   //  20% duty cycle - corresponds to max forward velocity, extreme right steering
int pwm_center_value = 1500;  //  15% duty cycle - corresponds to zero velocity, zero steering
int pwm_lowerlimit = 1000;    //  10% duty cycle - corresponds to max reverse velocity, extreme left steering
int pwm_upperlimit = 2000;   //  20% duty cycle - corresponds to max forward velocity, extreme right steering


std_msgs::Int32 str_msg;          // creater a ROS Publisher called chatter of type str_msg
ros::Publisher chatter("chatter", &str_msg);

rally_msgs::Pwm raw_pwm_msg;
ros::Publisher raw_pwm_pub("raw_pwm", &raw_pwm_msg);

#define PWM_PUBLISH_RATE 5 //hz

int kill_pin = 2;     // This is the GPIO pin for emergency stopping.
unsigned long duration = 0;

void messageDrive( const rally_msgs::Pwm& pwm )
{
  //  Serial.print("Pwm drive : ");
  //  Serial.println(pwm.pwm_drive);
  //  Serial.print("Pwm angle : ");
  //  Serial.println(pwm.pwm_angle);

  if (flagStop == false)
  {
    str_msg.data = pwm.throttle;
    chatter.publish( &str_msg );

    if (pwm.throttle < pwm_lowerlimit) // Pin 5 is connected to the ESC..dive motor
    {
      servoChannel1.writeMicroseconds(pwm_lowerlimit);    //  Safety lower limit
    }
    else if (pwm.throttle > pwm_upperlimit)
    {
      servoChannel1.writeMicroseconds(pwm_upperlimit);    //  Safety upper limit
    }
    else
    {
      servoChannel1.writeMicroseconds(pwm.throttle);     //  Incoming data
    }


    if (pwm.steering < pwm_lowerlimit) // Pin 6 is connected to the steering servo.
    {
      servoChannel2.writeMicroseconds(pwm_lowerlimit);    //  Safety lower limit
    }
    else if (pwm.steering > pwm_upperlimit)
    {
      servoChannel2.writeMicroseconds(pwm_upperlimit);    //  Safety upper limit
    }
    else
    {
      servoChannel2.writeMicroseconds(pwm.steering);     //  Incoming data
    }

  }
  //  else
  //  {
  //    servoChannel1.writeMicroseconds(pwm_center_value);
  //    servoChannel2.writeMicroseconds(pwm_center_value);
  //  }
}

void messageEmergencyStop( const std_msgs::Bool& flag )
{
  flagStop = flag.data;
  if (flagStop == true)
  {
    servoChannel1.writeMicroseconds(pwm_center_value);
    servoChannel2.writeMicroseconds(pwm_center_value);
  }
}

void messageResumeAuto( const std_msgs::Bool& flag )
{
  flagTakeover = !flag.data;
  flagStop = !flag.data;
  if (flagTakeover == false)
  {
    nh.loginfo("Resume Autonomous mode");
  }
}
ros::Subscriber<rally_msgs::Pwm> sub_drive("drive_pwm", &messageDrive );   // Subscribe to drive_pwm topic sent by Jetson
ros::Subscriber<std_msgs::Bool> sub_stop("eStop", &messageEmergencyStop );  // Subscribe to estop topic sent by Jetson
ros::Subscriber<std_msgs::Bool> sub_resumeAuto("resumeAuto", &messageResumeAuto );  // Subscribe to estop topic sent by Jetson

void setup() {
  // listen
  digitalWrite(CHANNEL2_IN_PIN, HIGH);
  pciSetup(CHANNEL2_IN_PIN);
  attachInterrupt(digitalPinToInterrupt(CHANNEL1_IN_PIN), calcChannel1, CHANGE);

  // Need to produce PWM signals so we need to setup the PWM registers. This setup happens next.
  //Frequency(5, 100); //  freq at which PWM signals is generated at pin 5.
  //analogWriteFrequency(6, 100);
  //analogWriteResolution(16); // Resolution for the PWM signal
  servoChannel1.attach(11);
  servoChannel2.attach(10);
  servoChannel1.writeMicroseconds(pwm_center_value); // Setup zero velocity and steering.
  servoChannel2.writeMicroseconds(pwm_center_value);
  pinMode(13, OUTPUT); // Teensy's onboard LED pin.
  digitalWrite(13, HIGH); // Setup LED.
  //  pinMode(kill_pin,INPUT); // Set emergency pin to accept inputs.
  //  digitalWrite(2,LOW);

  nh.initNode();  // intialize ROS node
  nh.subscribe(sub_drive); // start the subscribers.
  nh.subscribe(sub_stop);
  nh.subscribe(sub_resumeAuto);
  nh.advertise(raw_pwm_pub);
  nh.advertise(chatter);  // start the publisher..can be used for debugging.
  while (!nh.connected())
  {
    nh.spinOnce();
  }
  nh.loginfo("LINOBASE CONNECTED");
  delay(1);
}

void loop() {
  static unsigned long publish_pwm_time = 0;
  // create local variables to hold a local copies of the channel inputs
  // these are declared static so that thier values will be retained
  // between calls to loop.
  static uint32_t unChannel1In;
  static uint32_t unChannel2In;


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
    if (unChannel1In < 1300)
    {
      if (flagStop == false)
      {
        flagStop = true;
        flagTakeover = true;
        nh.loginfo("human take over");
      }
    }
    if (flagTakeover == true)
    {
      //nh.loginfo("overwrite");
      servoChannel1.writeMicroseconds(unChannel1In);
    }
    //servoChannel1.writeMicroseconds(unChannel1In);
    //////Serial.println();
    //Serial.print("CH1: ");
    //Serial.print(unChannel1In);
    //Serial.print(",");
  }

  if (bUpdateFlags & CHANNEL2_FLAG)
  {
    if (unChannel2In > 1700)
    {
      if (flagStop == false)
      {
        flagStop = true;
        flagTakeover = true;
        nh.loginfo("human take over");
      }
    }
    if (flagTakeover == true)
    {
      //nh.loginfo("overwrite");
      servoChannel2.writeMicroseconds(unChannel2In);
    }
    // remove the // from the line below to implement pass through updates to the servo on this channel -
    //servoChannel2.writeMicroseconds(unChannel2In);
    //Serial.print("CH2: ");
    //Serial.print(unChannel2In);
    //Serial.print(",");
  }

  if ((millis() - publish_pwm_time) >= (1000 / PWM_PUBLISH_RATE))
  {
    publishPWM(unChannel1In, unChannel2In);
    publish_pwm_time = millis();
  }
  bUpdateFlags = 0;

  nh.spinOnce();
  //  duration = pulseIn(kill_pin, HIGH, 30000);  // continuously monitor the kill pin.
  //  while(duration > 1900) // stop if kill pin activated..setup everything to zero.
  //  {
  //    duration = pulseIn(kill_pin, HIGH, 30000);
  //    servoChannel1.writeMicroseconds(pwm_center_value);
  //    servoChannel2.writeMicroseconds(pwm_center_value);
  //  }
  // put your main code here, to run repeatedly:
  /*
    if(Serial.available())
    {
    int spd = Serial.read();
    if(spd>127) {
      spd = spd-128;
      spd = map(spd,0,100,410,820);
      servoChannel1.writeMicroseconds(5,spd);
    }
    else {
      //angle servo
      spd = map(spd,0,100,410,820);
      servoChannel1.writeMicroseconds(6,spd);
    }

    }
  */
}

//Listen

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

void publishPWM(uint32_t CH1, uint32_t CH2)
{
  //pass accelerometer data to imu object
  raw_pwm_msg.steering = CH2;

  //pass gyroscope data to imu object
  raw_pwm_msg.throttle = CH1;

  //pass accelerometer data to imu object
  raw_pwm_msg.gear_shift = 0;

  //publish raw_imu_msg object to ROS
  raw_pwm_pub.publish(&raw_pwm_msg);
}

