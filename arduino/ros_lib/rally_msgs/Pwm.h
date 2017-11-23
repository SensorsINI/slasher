#ifndef _ROS_rally_msgs_Pwm_h
#define _ROS_rally_msgs_Pwm_h

#include <stdint.h>
#include <string.h>
#include <stdlib.h>
#include "ros/msg.h"

namespace rally_msgs
{

  class Pwm : public ros::Msg
  {
    public:
      float steering;
      float throttle;
      float gear_shift;

    Pwm():
      steering(0),
      throttle(0),
      gear_shift(0)
    {
    }

    virtual int serialize(unsigned char *outbuffer) const
    {
      int offset = 0;
      union {
        float real;
        uint32_t base;
      } u_steering;
      u_steering.real = this->steering;
      *(outbuffer + offset + 0) = (u_steering.base >> (8 * 0)) & 0xFF;
      *(outbuffer + offset + 1) = (u_steering.base >> (8 * 1)) & 0xFF;
      *(outbuffer + offset + 2) = (u_steering.base >> (8 * 2)) & 0xFF;
      *(outbuffer + offset + 3) = (u_steering.base >> (8 * 3)) & 0xFF;
      offset += sizeof(this->steering);
      union {
        float real;
        uint32_t base;
      } u_throttle;
      u_throttle.real = this->throttle;
      *(outbuffer + offset + 0) = (u_throttle.base >> (8 * 0)) & 0xFF;
      *(outbuffer + offset + 1) = (u_throttle.base >> (8 * 1)) & 0xFF;
      *(outbuffer + offset + 2) = (u_throttle.base >> (8 * 2)) & 0xFF;
      *(outbuffer + offset + 3) = (u_throttle.base >> (8 * 3)) & 0xFF;
      offset += sizeof(this->throttle);
      union {
        float real;
        uint32_t base;
      } u_gear_shift;
      u_gear_shift.real = this->gear_shift;
      *(outbuffer + offset + 0) = (u_gear_shift.base >> (8 * 0)) & 0xFF;
      *(outbuffer + offset + 1) = (u_gear_shift.base >> (8 * 1)) & 0xFF;
      *(outbuffer + offset + 2) = (u_gear_shift.base >> (8 * 2)) & 0xFF;
      *(outbuffer + offset + 3) = (u_gear_shift.base >> (8 * 3)) & 0xFF;
      offset += sizeof(this->gear_shift);
      return offset;
    }

    virtual int deserialize(unsigned char *inbuffer)
    {
      int offset = 0;
      union {
        float real;
        uint32_t base;
      } u_steering;
      u_steering.base = 0;
      u_steering.base |= ((uint32_t) (*(inbuffer + offset + 0))) << (8 * 0);
      u_steering.base |= ((uint32_t) (*(inbuffer + offset + 1))) << (8 * 1);
      u_steering.base |= ((uint32_t) (*(inbuffer + offset + 2))) << (8 * 2);
      u_steering.base |= ((uint32_t) (*(inbuffer + offset + 3))) << (8 * 3);
      this->steering = u_steering.real;
      offset += sizeof(this->steering);
      union {
        float real;
        uint32_t base;
      } u_throttle;
      u_throttle.base = 0;
      u_throttle.base |= ((uint32_t) (*(inbuffer + offset + 0))) << (8 * 0);
      u_throttle.base |= ((uint32_t) (*(inbuffer + offset + 1))) << (8 * 1);
      u_throttle.base |= ((uint32_t) (*(inbuffer + offset + 2))) << (8 * 2);
      u_throttle.base |= ((uint32_t) (*(inbuffer + offset + 3))) << (8 * 3);
      this->throttle = u_throttle.real;
      offset += sizeof(this->throttle);
      union {
        float real;
        uint32_t base;
      } u_gear_shift;
      u_gear_shift.base = 0;
      u_gear_shift.base |= ((uint32_t) (*(inbuffer + offset + 0))) << (8 * 0);
      u_gear_shift.base |= ((uint32_t) (*(inbuffer + offset + 1))) << (8 * 1);
      u_gear_shift.base |= ((uint32_t) (*(inbuffer + offset + 2))) << (8 * 2);
      u_gear_shift.base |= ((uint32_t) (*(inbuffer + offset + 3))) << (8 * 3);
      this->gear_shift = u_gear_shift.real;
      offset += sizeof(this->gear_shift);
     return offset;
    }

    const char * getType(){ return "rally_msgs/Pwm"; };
    const char * getMD5(){ return "564790c10bd71c70ddf5f2df84f30bfb"; };

  };

}
#endif