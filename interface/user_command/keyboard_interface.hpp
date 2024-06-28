#ifndef KEYBOARD_INTERFACE_HPP_
#define KEYBOARD_INTERFACE_HPP_

#include "user_command_interface.h"
#include "custom_types.h"
// #include <ncurses.h>
#include <cstdio>
#include <functional>
#include <termios.h>

#define AXIS_STEP 0.1

using namespace interface;
using namespace types;

class KeyboardInterface : public UserCommandInterface
{
private:
    UserCommand usr_cmd_;
    bool start_thread_flag_;
    std::thread kb_thread_;
    void ClipNumber(float &num, float low, float up){
        if(low > up) std::cerr << "error clip" << std::endl;
        if(num < low) num = low;
        if(num > up) num = up;
    }

    double GetCurrentTimeStamp(){
        static timespec startup_timestamp;
        timespec now_timestamp;
        if (startup_timestamp.tv_sec + startup_timestamp.tv_nsec == 0) {
            clock_gettime(CLOCK_MONOTONIC,&startup_timestamp);
        }
        clock_gettime(CLOCK_MONOTONIC,&now_timestamp);
        return (now_timestamp.tv_sec-startup_timestamp.tv_sec)*1e3 
            + (now_timestamp.tv_nsec-startup_timestamp.tv_nsec)/1e6;
    }

public:
    KeyboardInterface(){
        std::memset(&usr_cmd_, 0, sizeof(usr_cmd_));
    }
    ~KeyboardInterface(){}

    virtual void Start(){
        start_thread_flag_ = true;
        kb_thread_ = std::thread(std::bind(&KeyboardInterface::Run, this));
    }
    virtual void Stop(){
        start_thread_flag_ = false;
    }
    virtual UserCommand GetUserCommand(){return usr_cmd_;}

    virtual void SetMotionStateFeedback(const MotionStateFeedback& msfb){
        msfb_ = msfb;
    }


    void Run(){
        struct termios oldt, newt;
        tcgetattr(STDIN_FILENO, &oldt);
        newt = oldt;
        newt.c_lflag &= ~(ICANON | ECHO);
        tcsetattr(STDIN_FILENO, TCSANOW, &newt);

        char input;
        double forward_time_record = GetCurrentTimeStamp();
        double side_time_record = GetCurrentTimeStamp();
        double turnning_time_record = GetCurrentTimeStamp();
        std::cout << "Start Keyboard Listening" << std::endl;
        while (start_thread_flag_) {
            // std::cout << "time: " << current_time << " " << forward_time_record << std::endl;
            if(read(STDIN_FILENO, &input, 1) != -1){
                double current_time = GetCurrentTimeStamp();
                std::cout << "input: " << input << std::endl;
                if(input == 'e'){
                    usr_cmd_.target_mode = int(RobotMotionState::JointDamping);
                }
                switch(msfb_.current_state) {
                    case RobotMotionState::WaitingForStand:
                        if(input=='z'){
                            usr_cmd_.target_mode = int(RobotMotionState::StandingUp);
                        }
                    break;
                    case RobotMotionState::StandingUp:
                        if(input=='c'){
                            usr_cmd_.target_mode = int(RobotMotionState::RLControlMode);
                        }
                    break;
                    case RobotMotionState::RLControlMode:
                        if(input=='w') {
                            usr_cmd_.forward_vel_scale+=AXIS_STEP;
                            forward_time_record = current_time;
                        }  
                        else if(input=='s') {
                            usr_cmd_.forward_vel_scale-=AXIS_STEP;
                            forward_time_record = current_time;
                        }

                        if(input=='a') {
                            usr_cmd_.side_vel_scale+=AXIS_STEP;
                            side_time_record = current_time;
                        }
                        else if(input=='d') {
                            usr_cmd_.side_vel_scale-=AXIS_STEP;
                            side_time_record = current_time;
                        }
                        
                        if(input=='j') {
                            usr_cmd_.turnning_vel_scale+=AXIS_STEP;
                            turnning_time_record = current_time;
                        }
                        else if(input=='l') {
                            usr_cmd_.turnning_vel_scale-=AXIS_STEP;
                            turnning_time_record = current_time;
                        }

                        if(current_time - forward_time_record > 300.) usr_cmd_.forward_vel_scale = 0;
                        if(current_time - side_time_record > 300.) usr_cmd_.side_vel_scale = 0;
                        if(current_time - turnning_time_record > 300.) usr_cmd_.turnning_vel_scale = 0;

                        ClipNumber(usr_cmd_.forward_vel_scale, -1., 1.);
                        ClipNumber(usr_cmd_.side_vel_scale, -1., 1.);
                        ClipNumber(usr_cmd_.turnning_vel_scale, -1., 1.);
                    break;
                    default:
                        break;
                }
            }
            // std::this_thread::sleep_for(std::chrono::microseconds(100));
        }
        tcsetattr(STDIN_FILENO, TCSANOW, &oldt);
    }

};




#endif