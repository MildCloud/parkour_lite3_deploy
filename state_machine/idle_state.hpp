#ifndef IDLE_STATE_HPP_
#define IDLE_STATE_HPP_


#include "state_base.h"

class IdleState : public StateBase{
private:
    bool joint_normal_flag_ = false, imu_normal_flag_ = false;
    bool JointDataNormalCheck(){
        VecXf joint_pos = ri_ptr_->GetJointPosition();
        VecXf joint_vel = ri_ptr_->GetJointVelocity();
        VecXf joint_pos_lower(12), joint_pos_upper(12);
        Vec3f fl_lower = cp_ptr_->fl_joint_lower_;
        Vec3f fl_upper = cp_ptr_->fl_joint_upper_;
        Vec3f fr_lower(-fl_upper(0), fl_lower(1), fl_lower(2));
        Vec3f fr_upper(-fl_lower(0), fl_upper(1), fl_upper(2));
        joint_pos_lower << fl_lower, fr_lower, fl_lower, fr_lower;
        joint_pos_upper << fl_upper, fr_upper, fl_upper, fr_upper;
        for(int i=0;i<12;++i){
            if(joint_pos(i) > joint_pos_upper(i)+0.1 || joint_pos(i) < joint_pos_lower(i)-0.1) {
                std::cout << "joint pos " << i << " : " << joint_pos(i) << " | " 
                                                        << joint_pos_lower(i) << " " << joint_pos_upper(i) << std::endl;
                return false;
            }
            if(fabs(joint_vel(i)) > cp_ptr_->joint_vel_limit_(i%3) + 0.1) {
                std::cout << "joint vel " << i << " : " << joint_vel(i) << " | " << cp_ptr_->joint_vel_limit_(i%3) << std::endl;
                return false;
            }
        }
        return true;
    }
    bool ImuDataNormalCheck(){
        Vec3f rpy = ri_ptr_->GetImuRpy();
        Vec3f acc = ri_ptr_->GetImuAcc();
        Vec3f omg = ri_ptr_->GetImuOmega();
        for(int i=0;i<3;++i){
            if(fabs(rpy(i)) > M_PI){
                std::cout << "rpy " << i << " : " << rpy(i) << std::endl;
                return false;
            } 
            if(fabs(omg(i)) > M_PI/4.){
                std::cout << "omg " << i << " : " << omg(i) << std::endl;
                return false;
            }
        }
        if(acc.norm() < 0.5*gravity || acc.norm() > 1.5*gravity){
            std::cout << "acc " << " : " << acc.transpose() << std::endl;
            return false;
        }
        return true;
    }


public:
    IdleState(const RobotType& robot_type, const std::string& state_name, 
        std::shared_ptr<ControllerData> data_ptr):StateBase(robot_type, state_name, data_ptr){
        }
    ~IdleState(){}

    virtual void OnEnter() {
        StateBase::msfb_.current_state = RobotMotionState::WaitingForStand;
        uc_ptr_->SetMotionStateFeedback(StateBase::msfb_);
    };
    virtual void OnExit() {
    }
    virtual void Run() {
        joint_normal_flag_ = JointDataNormalCheck();
        imu_normal_flag_ = ImuDataNormalCheck();
        MatXf cmd = MatXf::Zero(12, 5);
        ri_ptr_->SetJointCommand(cmd);
    }
    virtual bool LoseControlJudge() {
        return false;
    }
    virtual StateName GetNextStateName() {
        if(!joint_normal_flag_ || !imu_normal_flag_) {
            std::cout << "joint status: " << joint_normal_flag_ << " | imu status: " << imu_normal_flag_ << std::endl;
            return StateName::kIdle;
        }
        
        if(uc_ptr_->GetUserCommand().target_mode == int(RobotMotionState::StandingUp)) return StateName::kStandUp;
        return StateName::kIdle;
    }
};




#endif