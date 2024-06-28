#ifndef STANDUP_STATE_HPP_
#define STANDUP_STATE_HPP_

#include "state_base.h"

class StandUpState : public StateBase{
private:
    VecXf init_joint_pos_, init_joint_vel_, current_joint_pos_, current_joint_vel_;
    float time_stamp_record_, run_time_;
    VecXf goal_joint_pos_, kp_, kd_;
    MatXf joint_cmd_;
    float stand_duration_ = 2.;

    void GetRobotJointValue(){
        current_joint_pos_ = ri_ptr_->GetJointPosition();
        current_joint_vel_ = ri_ptr_->GetJointVelocity();
        run_time_ = ri_ptr_->GetInterfaceTimeStamp();
    }

    void RecordJointData(){
        init_joint_pos_ = current_joint_pos_;
        init_joint_vel_ = current_joint_vel_;
        time_stamp_record_ = run_time_;
    }

    float GetCubicSplinePos(float x0, float v0, float xf, float vf, float t, float T){
        if(t >= T) return xf;
        float a, b, c, d;
        d = x0;
        c = v0;
        a = (vf*T - 2*xf + v0*T + 2*x0) / pow(T, 3);
        b = (3*xf - vf*T - 2*v0*T - 3*x0) / pow(T, 2);
        return a*pow(t, 3)+b*pow(t, 2)+c*t+d;
    }
    float GetCubicSplineVel(float x0, float v0, float xf, float vf, float t, float T){
        if(t >= T) return 0;
        float a, b, c;
        c = v0;
        a = (vf*T - 2*xf + v0*T + 2*x0) / pow(T, 3);
        b = (3*xf - vf*T - 2*v0*T - 3*x0) / pow(T, 2);
        return 3.*a*pow(t, 2) + 2.*b*t + c;
    }

public:
    StandUpState(const RobotType& robot_type, const std::string& state_name, 
        std::shared_ptr<ControllerData> data_ptr):StateBase(robot_type, state_name, data_ptr){
            goal_joint_pos_ = Vec3f(0., -1, 1.8).replicate(4, 1);
            kp_ = cp_ptr_->swing_leg_kp_.replicate(4, 1);
            kd_ = cp_ptr_->swing_leg_kd_.replicate(4, 1);
            joint_cmd_ = MatXf::Zero(12, 5);
            joint_cmd_.col(0) = kp_;
            joint_cmd_.col(2) = kd_;
            stand_duration_ = cp_ptr_->stand_duration_;
        }
    ~StandUpState(){}


    virtual void OnEnter() {
        GetRobotJointValue();
        RecordJointData();
        StateBase::msfb_.current_state = RobotMotionState::StandingUp;
        uc_ptr_->SetMotionStateFeedback(StateBase::msfb_);
    };
    virtual void OnExit() {
    }
    virtual void Run() {
        GetRobotJointValue();
        VecXf planning_joint_pos(current_joint_pos_.rows());
        VecXf planning_joint_vel(current_joint_pos_.rows());
        for(int i=0;i<current_joint_pos_.rows();++i){
            planning_joint_pos(i) = GetCubicSplinePos(init_joint_pos_(i), init_joint_vel_(i), goal_joint_pos_(i), 0, 
                                            run_time_ - time_stamp_record_, stand_duration_);
            planning_joint_vel(i) = GetCubicSplineVel(init_joint_pos_(i), init_joint_vel_(i), goal_joint_pos_(i), 0, 
                                            run_time_ - time_stamp_record_, stand_duration_);
        }
        joint_cmd_.col(1) = planning_joint_pos;
        joint_cmd_.col(3) = planning_joint_vel;
        ri_ptr_->SetJointCommand(joint_cmd_);
    }
    virtual bool LoseControlJudge() {
        if(uc_ptr_->GetUserCommand().target_mode == int(RobotMotionState::JointDamping)) return true;
        return false;
    }
    virtual StateName GetNextStateName() {
        if(run_time_ - time_stamp_record_ <= stand_duration_){
            return StateName::kStandUp;
        }else{
            if(uc_ptr_->GetUserCommand().target_mode == int(RobotMotionState::RLControlMode)){
                return StateName::kRLControl;
            }
        }
        return StateName::kStandUp;
    }
};



#endif