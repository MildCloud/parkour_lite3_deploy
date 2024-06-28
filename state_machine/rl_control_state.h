#ifndef RL_CONTROL_STATE_H_
#define RL_CONTROL_STATE_H_

#include "state_base.h"
#include "jueying_policy_runner.h"
#include "depth_sensor.h"

class RLControlState : public StateBase
{
private:
    RobotBasicState rbs_;
    int state_run_cnt_;
    std::shared_ptr<JueyingPolicyRunner> current_policy_; 
    std::shared_ptr<JueyingPolicyRunner> common_policy_, speed_policy_, tumbler_stand_policy_, tumbler_forward_policy_, parkour_policy_;
    std::shared_ptr<DepthSensor> dp_sensor_;
    std::thread run_policy_thread_;
    bool start_flag_ = true;

    void CommonPolicyInitialize();
    void SpeedPolicyInitialize();
    void TumblerStandPolicyInitialize();
    void TumblerForwardPolicyInitialize();
    void ParkourPolicyInitialize();

    Mat3f RpyToRm(const Vec3f &rpy);
    VecXf GetFootPos(const VecXf& joint_pos);
    VecXf ConventionShift(const VecXf& x);

    void UpdateRobotObservation(){
        rbs_.base_rpy     = ri_ptr_->GetImuRpy();
        rbs_.base_rot_mat = RpyToRm(rbs_.base_rpy);
        rbs_.base_omega   = ri_ptr_->GetImuOmega(); // which frame
        rbs_.base_acc     = ri_ptr_->GetImuAcc(); // which frame
        rbs_.joint_pos    = ri_ptr_->GetJointPosition();
        rbs_.joint_vel    = ri_ptr_->GetJointVelocity();
        rbs_.joint_tau    = ri_ptr_->GetJointTorque();
        rbs_.cmd_vel_normlized      = Vec3f(uc_ptr_->GetUserCommand().forward_vel_scale, 
                                    uc_ptr_->GetUserCommand().side_vel_scale, 
                                    uc_ptr_->GetUserCommand().turnning_vel_scale);
        // rbs_.foot_pos     = GetFootPos(rbs_.joint_pos);
    }

    void PolicyRunner(){
        int run_cnt_record = -1;
        while (start_flag_){
            
            if(state_run_cnt_%current_policy_->decimation_ == 0 && state_run_cnt_ != run_cnt_record){
                timespec start_timestamp, end_timestamp;
                clock_gettime(CLOCK_MONOTONIC,&start_timestamp);
                auto ra = current_policy_->GetPolicyOutput(rbs_);
                MatXf res = ra.ConvertToMat();
                // std::cout << "res" << res << std::endl;
                ri_ptr_->SetJointCommand(res);
                run_cnt_record = state_run_cnt_;
                clock_gettime(CLOCK_MONOTONIC,&end_timestamp);
                std::cout << "cost_time:  " << (end_timestamp.tv_sec-start_timestamp.tv_sec)*1e3 
                    + (end_timestamp.tv_nsec-start_timestamp.tv_nsec)/1e6 << " ms\n";
            }
            std::this_thread::sleep_for(std::chrono::microseconds(100)); // TODO modify the control freq here
        }
    }

public:
    RLControlState(const RobotType& robot_type, const std::string& state_name, 
        std::shared_ptr<ControllerData> data_ptr):StateBase(robot_type, state_name, data_ptr){
            dp_sensor_ = std::make_shared<DepthSensor>();

            CommonPolicyInitialize();

            if(robot_type==RobotType::P50){
                SpeedPolicyInitialize();
            }else if(robot_type==RobotType::Lite3){
                TumblerStandPolicyInitialize();
            }

            ParkourPolicyInitialize();
            current_policy_ = parkour_policy_;
            std::memset(&rbs_, 0, sizeof(rbs_));
        }
    ~RLControlState(){}

    virtual void OnEnter() {
        state_run_cnt_ = -1;
        start_flag_ = true;
        run_policy_thread_ = std::thread(std::bind(&RLControlState::PolicyRunner, this));
        dp_sensor_->start();
        current_policy_->OnEnter(); // Move above the run_policy_thread?
        StateBase::msfb_.current_state = RobotMotionState::RLControlMode;
        uc_ptr_->SetMotionStateFeedback(StateBase::msfb_);
    };

    virtual void OnExit() { 
        start_flag_ = false;
        run_policy_thread_.join();
        state_run_cnt_ = -1;
    }

    virtual void Run() {
        UpdateRobotObservation();
        state_run_cnt_++;
    }

    virtual bool LoseControlJudge() {
        if(uc_ptr_->GetUserCommand().target_mode == int(RobotMotionState::JointDamping)) return true;
        return false;
    }

    virtual StateName GetNextStateName() {
        return StateName::kRLControl;
    }
};


#endif