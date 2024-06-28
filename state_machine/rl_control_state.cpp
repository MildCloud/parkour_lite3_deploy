#include "rl_control_state.h"

void RLControlState::CommonPolicyInitialize(){
    const std::string p = cp_ptr_->common_policy_path_;
    common_policy_ = std::make_shared<JueyingPolicyRunner>("common", p, 45, 5);
    common_policy_->SetPDGain(cp_ptr_->common_policy_p_gain_, cp_ptr_->common_policy_d_gain_);
    common_policy_->SetTorqueLimit(cp_ptr_->torque_limit_);
    common_policy_->DisplayPolicyInfo();

    common_policy_->UpdateObservation = [&](const RobotBasicState& ro){
        int obs_dim = common_policy_->obs_dim_;
        int obs_his_num = common_policy_->obs_history_num_;
        Vec3f cmd_vel = ro.cmd_vel_normlized.cwiseProduct(common_policy_->max_cmd_vel_);

        common_policy_->current_observation_.setZero(obs_dim);
        Vec3f project_gravity = ro.base_rot_mat.transpose() * Vec3f(0., 0., -1);
        common_policy_->current_observation_ << common_policy_->omega_scale_*ro.base_omega,
                                                project_gravity,
                                                cmd_vel.cwiseProduct(common_policy_->cmd_vel_scale_),
                                                ro.joint_pos - common_policy_->dof_pos_default_,
                                                common_policy_->dof_vel_scale_*ro.joint_vel,
                                                common_policy_->last_action_;

        VecXf obs_history_record = common_policy_->observation_history_.segment(obs_dim, (obs_his_num-1)*obs_dim).eval();
        common_policy_->observation_history_.segment(0, (obs_his_num-1)*obs_dim) = obs_history_record;
        common_policy_->observation_history_.segment((obs_his_num-1)*obs_dim, obs_dim) = common_policy_->current_observation_;

        common_policy_->observation_total_.segment(0, obs_dim) = common_policy_->current_observation_;
        common_policy_->observation_total_.segment(obs_dim, obs_dim*obs_his_num) = common_policy_->observation_history_;
    };
}

void RLControlState::SpeedPolicyInitialize(){
    const std::string p = GetAbsPath()+"/../state_machine/policy/policy_est_p50_v2.0.3.pt";
    speed_policy_ = std::make_shared<JueyingPolicyRunner>("speed", p, 141, 0);
    speed_policy_->SetPDGain(Vec3f(400.0, 400.0, 500.0), Vec3f(3., 3. , 3.));
    speed_policy_->SetCmdMaxVel(Vec3f(4.0, 0.5, 0.5));
    speed_policy_->SetTorqueLimit(cp_ptr_->torque_limit_);
    speed_policy_->DisplayPolicyInfo();
    
    speed_policy_->UpdateObservation = [&](const RobotBasicState& ro){
        int obs_dim = speed_policy_->obs_dim_;
        int obs_his_num = speed_policy_->obs_history_num_;
        int buffer_size = speed_policy_->buffer_size_;
        VecXf dof_pos_default = speed_policy_->dof_pos_default_;
        Vec3f cmd_vel = ro.cmd_vel_normlized.cwiseProduct(speed_policy_->max_cmd_vel_);

        speed_policy_->current_observation_.setZero(obs_dim);
        Vec3f project_gravity = ro.base_rot_mat.transpose() * Vec3f(0., 0., -1);

        speed_policy_->current_observation_ << speed_policy_->omega_scale_*ro.base_omega,
                                                project_gravity,
                                                cmd_vel.cwiseProduct(speed_policy_->cmd_vel_scale_),
                                                ro.joint_pos - dof_pos_default,
                                                speed_policy_->dof_vel_scale_*ro.joint_vel,
                                                speed_policy_->last_action_,
                                                speed_policy_->action_buffer_[buffer_size-2],
                                                speed_policy_->dof_pos_buffer_[buffer_size-1] - dof_pos_default,
                                                speed_policy_->dof_pos_buffer_[buffer_size-2] - dof_pos_default,
                                                speed_policy_->dof_pos_buffer_[buffer_size-3] - dof_pos_default,
                                                speed_policy_->dof_vel_scale_*speed_policy_->dof_vel_buffer_[buffer_size-1],
                                                speed_policy_->dof_vel_scale_*speed_policy_->dof_vel_buffer_[buffer_size-2],
                                                speed_policy_->dof_vel_scale_*speed_policy_->dof_vel_buffer_[buffer_size-3],
                                                GetFootPos(ro.joint_pos);

        speed_policy_->dof_pos_buffer_.push_back(ro.joint_pos);
        speed_policy_->dof_vel_buffer_.push_back(ro.joint_vel);
        speed_policy_->dof_vel_buffer_.pop_front();
        speed_policy_->dof_pos_buffer_.pop_front();

        speed_policy_->observation_total_ = speed_policy_->current_observation_;
    };
}

void RLControlState::TumblerStandPolicyInitialize(){
    const std::string p = GetAbsPath()+"/../state_machine/policy/stand_student.pt";
    
    tumbler_stand_policy_ = std::make_shared<JueyingPolicyRunner>("tumbler_stand", p, 45, 55, 12, 3);
    tumbler_stand_policy_->SetPDGain(Vec3f(33., 25., 25), Vec3f(0.8, 0.8, 0.8));
    tumbler_stand_policy_->SetDefaultJointPos(Vec3f(0.1, -1., 1.8));
    tumbler_stand_policy_->SetTorqueLimit(cp_ptr_->torque_limit_);
    tumbler_stand_policy_->DisplayPolicyInfo();

    tumbler_stand_policy_->UpdateObservation = [&](const RobotBasicState& ro){
        auto ptr = tumbler_stand_policy_;
        int obs_dim = ptr->obs_dim_;
        int obs_his_num = ptr->obs_history_num_;    
        int cmd_dim = ptr->cmd_dim_;

        Eigen::AngleAxisf pitchAngle(ro.base_rpy(1), Vec3f::UnitY());
        Eigen::AngleAxisf rollAngle(ro.base_rpy(0), Vec3f::UnitX());
        Eigen::Quaternion<float> q = pitchAngle*rollAngle;
        Vec3f project_gravity = q.matrix() * Vec3f(0., 0., -1.);
        // Vec3f project_gravity = ro.base_rot_mat.transpose() * Vec3f(0., 0., -1.);
        VecXf obs_history_record = ptr->observation_history_.segment(obs_dim, (obs_his_num-1)*obs_dim).eval();
        ptr->observation_history_.segment(0, (obs_his_num-1)*obs_dim) = obs_history_record;
        ptr->observation_history_.segment((obs_his_num-1)*obs_dim, obs_dim) = ptr->current_observation_;

        ptr->current_observation_.setZero(obs_dim);
        ptr->current_observation_ << ptr->last_action_,
                                    1./1.3*ro.base_omega,
                                    ConventionShift(ro.joint_pos),
                                    ConventionShift(ro.joint_vel),
                                    1./9.815*ro.base_acc,
                                    project_gravity;

        ptr->observation_total_.segment(0, obs_dim*obs_his_num) = ptr->observation_history_;
        ptr->observation_total_.segment(obs_dim*obs_his_num, obs_dim) = ptr->current_observation_;
        
        ptr->observation_total_.segment(obs_dim*obs_his_num+obs_dim, cmd_dim) = Vec3f(0., cos(0.), sin(0.));
    };

    tumbler_stand_policy_->UpdateAction = [&](VecXf& action){
        VecXf low_limit = Vec3f(-0.49, -2.7, 0.45).replicate(4, 1) - 0.1*VecXf::Ones(12);
        VecXf high_limit = Vec3f(0.49, 0.33, 2.7).replicate(4, 1) + 0.1*VecXf::Ones(12);
        float lpf = 0.766667;
        VecXf output_action(12);
        auto ptr = tumbler_stand_policy_;
        
        if(ptr->run_cnt_ <= ptr->obs_history_num_+1){
            output_action = ptr->dof_pos_default_;
        }else{         
            ptr->last_action_ = lpf*ptr->last_action_ + (1.-lpf)*action;
            output_action = low_limit+0.5*(ptr->last_action_+VecXf::Ones(12)).cwiseProduct(high_limit-low_limit);  
        }
        output_action = output_action.cwiseMax(low_limit + 0.1*VecXf::Ones(12)).cwiseMin(high_limit - 0.1*VecXf::Ones(12));
        ptr->last_action_ = 2.*(output_action - low_limit).cwiseQuotient(high_limit-low_limit)-VecXf::Ones(12);
        ptr->last_action_ = ptr->last_action_.cwiseMax(-1*VecXf::Ones(12)).cwiseMin(VecXf::Ones(12));
        // std::cout << "last_action:  " << ptr->last_action_.transpose() << std::endl;
        output_action = ConventionShift(output_action);
        
        RobotAction ra;
        ra.goal_joint_pos = output_action;
        ra.goal_joint_vel = VecXf::Zero(ptr->act_dim_);
        ra.tau_ff = VecXf::Zero(ptr->act_dim_);
        ra.kp = ptr->kp_;
        ra.kd = ptr->kd_;
        return ra;
    };
}

void RLControlState::TumblerForwardPolicyInitialize(){
    const std::string p = GetAbsPath()+"/../state_machine/policy/forward_student.pt";
    
    tumbler_stand_policy_ = std::make_shared<JueyingPolicyRunner>("tumbler_forward", p, 45, 55, 12, 3);
    tumbler_stand_policy_->SetPDGain(Vec3f(33., 25., 25), Vec3f(0.7, 0.7, 0.7));
    tumbler_stand_policy_->SetDefaultJointPos(Vec3f(0.1, -1., 1.8));
    tumbler_stand_policy_->SetTorqueLimit(cp_ptr_->torque_limit_);
    tumbler_stand_policy_->DisplayPolicyInfo();

    tumbler_stand_policy_->UpdateObservation = [&](const RobotBasicState& ro){
        auto ptr = tumbler_stand_policy_;
        int obs_dim = ptr->obs_dim_;
        int obs_his_num = ptr->obs_history_num_;    
        int cmd_dim = ptr->cmd_dim_;

        Eigen::AngleAxisf pitchAngle(ro.base_rpy(1), Vec3f::UnitY());
        Eigen::AngleAxisf rollAngle(ro.base_rpy(0), Vec3f::UnitX());
        Eigen::Quaternion<float> q = pitchAngle*rollAngle;
        Vec3f project_gravity = q.matrix() * Vec3f(0., 0., -1.);
        // Vec3f project_gravity = ro.base_rot_mat.transpose() * Vec3f(0., 0., -1.);
        VecXf obs_history_record = ptr->observation_history_.segment(obs_dim, (obs_his_num-1)*obs_dim).eval();
        ptr->observation_history_.segment(0, (obs_his_num-1)*obs_dim) = obs_history_record;
        ptr->observation_history_.segment((obs_his_num-1)*obs_dim, obs_dim) = ptr->current_observation_;

        ptr->current_observation_.setZero(obs_dim);
        ptr->current_observation_ << ptr->last_action_,
                                    1./1.3*ro.base_omega,
                                    ConventionShift(ro.joint_pos),
                                    ConventionShift(ro.joint_vel),
                                    1./9.815*ro.base_acc,
                                    project_gravity;

        ptr->observation_total_.segment(0, obs_dim*obs_his_num) = ptr->observation_history_;
        ptr->observation_total_.segment(obs_dim*obs_his_num, obs_dim) = ptr->current_observation_;
        
        ptr->observation_total_.segment(obs_dim*obs_his_num+obs_dim, cmd_dim) = Vec3f(0., cos(0.), sin(0.));
    };

    tumbler_stand_policy_->UpdateAction = [&](VecXf& action){
        VecXf low_limit = Vec3f(-0.49, -2.7, 0.45).replicate(4, 1) - 0.1*VecXf::Ones(12);
        VecXf high_limit = Vec3f(0.49, 0.33, 2.7).replicate(4, 1) + 0.1*VecXf::Ones(12);
        float lpf = 0.766667;
        VecXf output_action(12);
        auto ptr = tumbler_stand_policy_;
        
        if(ptr->run_cnt_ <= ptr->obs_history_num_+1){
            output_action = ptr->dof_pos_default_;
        }else{         
            ptr->last_action_ = lpf*ptr->last_action_ + (1.-lpf)*action;
            output_action = low_limit+0.5*(ptr->last_action_+VecXf::Ones(12)).cwiseProduct(high_limit-low_limit);  
        }
        output_action = output_action.cwiseMax(low_limit + 0.1*VecXf::Ones(12)).cwiseMin(high_limit - 0.1*VecXf::Ones(12));
        ptr->last_action_ = 2.*(output_action - low_limit).cwiseQuotient(high_limit-low_limit)-VecXf::Ones(12);
        ptr->last_action_ = ptr->last_action_.cwiseMax(-1*VecXf::Ones(12)).cwiseMin(VecXf::Ones(12));
        // std::cout << "last_action:  " << ptr->last_action_.transpose() << std::endl;
        output_action = ConventionShift(output_action);
        
        RobotAction ra;
        ra.goal_joint_pos = output_action;
        ra.goal_joint_vel = VecXf::Zero(ptr->act_dim_);
        ra.tau_ff = VecXf::Zero(ptr->act_dim_);
        ra.kp = ptr->kp_;
        ra.kd = ptr->kd_;
        return ra;
    };   
}

void RLControlState::ParkourPolicyInitialize() {
    const std::string actor_path = GetAbsPath() + "/../state_machine/policy/parkour.pt";

    parkour_policy_ = std::make_shared<JueyingPolicyRunner>("parkour", actor_path, 53, 10);
    parkour_policy_->SetPDGain(Vec3f(40., 40., 40.), Vec3f(1, 1, 1));
    parkour_policy_->SetDefaultJointPos(Vec3f(0.0, -1, 1.8));
    // VecXf parkour_default_joint_pos;
    // parkour_default_joint_pos.setZero(12);
    // parkour_default_joint_pos << 0.1, -0.8, 1.5, -0.1, -0.8, 1.5, 0.1, -1, 1.5, -0.1, -1, 1.5;
    // parkour_policy_->SetDefaultJointPos12(parkour_default_joint_pos);
    parkour_policy_->SetTorqueLimit(cp_ptr_->torque_limit_);
    parkour_policy_->DisplayPolicyInfo();

    parkour_policy_->UpdateObservation = [&](const RobotBasicState& ro) {
        int obs_dim = parkour_policy_->obs_dim_;
        int obs_his_num = parkour_policy_->obs_history_num_;
        Vec3f cmd_vel = ro.cmd_vel_normlized.cwiseProduct(parkour_policy_->max_cmd_vel_);
        torch::Tensor obs_prop_tensor_, obs_priv_explicit_tensor_;
        std::vector<c10::IValue> obs_prop_vector_{};

        // current_observation_ is the proprioceptive observation
        parkour_policy_->current_observation_.setZero(obs_dim);
        parkour_policy_->current_observation_ << parkour_policy_->omega_scale_ * ro.base_omega,
                                                 ro.base_rpy.head<2>(),
                                                 Vec3f(0.0, 0.0, 0.0),
                                                 0.f,
                                                 0.f,
                                                 //  0.5f, // speed
                                                 cmd_vel.cwiseProduct(parkour_policy_->cmd_vel_scale_)[0],
                                                 1.0f,
                                                 0.f,
                                                 ro.joint_pos - parkour_policy_->dof_pos_default_,
                                                 parkour_policy_->dof_vel_scale_ * ro.joint_vel,
                                                 parkour_policy_->action_buffer_.back(),
                                                 Vec4f(0., 0., 0., 0.);

        // std::cout << "parkour_policy_->current_observation_" << parkour_policy_->current_observation_ << std::endl;
        std::cout << "command vel" << cmd_vel.cwiseProduct(parkour_policy_->cmd_vel_scale_)[0] << std::endl;
        dp_sensor_->get_prop(parkour_policy_->current_observation_);
        parkour_policy_->depth_latent_tensor_ = dp_sensor_->get_latent();
        parkour_policy_->yaw_tensor_ = dp_sensor_->get_yaw();
        parkour_policy_->observation_total_.segment(0, obs_dim) = parkour_policy_->current_observation_;
        parkour_policy_->observation_total_.segment(obs_dim, 132) = VecXf::Zero(132); // scan_dot
        parkour_policy_->observation_total_.segment(obs_dim+132, 9) = VecXf::Zero(9); // priv_explicit
        parkour_policy_->observation_total_.segment(obs_dim+132+9, 29) = VecXf::Zero(29); // priv_latent
        parkour_policy_->observation_total_.segment(obs_dim+132+9+29, obs_his_num*obs_dim) = parkour_policy_->observation_history_;

        // Update history observation after computing observation_total_
        VecXf obs_history_record = parkour_policy_->observation_history_.segment(obs_dim, (obs_his_num-1)*obs_dim).eval();
        parkour_policy_->observation_history_.segment(0, (obs_his_num-1)*obs_dim) = obs_history_record;
        parkour_policy_->observation_history_.segment((obs_his_num-1)*obs_dim, obs_dim) = parkour_policy_->current_observation_;
        parkour_policy_->dof_pos_buffer_.push_back(ro.joint_pos);
        parkour_policy_->dof_vel_buffer_.push_back(ro.joint_vel);
        parkour_policy_->dof_vel_buffer_.pop_front();
        parkour_policy_->dof_pos_buffer_.pop_front();
    };
    // TODO parkour_policy_->UpdateAction 
}

VecXf RLControlState::GetFootPos(const VecXf& joint_pos){
    float l0, l1, l2;
    float s1, s2, s3;
    float x, y, z, zt;
    VecXf foot_pos = VecXf::Zero(12);

    for (int i=0;i<4;i++){  
        l0 = cp_ptr_->hip_len_; l1 = cp_ptr_->thigh_len_; l2 = cp_ptr_->shank_len_;
        if(i==0||i==2) l0 = -cp_ptr_->hip_len_;
        s1 = joint_pos[3*i];
        s2 = joint_pos[3*i+1];
        s3 = joint_pos[3*i+2];

        x = l1 * sin(s2) + l2 * sin(s2 + s3);
        zt = -(l1 * cos(s2) + l2 * cos(s2 + s3));
        y = zt * sin(s1) - l0 * cos(s1);
        z = zt * cos(s1) + l0 * sin(s1);

        foot_pos[3*i] = x;
        foot_pos[3*i+1] = y;
        foot_pos[3*i+2] = z;
    }
    return foot_pos;
}

Mat3f RLControlState::RpyToRm(const Vec3f &rpy){
    Mat3f rm;
    Eigen::AngleAxisf yawAngle(rpy(2), Vec3f::UnitZ());
    Eigen::AngleAxisf pitchAngle(rpy(1), Vec3f::UnitY());
    Eigen::AngleAxisf rollAngle(rpy(0), Vec3f::UnitX());
    Eigen::Quaternion<float> q = yawAngle*pitchAngle*rollAngle;
    return q.matrix();
}

VecXf RLControlState::ConventionShift(const VecXf& dr_convention){
    VecXf x(12);
    x.segment(0,3) = dr_convention.segment(3,3);
    x.segment(3,3) = dr_convention.segment(0,3);
    x.segment(6,3) = dr_convention.segment(9,3);
    x.segment(9,3) = dr_convention.segment(6,3);
    return x;
}
