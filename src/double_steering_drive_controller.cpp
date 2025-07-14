// Copyright 2020 PAL Robotics S.L.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

/*
 * Author: Jaerak Son
 */

#include <memory>
#include <queue>
#include <string>
#include <utility>
#include <vector>
#include <Eigen/Dense>

#include "double_steering_drive_controller/double_steering_drive_controller.hpp"
#include "hardware_interface/types/hardware_interface_type_values.hpp"
#include "lifecycle_msgs/msg/state.hpp"
#include "rclcpp/logging.hpp"
#include "tf2/LinearMath/Quaternion.h"
#include "visualization_msgs/msg/marker.hpp"

namespace
{
constexpr auto DEFAULT_COMMAND_TOPIC = "~/cmd_vel";
constexpr auto DEFAULT_COMMAND_UNSTAMPED_TOPIC = "~/cmd_vel_unstamped";
constexpr auto DEFAULT_COMMAND_OUT_TOPIC = "~/cmd_vel_out";
constexpr auto DEFAULT_ODOMETRY_TOPIC = "~/odom";
constexpr auto DEFAULT_TRANSFORM_TOPIC = "/tf";
}  // namespace

namespace double_steering_drive_controller
{
using namespace std::chrono_literals;
using controller_interface::interface_configuration_type;
using controller_interface::InterfaceConfiguration;
using hardware_interface::HW_IF_POSITION;
using hardware_interface::HW_IF_VELOCITY;
using lifecycle_msgs::msg::State;

DoubleSteeringDriveController::DoubleSteeringDriveController() : controller_interface::ControllerInterface() {}

const char * DoubleSteeringDriveController::wheel_feedback_type() const
{
  // position_feedback 파라미터에 따라 피드백 타입을 결정합니다.
  return params_.position_feedback ? HW_IF_POSITION : HW_IF_VELOCITY;
}

const char * DoubleSteeringDriveController::steering_feedback_type() const
{
  // position_feedback 파라미터에 따라 스티어링 피드백 타입을 결정합니다.
  return params_.steering_position_feedback ? HW_IF_POSITION : HW_IF_VELOCITY;
}

controller_interface::CallbackReturn DoubleSteeringDriveController::on_init()
{
  try
  {
    // 파라미터 리스너를 생성하고 파라미터를 가져옵니다.
    param_listener_ = std::make_shared<ParamListener>(get_node());
    params_ = param_listener_->get_params();
  }
  catch (const std::exception & e)
  {
    fprintf(stderr, "Exception thrown during init stage with message: %s \n", e.what());
    return controller_interface::CallbackReturn::ERROR;
  }

  return controller_interface::CallbackReturn::SUCCESS;
}

InterfaceConfiguration DoubleSteeringDriveController::command_interface_configuration() const
{
  // 제어에 사용할 joint/velocity 인터페이스 목록을 생성합니다.
  std::vector<std::string> conf_names;
  for (const auto & joint_name : params_.front_wheel_names)
  {
    conf_names.push_back(joint_name + "/" + HW_IF_VELOCITY);
  }
  for (const auto & joint_name : params_.rear_wheel_names)
  {
    conf_names.push_back(joint_name + "/" + HW_IF_VELOCITY);
  }
  conf_names.push_back(params_.front_steering_name + "/" + HW_IF_POSITION);
  conf_names.push_back(params_.rear_steering_name + "/" + HW_IF_POSITION);
  return {interface_configuration_type::INDIVIDUAL, conf_names};
}

InterfaceConfiguration DoubleSteeringDriveController::state_interface_configuration() const
{
  // 상태 피드백에 사용할 joint/position 또는 joint/velocity 인터페이스 목록을 생성합니다.
  std::vector<std::string> conf_names;
  for (const auto & joint_name : params_.front_wheel_names)
  {
    conf_names.push_back(joint_name + "/" + wheel_feedback_type());
  }
  for (const auto & joint_name : params_.rear_wheel_names)
  {
    conf_names.push_back(joint_name + "/" + wheel_feedback_type());
  }
  conf_names.push_back(params_.front_steering_name + "/" + steering_feedback_type());
  conf_names.push_back(params_.rear_steering_name + "/" + steering_feedback_type());
  return {interface_configuration_type::INDIVIDUAL, conf_names};
}

controller_interface::return_type DoubleSteeringDriveController::update(
  const rclcpp::Time & time, const rclcpp::Duration & period)
{
  double front_steering_feedback = 0.0;
  double rear_steering_feedback = 0.0;

  bool old_updated = false;
  bool zero_command = false;
  
  // 주기적으로 호출되어 로봇의 상태를 업데이트하고, 명령을 바퀴에 전달합니다.
  auto logger = get_node()->get_logger();
  if (get_state().id() == State::PRIMARY_STATE_INACTIVE)
  {
    if (!is_halted)
    {
      halt(); // 비활성 상태면 바퀴를 정지시킵니다.
      is_halted = true;
    }
    return controller_interface::return_type::OK;
  }

  std::shared_ptr<Twist> last_command_msg;
  received_velocity_msg_ptr_.get(last_command_msg);

  if (last_command_msg == nullptr)
  {
    RCLCPP_WARN(logger, "Velocity message received was a nullptr.");
    return controller_interface::return_type::ERROR;
  }

  const auto age_of_last_command = time - last_command_msg->header.stamp;
  // 명령이 너무 오래되었으면 정지 명령으로 대체합니다.
  if (age_of_last_command > cmd_vel_timeout_)
  {
    last_command_msg->twist.linear.x = 0.0;
    last_command_msg->twist.linear.y = 0.0;
    last_command_msg->twist.angular.z = 0.0;
    old_updated = true;
  }

  // 명령 제한자(SpeedLimiter)로 속도 제한
  Twist command = *last_command_msg;
  double & linear_x_command = command.twist.linear.x;
  double & linear_y_command = command.twist.linear.y;
  double & angular_command = command.twist.angular.z;

  if (linear_x_command == 0.0 && linear_y_command == 0.0 && angular_command == 0.0)
  {
    zero_command = true; // 명령이 모두 0이면 zero_command 플래그 설정
  }

  previous_update_timestamp_ = time;

  // 파라미터에 따라 바퀴 간격, 반지름 계산
  const double wheel_base = params_.wheel_base_multiplier * params_.wheel_base;
  const double front_wheel_radius = params_.front_wheel_radius_multiplier * params_.wheel_radius;
  const double rear_wheel_radius = params_.rear_wheel_radius_multiplier * params_.wheel_radius;

  if (params_.open_loop)
  {
    // 오픈 루프 모드: 명령만으로 오도메트리 갱신
    odometry_.updateOpenLoop(linear_x_command, linear_y_command, angular_command, time);
  }
  else
  {

    // 클로즈 루프 모드: 바퀴 피드백으로 오도메트리 갱신
    double front_wheel_feedback_mean = 0.0;
    double rear_wheel_feedback_mean = 0.0;
    
    // 전방 및 후방 바퀴의 피드백 값을 평균 계산
    for (size_t index = 0; index < static_cast<size_t>(params_.wheels_per_side); ++index)
    {
      const double front_wheel_feedback = registered_front_wheel_handles_[index].feedback.get().get_value();
      const double rear_wheel_feedback = registered_rear_wheel_handles_[index].feedback.get().get_value();

      // RCLCPP_INFO(logger, "Front Wheel Feedback: %f, Rear Wheel Feedback: %f",
      //             front_wheel_feedback, rear_wheel_feedback);

      // 피드백 값이 유효하지 않은 경우 에러 처리
      if (std::isnan(front_wheel_feedback) || std::isnan(rear_wheel_feedback))
      {
        RCLCPP_ERROR(
          logger, "Either the front or rear wheel %s is invalid for index [%zu]", wheel_feedback_type(),
          index);
          return controller_interface::return_type::ERROR;
        }
        
        // 전방 및 후방 피드백 값을 누적
        front_wheel_feedback_mean += front_wheel_feedback;
        rear_wheel_feedback_mean += rear_wheel_feedback;
      }
      
      // 평균 계산
      front_wheel_feedback_mean /= static_cast<double>(params_.wheels_per_side);
      rear_wheel_feedback_mean /= static_cast<double>(params_.wheels_per_side);

      // RCLCPP_INFO(logger, "Front Wheel Feedback Mean: %f, Rear Wheel Feedback Mean: %f",
      //             front_wheel_feedback_mean, rear_wheel_feedback_mean);

    // 전방 및 후방 스티어링의 피드백 값을 계산
    front_steering_feedback = registered_front_steering_handle_->feedback.get().get_value();
    rear_steering_feedback = registered_rear_steering_handle_->feedback.get().get_value();

    // 피드백 값이 유효하지 않은 경우 에러 처리
    if (std::isnan(front_steering_feedback) || std::isnan(rear_steering_feedback))
    {
      RCLCPP_ERROR(
        logger, "Either the front or rear steering %s is invalid",
        steering_feedback_type());
      return controller_interface::return_type::ERROR;
    }

    // steering 시각화
    if (steering_marker_pub_)
    {
      // 단위 벡터 길이 (m)
      constexpr double arrow_length = 1.0;

      // FRONT steering vector
      visualization_msgs::msg::Marker front_marker;
      front_marker.header.frame_id = params_.base_frame_id;
      front_marker.header.stamp = time;
      front_marker.ns = "front_steering_vector";
      front_marker.id = 0;
      front_marker.type = visualization_msgs::msg::Marker::ARROW;
      front_marker.action = visualization_msgs::msg::Marker::ADD;
      front_marker.scale.x = 0.05; // shaft diameter
      front_marker.scale.y = 0.1;  // head diameter
      front_marker.scale.z = 0.1;  // head length
      front_marker.color.r = 0.0;
      front_marker.color.g = 1.0;
      front_marker.color.b = 0.0;
      front_marker.color.a = 1.0;

      geometry_msgs::msg::Point p_front_start, p_front_end;
      p_front_start.x = wheel_base / 2.0;
      p_front_start.y = 0.0;
      p_front_start.z = 0.0;

      p_front_end.x = p_front_start.x + arrow_length * std::cos(front_steering_feedback);
      p_front_end.y = p_front_start.y + arrow_length * std::sin(front_steering_feedback);
      p_front_end.z = 0.0;

      front_marker.points.push_back(p_front_start);
      front_marker.points.push_back(p_front_end);

      steering_marker_pub_->publish(front_marker);

      // REAR steering vector
      visualization_msgs::msg::Marker rear_marker = front_marker;
      rear_marker.ns = "rear_steering_vector";
      rear_marker.id = 1;
      rear_marker.color.r = 1.0;
      rear_marker.color.g = 1.0;
      rear_marker.color.b = 0.0;

      geometry_msgs::msg::Point p_rear_start, p_rear_end;
      p_rear_start.x = -wheel_base / 2.0;
      p_rear_start.y = 0.0;
      p_rear_start.z = 0.0;

      p_rear_end.x = p_rear_start.x + arrow_length * std::cos(rear_steering_feedback);
      p_rear_end.y = p_rear_start.y + arrow_length * std::sin(rear_steering_feedback);
      p_rear_end.z = 0.0;

      rear_marker.points.clear();
      rear_marker.points.push_back(p_rear_start);
      rear_marker.points.push_back(p_rear_end);

      steering_marker_pub_->publish(rear_marker);
    }

    // 위치 피드백을 사용하는 경우
    if (params_.position_feedback)
    {
      // RCLCPP_INFO(logger, "Front Wheel Feedback Mean: %f, Rear Wheel Feedback Mean: %f",
      //             front_wheel_feedback_mean, rear_wheel_feedback_mean);
      // RCLCPP_INFO(logger, "Front Steering Feedback: %f, Rear Steering Feedback: %f",
      //             front_steering_feedback, rear_steering_feedback);
      // RCLCPP_INFO(logger, "Front Wheel vel: %f, Rear Wheel vel: %f",
      //             front_wheel_feedback_mean * front_wheel_radius * period.seconds(), rear_wheel_feedback_mean * rear_wheel_radius * period.seconds());            
      // 바퀴 위치 피드백을 사용하여 오도메트리 갱신
      odometry_.update(front_wheel_feedback_mean, rear_wheel_feedback_mean, front_steering_feedback, rear_steering_feedback, wheel_base, time);
    }
    // 속도 피드백을 사용하여 오도메트리 갱신
    else
    {
      // RCLCPP_INFO(logger, "Front Wheel Feedback Mean: %f, Rear Wheel Feedback Mean: %f",
      //             front_wheel_feedback_mean, rear_wheel_feedback_mean);
      // RCLCPP_INFO(logger, "Front Steering Feedback: %f, Rear Steering Feedback: %f",
      //             front_steering_feedback, rear_steering_feedback);
      // RCLCPP_INFO(logger, "Front Wheel vel: %f, Rear Wheel vel: %f",
      //             front_wheel_feedback_mean * front_wheel_radius * period.seconds(), rear_wheel_feedback_mean * rear_wheel_radius * period.seconds());
      odometry_.updateFromVelocity(
        front_wheel_feedback_mean * front_wheel_radius * period.seconds(),
        rear_wheel_feedback_mean * rear_wheel_radius * period.seconds(),
        front_steering_feedback, rear_steering_feedback, wheel_base, time);
    }
  }

  tf2::Quaternion orientation;
  orientation.setRPY(0.0, 0.0, odometry_.getHeading());

  bool should_publish = false;
  try
  {
    // publish_period_마다 토픽 발행
    if (previous_publish_timestamp_ + publish_period_ < time)
    {
      previous_publish_timestamp_ += publish_period_;
      should_publish = true;
    }
  }
  catch (const std::runtime_error &)
  {
    // 시간 소스가 바뀌면 예외 처리 후 타임스탬프 초기화
    previous_publish_timestamp_ = time;
    should_publish = true;
  }

  if (should_publish)
  {
    // 오도메트리 토픽 발행
    if (realtime_odometry_publisher_->trylock())
    {
      auto & odometry_message = realtime_odometry_publisher_->msg_;
      odometry_message.header.stamp = time;
      odometry_message.pose.pose.position.x = odometry_.getX();
      odometry_message.pose.pose.position.y = odometry_.getY();
      odometry_message.pose.pose.orientation.x = orientation.x();
      odometry_message.pose.pose.orientation.y = orientation.y();
      odometry_message.pose.pose.orientation.z = orientation.z();
      odometry_message.pose.pose.orientation.w = orientation.w();
      odometry_message.twist.twist.linear.x = odometry_.getLinearX();
      odometry_message.twist.twist.linear.y = odometry_.getLinearY();
      odometry_message.twist.twist.angular.z = odometry_.getAngular();
      realtime_odometry_publisher_->unlockAndPublish();
    }

    // tf 변환 발행
    if (params_.enable_odom_tf && realtime_odometry_transform_publisher_->trylock())
    {
      auto & transform = realtime_odometry_transform_publisher_->msg_.transforms.front();
      transform.header.stamp = time;
      transform.transform.translation.x = odometry_.getX();
      transform.transform.translation.y = odometry_.getY();
      transform.transform.rotation.x = orientation.x();
      transform.transform.rotation.y = orientation.y();
      transform.transform.rotation.z = orientation.z();
      transform.transform.rotation.w = orientation.w();
      realtime_odometry_transform_publisher_->unlockAndPublish();
    }
  }

  // 속도 제한 적용 (가속도, 저크 등)
  auto & last_command = previous_commands_.back().twist;
  auto & second_to_last_command = previous_commands_.front().twist;
  limiter_linear_x_.limit(
    linear_x_command, last_command.linear.x, second_to_last_command.linear.x, period.seconds());
  limiter_linear_y_.limit(
    linear_y_command, last_command.linear.y, second_to_last_command.linear.y, period.seconds());
  // 각속도 제한 적용
  limiter_angular_.limit(
    angular_command, last_command.angular.z, second_to_last_command.angular.z, period.seconds());

  previous_commands_.pop();
  previous_commands_.emplace(command);

  // 제한된 속도 명령 발행
  if (publish_limited_velocity_ && realtime_limited_velocity_publisher_->trylock())
  {
    auto & limited_velocity_command = realtime_limited_velocity_publisher_->msg_;
    limited_velocity_command.header.stamp = time;
    limited_velocity_command.twist = command.twist;
    realtime_limited_velocity_publisher_->unlockAndPublish();
  }

  // 바퀴 속도 계산 및 설정
  Eigen::Vector3d linear_command_vec(linear_x_command, linear_y_command, 0.0);
  Eigen::Vector3d angular_command_vec(0.0, 0.0, angular_command);
  Eigen::Vector3d front_base_vec(wheel_base / 2.0, 0.0, 0.0);
  Eigen::Vector3d rear_base_vec(-wheel_base / 2.0, 0.0, 0.0);

  Eigen::Vector3d v_rot_f = angular_command_vec.cross(front_base_vec);
  Eigen::Vector3d v_rot_r = angular_command_vec.cross(rear_base_vec);

  Eigen::Vector3d v_front_vec = linear_command_vec + v_rot_f;
  Eigen::Vector3d v_rear_vec = linear_command_vec + v_rot_r;

  double velocity_front = v_front_vec.head<2>().norm()/front_wheel_radius;
  double velocity_rear = v_rear_vec.head<2>().norm()/rear_wheel_radius;

  // 최대 조향 각도 변화량 계산
  double dt = period.seconds();
  double max_steering_delta = params_.max_steering_velocity * dt;

  //front_steering_angle 계산
  // 1) front_steering_angle 계산
  double target_steering_front;
  if (old_updated || zero_command) {
    target_steering_front = last_front_steering_angle_; // 이전 업데이트가 오래된 경우 last_front_steering_angle_를 사용
  }
  else{
    target_steering_front = atan2(v_front_vec.y(), v_front_vec.x());
  }

  // 2) front_steering_angle과 last_front_steering_angle_ 사이의 delta 계산
  double front_steering_delta = target_steering_front - last_front_steering_angle_;
  front_steering_delta = wrapAngle(front_steering_delta);

  // 3) delta가 90도(π/2) 초과면 flip
  if (abs(front_steering_delta) > M_PI / 2.0) {
    // 각도를 180도 돌려서 최소 회전각으로 유지
    target_steering_front = wrapAngle(target_steering_front + M_PI);
    // 속도 반전
    velocity_front *= -1.0;

    // delta 재계산
    front_steering_delta = target_steering_front - last_front_steering_angle_;
    front_steering_delta = wrapAngle(front_steering_delta);
  }

  // 4) 예상 누적 각도가 허용 범위를 초과하면 제한
  double predictive_cumulative_front_steering_angle_ = cumulative_front_steering_angle_ + front_steering_delta;
  if (abs(predictive_cumulative_front_steering_angle_) > params_.max_steering_angle)
  {
    RCLCPP_WARN(
      logger, "\033[33mFront steering angle exceeds max limit, flipping direction. "
              "Cumulative front steering angle: %.2f rad, Max limit: %.2f rad\033[0m",
      predictive_cumulative_front_steering_angle_, params_.max_steering_angle);
    // 각도를 180도 돌려서 최소 회전각으로 유지
    target_steering_front = wrapAngle(target_steering_front + M_PI);
    // 속도 반전
    velocity_front *= -1.0;
    // delta 재계산
    front_steering_delta = target_steering_front - last_front_steering_angle_;
    front_steering_delta = wrapAngle(front_steering_delta);
  }

  double scale_front = 1.0;
  if(params_.reduce_wheel_speed_until_steering_reached) {
    scale_front = computeSpeedScale(abs(front_steering_delta), 
                                     params_.min_phi_delta, 
                                     params_.max_phi_delta, 
                                     params_.steering_speed_scale_exponent, 
                                     params_.min_reduced_scale);
  }
  velocity_front *= scale_front;

  // 4) delta가 허용 범위를 초과하면 제한
  // 각도 변화량이 최대 조향 각도 변화량을 초과하면 제한
  // 각도를 업데이트하여 누적 각도와 일치시킴
  if (front_steering_delta >= max_steering_delta) {
      front_steering_delta = max_steering_delta;
      last_front_steering_angle_ += max_steering_delta; // 각도를 업데이트
    } else if (front_steering_delta < -max_steering_delta) {
      front_steering_delta = -max_steering_delta;
      last_front_steering_angle_ -= max_steering_delta; // 각도를 업데이트
    } else {
      // delta가 허용 범위 내에 있으면 그대로 사용
      last_front_steering_angle_ = target_steering_front;
    }

  cumulative_front_steering_angle_ += front_steering_delta;

  double steering_angle_front = last_front_steering_angle_;
  // double steering_front_turns = cumulative_front_steering_angle_ / (2.0 * M_PI);


  // rear_steering_angle 계산
  // 1) rear_steering_angle 계산
  double target_steering_rear;
  if (old_updated || zero_command) {
    target_steering_rear = last_rear_steering_angle_; // 이전 업데이트가 오래된 경우 last_rear_steering_angle_를 사용
  }
  else{
    target_steering_rear = atan2(v_rear_vec.y(), v_rear_vec.x());
  }

  // 2) rear_steering_angle과 last_rear_steering_angle_ 사이의 delta 계산
  double rear_steering_delta = target_steering_rear - last_rear_steering_angle_;
  rear_steering_delta = wrapAngle(rear_steering_delta);

  // 3) delta가 90도(π/2) 초과면 flip
  if (abs(rear_steering_delta) > M_PI / 2.0) {
    // 각도를 180도 돌려서 최소 회전각으로 유지
    target_steering_rear = wrapAngle(target_steering_rear + M_PI);
    // 속도 반전
    velocity_rear *= -1.0;
    // delta 재계산
    rear_steering_delta = target_steering_rear - last_rear_steering_angle_;
    rear_steering_delta = wrapAngle(rear_steering_delta);
  }

  // 4) 예상 누적 각도가 허용 범위를 초과하면 제한
  double predictive_cumulative_rear_steering_angle_ = cumulative_rear_steering_angle_ + rear_steering_delta;
  if (abs(predictive_cumulative_rear_steering_angle_) > params_.max_steering_angle)
  {
    RCLCPP_WARN(
      logger, "\033[33mRear steering angle exceeds max limit, flipping direction. "
              "Cumulative rear steering angle: %.2f rad, Max limit: %.2f rad\033[0m",
      predictive_cumulative_rear_steering_angle_, params_.max_steering_angle);
    // 각도를 180도 돌려서 최소 회전각으로 유지
    target_steering_rear = wrapAngle(target_steering_rear + M_PI);
    // 속도 반전
    velocity_rear *= -1.0;
    // delta 재계산
    rear_steering_delta = target_steering_rear - last_rear_steering_angle_;
    rear_steering_delta = wrapAngle(rear_steering_delta);
  }
  
  // rear wheel속도 스케일링
  // 조향 각도 변화량에 따라 속도를 조정
  // 조향 각도 변화량이 클수록 속도를 줄입니다.
  double scale_rear = 1.0;
  if(params_.reduce_wheel_speed_until_steering_reached) {
    scale_rear = computeSpeedScale(abs(rear_steering_delta),
                                   params_.min_phi_delta,
                                   params_.max_phi_delta,
                                   params_.steering_speed_scale_exponent,
                                   params_.min_reduced_scale);
  }
  velocity_rear *= scale_rear;

  if (rear_steering_delta >= max_steering_delta) {
      rear_steering_delta = max_steering_delta;
      last_rear_steering_angle_ += max_steering_delta; // 각도를 업데이트
    } else if (rear_steering_delta < -max_steering_delta) {
      rear_steering_delta = -max_steering_delta;
      last_rear_steering_angle_ -= max_steering_delta; // 각도를 업데이트
    } else {
      // delta가 허용 범위 내에 있으면 그대로 사용
      last_rear_steering_angle_ = target_steering_rear;
    }

  // 누적 조향 각도 업데이트
  cumulative_rear_steering_angle_ += rear_steering_delta;

  double steering_angle_rear = last_rear_steering_angle_;
  // double steering_rear_turns = cumulative_rear_steering_angle_ / (2.0 * M_PI);


  // RCLCPP_INFO(
  //   logger, "Front steering angle: %.2f rad (%.2f turns), Rear steering angle: %.2f rad (%.2f turns)",
  //   steering_angle_front, steering_front_turns, steering_angle_rear, steering_rear_turns);

  // Update steering angles
  registered_front_steering_handle_->position.get().set_value(steering_angle_front);
  registered_rear_steering_handle_->position.get().set_value(steering_angle_rear);
  // 각 바퀴에 속도 명령 전달
  for (size_t index = 0; index < static_cast<size_t>(params_.wheels_per_side); ++index)
  {
    registered_front_wheel_handles_[index].velocity.get().set_value(velocity_front);
    registered_rear_wheel_handles_[index].velocity.get().set_value(velocity_rear);
  }

  return controller_interface::return_type::OK;
}

controller_interface::CallbackReturn DoubleSteeringDriveController::on_configure(
  const rclcpp_lifecycle::State &)
{
  auto logger = get_node()->get_logger();

  // 파라미터가 변경되었으면 갱신
  if (param_listener_->is_old(params_))
  {
    params_ = param_listener_->get_params();
    RCLCPP_INFO(logger, "Parameters were updated");
  }

  // 앞뒤 바퀴 개수 일치 확인
  if (params_.front_wheel_names.size() != params_.rear_wheel_names.size())
  {
    RCLCPP_ERROR(
      logger, "The number of front wheels [%zu] and the number of rear wheels [%zu] are different",
      params_.front_wheel_names.size(), params_.rear_wheel_names.size());
    return controller_interface::CallbackReturn::ERROR;
  }

  // 바퀴 이름이 비어있는지 확인
  if (params_.front_wheel_names.empty())
  {
    RCLCPP_ERROR(logger, "Wheel names parameters are empty!");
    return controller_interface::CallbackReturn::ERROR;
  }

  // 바퀴 파라미터 및 오도메트리 설정
  const double wheel_base = params_.wheel_base_multiplier * params_.wheel_base;
  const double front_wheel_radius = params_.front_wheel_radius_multiplier * params_.wheel_radius;
  const double rear_wheel_radius = params_.rear_wheel_radius_multiplier * params_.wheel_radius;

  odometry_.setWheelParams(wheel_base, front_wheel_radius, rear_wheel_radius);
  odometry_.setVelocityRollingWindowSize(static_cast<size_t>(params_.velocity_rolling_window_size));

  cmd_vel_timeout_ = std::chrono::milliseconds{static_cast<int>(params_.cmd_vel_timeout * 1000.0)};
  publish_limited_velocity_ = params_.publish_limited_velocity;
  use_stamped_vel_ = params_.use_stamped_vel;

  // 속도 제한자 초기화
  limiter_linear_x_ = SpeedLimiter(
    params_.linear.x.has_velocity_limits, params_.linear.x.has_acceleration_limits,
    params_.linear.x.has_jerk_limits, params_.linear.x.min_velocity, params_.linear.x.max_velocity,
    params_.linear.x.min_acceleration, params_.linear.x.max_acceleration, params_.linear.x.min_jerk,
    params_.linear.x.max_jerk);
  
  limiter_linear_y_ = SpeedLimiter(
    params_.linear.y.has_velocity_limits, params_.linear.y.has_acceleration_limits,
    params_.linear.y.has_jerk_limits, params_.linear.y.min_velocity, params_.linear.y.max_velocity,
    params_.linear.y.min_acceleration, params_.linear.y.max_acceleration, params_.linear.y.min_jerk,
    params_.linear.y.max_jerk);

  limiter_angular_ = SpeedLimiter(
    params_.angular.z.has_velocity_limits, params_.angular.z.has_acceleration_limits,
    params_.angular.z.has_jerk_limits, params_.angular.z.min_velocity,
    params_.angular.z.max_velocity, params_.angular.z.min_acceleration,
    params_.angular.z.max_acceleration, params_.angular.z.min_jerk, params_.angular.z.max_jerk);

  if (!reset())
  {
    return controller_interface::CallbackReturn::ERROR;
  }

  // 좌우 바퀴 개수 저장
  params_.wheels_per_side = params_.front_wheel_names.size();

  // 제한된 속도 퍼블리셔 초기화
  if (publish_limited_velocity_)
  {
    limited_velocity_publisher_ =
      get_node()->create_publisher<Twist>(DEFAULT_COMMAND_OUT_TOPIC, rclcpp::SystemDefaultsQoS());
    realtime_limited_velocity_publisher_ =
      std::make_shared<realtime_tools::RealtimePublisher<Twist>>(limited_velocity_publisher_);
  }

  const Twist empty_twist;
  received_velocity_msg_ptr_.set(std::make_shared<Twist>(empty_twist));

  // 이전 명령 큐 초기화
  previous_commands_.emplace(empty_twist);
  previous_commands_.emplace(empty_twist);

  // 명령 구독자 초기화 (Stamped/Unstamped)
  if (use_stamped_vel_)
  {
    velocity_command_subscriber_ = get_node()->create_subscription<Twist>(
      DEFAULT_COMMAND_TOPIC, rclcpp::SystemDefaultsQoS(),
      [this](const std::shared_ptr<Twist> msg) -> void
      {
        if (!subscriber_is_active_)
        {
          RCLCPP_WARN(
            get_node()->get_logger(), "Can't accept new commands. subscriber is inactive");
          return;
        }
        if ((msg->header.stamp.sec == 0) && (msg->header.stamp.nanosec == 0))
        {
          RCLCPP_WARN_ONCE(
            get_node()->get_logger(),
            "Received TwistStamped with zero timestamp, setting it to current "
            "time, this message will only be shown once");
          msg->header.stamp = get_node()->get_clock()->now();
        }
        received_velocity_msg_ptr_.set(std::move(msg));
      });
  }
  else
  {
    velocity_command_unstamped_subscriber_ =
      get_node()->create_subscription<geometry_msgs::msg::Twist>(
        DEFAULT_COMMAND_UNSTAMPED_TOPIC, rclcpp::SystemDefaultsQoS(),
        [this](const std::shared_ptr<geometry_msgs::msg::Twist> msg) -> void
        {
          if (!subscriber_is_active_)
          {
            RCLCPP_WARN(
              get_node()->get_logger(), "Can't accept new commands. subscriber is inactive");
            return;
          }

          // Stamped 명령으로 변환하여 저장
          std::shared_ptr<Twist> twist_stamped;
          received_velocity_msg_ptr_.get(twist_stamped);
          twist_stamped->twist = *msg;
          twist_stamped->header.stamp = get_node()->get_clock()->now();
        });
  }

  // 오도메트리 퍼블리셔 및 메시지 초기화
  odometry_publisher_ = get_node()->create_publisher<nav_msgs::msg::Odometry>(
    DEFAULT_ODOMETRY_TOPIC, rclcpp::SystemDefaultsQoS());
  realtime_odometry_publisher_ =
    std::make_shared<realtime_tools::RealtimePublisher<nav_msgs::msg::Odometry>>(
      odometry_publisher_);

  // tf prefix 처리
  std::string tf_prefix = "";
  if (params_.tf_frame_prefix_enable)
  {
    if (params_.tf_frame_prefix != "")
    {
      tf_prefix = params_.tf_frame_prefix;
    }
    else
    {
      tf_prefix = std::string(get_node()->get_namespace());
    }

    // prefix 형식 보정
    if (tf_prefix.back() != '/')
    {
      tf_prefix = tf_prefix + "/";
    }
    if (tf_prefix.front() == '/')
    {
      tf_prefix.erase(0, 1);
    }
  }

  const auto odom_frame_id = tf_prefix + params_.odom_frame_id;
  const auto base_frame_id = tf_prefix + params_.base_frame_id;

  auto & odometry_message = realtime_odometry_publisher_->msg_;
  odometry_message.header.frame_id = odom_frame_id;
  odometry_message.child_frame_id = base_frame_id;

  // 오도메트리/TF 발행 주기 설정
  publish_rate_ = params_.publish_rate;
  publish_period_ = rclcpp::Duration::from_seconds(1.0 / publish_rate_);

  // 오도메트리 메시지 초기화
  odometry_message.twist =
    geometry_msgs::msg::TwistWithCovariance(rosidl_runtime_cpp::MessageInitialization::ALL);

  constexpr size_t NUM_DIMENSIONS = 6;
  for (size_t index = 0; index < 6; ++index)
  {
    // 공분산 대각선 값 설정
    const size_t diagonal_index = NUM_DIMENSIONS * index + index;
    odometry_message.pose.covariance[diagonal_index] = params_.pose_covariance_diagonal[index];
    odometry_message.twist.covariance[diagonal_index] = params_.twist_covariance_diagonal[index];
  }

  // tf 퍼블리셔 및 메시지 초기화
  odometry_transform_publisher_ = get_node()->create_publisher<tf2_msgs::msg::TFMessage>(
    DEFAULT_TRANSFORM_TOPIC, rclcpp::SystemDefaultsQoS());
  realtime_odometry_transform_publisher_ =
    std::make_shared<realtime_tools::RealtimePublisher<tf2_msgs::msg::TFMessage>>(
      odometry_transform_publisher_);

  // odom/base_link 변환만 관리
  auto & odometry_transform_message = realtime_odometry_transform_publisher_->msg_;
  odometry_transform_message.transforms.resize(1);
  odometry_transform_message.transforms.front().header.frame_id = odom_frame_id;
  odometry_transform_message.transforms.front().child_frame_id = base_frame_id;

  previous_update_timestamp_ = get_node()->get_clock()->now();

  steering_marker_pub_ = get_node()->create_publisher<visualization_msgs::msg::Marker>(
    "steering_angle_markers", rclcpp::SystemDefaultsQoS());

  return controller_interface::CallbackReturn::SUCCESS;
}

controller_interface::CallbackReturn DoubleSteeringDriveController::on_activate(
  const rclcpp_lifecycle::State &)
{
  // 앞뒤 바퀴, 스티어링 핸들 등록
  const auto front_wheel_result =
    configure_wheel_side("front", params_.front_wheel_names, registered_front_wheel_handles_);
  const auto rear_wheel_result =
    configure_wheel_side("rear", params_.rear_wheel_names, registered_rear_wheel_handles_);
  const auto front_steering_result =
    configure_steering_joint(params_.front_steering_name, registered_front_steering_handle_);
  const auto rear_steering_result =
    configure_steering_joint(params_.rear_steering_name, registered_rear_steering_handle_);

  if (
    front_wheel_result == controller_interface::CallbackReturn::ERROR ||
    rear_wheel_result == controller_interface::CallbackReturn::ERROR ||
    front_steering_result == controller_interface::CallbackReturn::ERROR ||
    rear_steering_result == controller_interface::CallbackReturn::ERROR)  
  {
    return controller_interface::CallbackReturn::ERROR;
  }

  if (registered_front_wheel_handles_.empty() || registered_rear_wheel_handles_.empty() ||
      !registered_front_steering_handle_.has_value() || !registered_rear_steering_handle_.has_value())
  {
    RCLCPP_ERROR(
      get_node()->get_logger(),
      "Either front wheel interfaces, rear wheel interfaces, front steering interfaces, or rear steering interfaces are non existent");
    return controller_interface::CallbackReturn::ERROR;
  }

  is_halted = false;
  subscriber_is_active_ = true;

  RCLCPP_DEBUG(get_node()->get_logger(), "Subscriber and publisher are now active.");
  return controller_interface::CallbackReturn::SUCCESS;
}

controller_interface::CallbackReturn DoubleSteeringDriveController::on_deactivate(
  const rclcpp_lifecycle::State &)
{
  // 비활성화 시 구독자 비활성화 및 바퀴 정지
  subscriber_is_active_ = false;
  if (!is_halted)
  {
    halt();
    is_halted = true;
  }
  registered_front_wheel_handles_.clear();
  registered_rear_wheel_handles_.clear();
  registered_front_steering_handle_.reset();
  registered_rear_steering_handle_.reset();
  return controller_interface::CallbackReturn::SUCCESS;
}

controller_interface::CallbackReturn DoubleSteeringDriveController::on_cleanup(
  const rclcpp_lifecycle::State &)
{
  // 리셋 및 메시지 초기화
  if (!reset())
  {
    return controller_interface::CallbackReturn::ERROR;
  }

  received_velocity_msg_ptr_.set(std::make_shared<Twist>());
  return controller_interface::CallbackReturn::SUCCESS;
}

controller_interface::CallbackReturn DoubleSteeringDriveController::on_error(const rclcpp_lifecycle::State &)
{
  // 에러 발생 시 리셋
  if (!reset())
  {
    return controller_interface::CallbackReturn::ERROR;
  }
  return controller_interface::CallbackReturn::SUCCESS;
}

bool DoubleSteeringDriveController::reset()
{
  // 오도메트리 및 내부 상태 초기화
  odometry_.resetOdometry();

  // 명령 큐 비우기
  std::queue<Twist> empty;
  std::swap(previous_commands_, empty);

  registered_front_wheel_handles_.clear();
  registered_rear_wheel_handles_.clear();
  registered_front_steering_handle_.reset();
  registered_rear_steering_handle_.reset();

  subscriber_is_active_ = false;
  velocity_command_subscriber_.reset();
  velocity_command_unstamped_subscriber_.reset();

  received_velocity_msg_ptr_.set(nullptr);
  is_halted = false;
  return true;
}

void DoubleSteeringDriveController::halt()
{
  // 모든 바퀴를 정지시킵니다.
  const auto halt_wheels = [](auto & wheel_handles)
  {
    for (const auto & wheel_handle : wheel_handles)
    {
      wheel_handle.velocity.get().set_value(0.0);
    }
  };

  halt_wheels(registered_front_wheel_handles_);
  halt_wheels(registered_rear_wheel_handles_);
}

double DoubleSteeringDriveController::wrapAngle(double angle)
{
  while (angle > M_PI) angle -= 2.0 * M_PI;
  while (angle < -M_PI) angle += 2.0 * M_PI;
  return angle;
}

double DoubleSteeringDriveController::computeSpeedScale(double phi_delta, double min_phi_delta, double max_phi_delta, double p, double min_scale)
{
  phi_delta = std::max(phi_delta, 0.0);

  if (phi_delta >= max_phi_delta)
  {
      return min_scale;
  }
  else
  {
      double ratio = phi_delta / max_phi_delta;
      double cos_term = (cos(M_PI * ratio) + 1.0) / 2.0;  // 0~1
      double scale = min_scale + (1.0 - min_scale) * pow(cos_term, p);
      return scale;
  }
}

controller_interface::CallbackReturn DoubleSteeringDriveController::configure_wheel_side(
  const std::string & side, const std::vector<std::string> & wheel_names,
  std::vector<WheelHandle> & registered_handles)
{
  // 각 바퀴 이름에 대해 상태/명령 핸들을 찾아 등록합니다.
  auto logger = get_node()->get_logger();

  if (wheel_names.empty())
  {
    RCLCPP_ERROR(logger, "No '%s' wheel names specified", side.c_str());
    return controller_interface::CallbackReturn::ERROR;
  }

  // 핸들 등록
  registered_handles.reserve(wheel_names.size());
  for (const auto & wheel_name : wheel_names)
  {
    const auto interface_name = wheel_feedback_type();
    const auto state_handle = std::find_if(
      state_interfaces_.cbegin(), state_interfaces_.cend(),
      [&wheel_name, &interface_name](const auto & interface)
      {
        return interface.get_prefix_name() == wheel_name &&
               interface.get_interface_name() == interface_name;
      });

    if (state_handle == state_interfaces_.cend())
    {
      RCLCPP_ERROR(logger, "Unable to obtain joint state handle for %s", wheel_name.c_str());
      return controller_interface::CallbackReturn::ERROR;
    }

    const auto command_handle = std::find_if(
      command_interfaces_.begin(), command_interfaces_.end(),
      [&wheel_name](const auto & interface)
      {
        return interface.get_prefix_name() == wheel_name &&
               interface.get_interface_name() == HW_IF_VELOCITY;
      });

    if (command_handle == command_interfaces_.end())
    {
      RCLCPP_ERROR(logger, "Unable to obtain joint command handle for %s", wheel_name.c_str());
      return controller_interface::CallbackReturn::ERROR;
    }

    registered_handles.emplace_back(
      WheelHandle{std::ref(*state_handle), std::ref(*command_handle)});
  }

  return controller_interface::CallbackReturn::SUCCESS;
}

controller_interface::CallbackReturn DoubleSteeringDriveController::configure_steering_joint(
  const std::string & joint_name,
  std::optional<SteeringHandle> & registered_handle)
{
  auto logger = get_node()->get_logger();

  const auto interface_name = steering_feedback_type();
  const auto state_handle = std::find_if(
    state_interfaces_.cbegin(), state_interfaces_.cend(),
    [&joint_name, &interface_name](const auto & interface)
    {
      return interface.get_prefix_name() == joint_name &&
             interface.get_interface_name() == interface_name;
    });

  if (state_handle == state_interfaces_.cend())
  {
    RCLCPP_ERROR(logger,
                 "Unable to obtain joint state handle for steering joint '%s'",
                 joint_name.c_str());
    return controller_interface::CallbackReturn::ERROR;
  }

  const auto command_handle = std::find_if(
    command_interfaces_.begin(), command_interfaces_.end(),
    [&joint_name](const auto & interface)
    {
      return interface.get_prefix_name() == joint_name &&
             interface.get_interface_name() == hardware_interface::HW_IF_POSITION;
    });

  if (command_handle == command_interfaces_.end())
  {
    RCLCPP_ERROR(logger,
                 "Unable to obtain joint command handle for steering joint '%s'",
                 joint_name.c_str());
    return controller_interface::CallbackReturn::ERROR;
  }

  registered_handle.emplace(
      std::ref(*state_handle),
      std::ref(*command_handle));

  RCLCPP_INFO(logger, "Successfully configured steering joint: %s", joint_name.c_str());
  return controller_interface::CallbackReturn::SUCCESS;
}
}  // namespace double_steering_drive_controller

#include "class_loader/register_macro.hpp"

CLASS_LOADER_REGISTER_CLASS(
  double_steering_drive_controller::DoubleSteeringDriveController, controller_interface::ControllerInterface)
