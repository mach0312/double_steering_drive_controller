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
 * Author: Enrique Fernández
 */

#include "double_steering_drive_controller/double_steering_drive_odometry.hpp"

namespace double_steering_drive_controller
{
Odometry::Odometry(size_t velocity_rolling_window_size)
: timestamp_(0.0),
  x_(0.0),
  y_(0.0),
  heading_(0.0),
  linear_x_(0.0),
  linear_y_(0.0),
  angular_(0.0),
  wheel_base_(0.0),
  front_wheel_radius_(0.0),
  rear_wheel_radius_(0.0),
  front_wheel_old_pos_(0.0),
  rear_wheel_old_pos_(0.0),
  front_steering_old_pos_(0.0),
  rear_steering_old_pos_(0.0),
  velocity_rolling_window_size_(velocity_rolling_window_size),
  linear_x_accumulator_(velocity_rolling_window_size),
  linear_y_accumulator_(velocity_rolling_window_size),
  angular_accumulator_(velocity_rolling_window_size)
{
}

void Odometry::init(const rclcpp::Time & time)
{
  // Reset accumulators and timestamp:
  resetAccumulators();
  timestamp_ = time;
}

// wheel 포지션 기반 제어시 odometry 업데이트
bool Odometry::update(double front_pos, double rear_pos, double front_steering_pos, double rear_steering_pos, double wheel_base, const rclcpp::Time & time)
{
  // We cannot estimate the speed with very small time intervals:
  const double dt = time.seconds() - timestamp_.seconds();
  if (dt < 0.0001)
  {
    return false;  // Interval too small to integrate with
  }

  // Get current wheel joint positions:
  const double front_wheel_cur_pos = front_pos * front_wheel_radius_;
  const double rear_wheel_cur_pos = rear_pos * rear_wheel_radius_;

  // printf("Odometry Update: Front Wheel Cur Pos: %f, Front Wheel Old Pos: %f\n", front_wheel_cur_pos, front_wheel_old_pos_);
  // printf("Odometry Update: Rear Wheel Cur Pos: %f, Rear Wheel Old Pos: %f\n", rear_wheel_cur_pos, rear_wheel_old_pos_);

  // Estimate velocity of wheels using old and current position:
  const double front_wheel_est_vel = front_wheel_cur_pos - front_wheel_old_pos_;
  const double rear_wheel_est_vel = rear_wheel_cur_pos - rear_wheel_old_pos_;

  // Update old position with current:
  front_wheel_old_pos_ = front_wheel_cur_pos;
  rear_wheel_old_pos_ = rear_wheel_cur_pos;

  // printf("Odometry Update: Front Wheel Est Vel: %f, Rear Wheel Est Vel: %f\n", front_wheel_est_vel, rear_wheel_est_vel);

  updateFromVelocity(front_wheel_est_vel, rear_wheel_est_vel, front_steering_pos, rear_steering_pos, wheel_base, time);

  return true;
}

// wheel 속도 기반 제어시 odometry 업데이트
bool Odometry::updateFromVelocity(double front_vel, double rear_vel, double front_steering_pos, double rear_steering_pos, double wheel_base, const rclcpp::Time & time)
{
  const double dt = time.seconds() - timestamp_.seconds();
  // printf("Odometry Update: dt: %f\n", dt);
  if (dt < 0.0001)
  {
    return false;  // Interval too small to integrate with
  }

  // printf("Odometry Update: Front Wheel Velocity: %f, Rear Wheel Velocity: %f\n", front_vel, rear_vel);
  const double v_front_x = front_vel * cos(front_steering_pos);
  const double v_front_y = front_vel * sin(front_steering_pos);
  const double v_rear_x = rear_vel * cos(rear_steering_pos);
  const double v_rear_y = rear_vel * sin(rear_steering_pos);

  // printf("Odometry Update: Front Wheel Vel: (%f, %f), Rear Wheel Vel: (%f, %f)\n", v_front_x, v_front_y, v_rear_x, v_rear_y);

  // 이 친구들은 속도가 아닌 거리값임.
  const double linear_x = (v_front_x + v_rear_x) / 2.0;
  const double linear_y = (v_front_y + v_rear_y) / 2.0;
  const double angular_z = (v_front_y - v_rear_y) / wheel_base;

  // printf("Odometry Update: Linear X: %f, Linear Y: %f, Angular Z: %f\n", linear_x, linear_y, angular_z);

  integrateRungeKutta2(linear_x, linear_y, angular_z);

  timestamp_ = time;

  // Estimate speeds using a rolling mean to filter them out:
  linear_x_accumulator_.accumulate(linear_x / dt);
  linear_y_accumulator_.accumulate(linear_y / dt);
  angular_accumulator_.accumulate(angular_z / dt);

  linear_x_ = linear_x_accumulator_.getRollingMean();
  linear_y_ = linear_y_accumulator_.getRollingMean();
  angular_ = angular_accumulator_.getRollingMean();

  return true;
}

void Odometry::updateOpenLoop(double linear_x, double linear_y, double angular, const rclcpp::Time & time)
{
  /// Save last linear_x, linear_y and angular velocity:
  linear_x_ = linear_x;
  linear_y_ = linear_y;
  angular_ = angular;

  /// Integrate odometry:
  const double dt = time.seconds() - timestamp_.seconds();
  timestamp_ = time;
  integrateExact(linear_x_ * dt, linear_y_ * dt, angular * dt);
}

void Odometry::resetOdometry()
{
  x_ = 0.0;
  y_ = 0.0;
  heading_ = 0.0;
}

void Odometry::setWheelParams(
  double wheel_base, double front_wheel_radius, double rear_wheel_radius)
{
  wheel_base_ = wheel_base;
  front_wheel_radius_ = front_wheel_radius;
  rear_wheel_radius_ = rear_wheel_radius;
}

void Odometry::setVelocityRollingWindowSize(size_t velocity_rolling_window_size)
{
  velocity_rolling_window_size_ = velocity_rolling_window_size;

  resetAccumulators();
}

void Odometry::integrateRungeKutta2(double linear_x, double linear_y, double angular)
{
  const double direction = heading_ + angular * 0.5;

  /// Runge-Kutta 2nd order integration:
  x_ += linear_x * cos(direction) - linear_y * sin(direction);
  y_ += linear_x * sin(direction) + linear_y * cos(direction);
  heading_ += angular;
}

void Odometry::integrateExact(double linear_x, double linear_y, double angular)
{
  if (fabs(angular) < 1e-6)
  {
    integrateRungeKutta2(linear_x, linear_y, angular);
  }
  else
  {
    /// Exact integration (should solve problems when angular is zero):
    // const double heading_old = heading_;
    // const double r = linear_x / angular;
    heading_ += angular;
    double global_dx = linear_x * cos(heading_) - linear_y * sin(heading_);
    double global_dy = linear_x * sin(heading_) + linear_y * cos(heading_);
    x_ += global_dx;
    y_ += global_dy;
  }
}

void Odometry::resetAccumulators()
{
  linear_x_accumulator_ = RollingMeanAccumulator(velocity_rolling_window_size_);
  linear_y_accumulator_ = RollingMeanAccumulator(velocity_rolling_window_size_);
  angular_accumulator_ = RollingMeanAccumulator(velocity_rolling_window_size_);
}

}  // namespace double_steering_drive_controller
