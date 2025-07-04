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

#ifndef DOUBLE_STEERING_DRIVE_CONTROLLER__ODOMETRY_HPP_
#define DOUBLE_STEERING_DRIVE_CONTROLLER__ODOMETRY_HPP_

#include <cmath>

#include "rclcpp/time.hpp"
#include "rcppmath/rolling_mean_accumulator.hpp"

namespace double_steering_drive_controller
{
class Odometry
{
public:
  explicit Odometry(size_t velocity_rolling_window_size = 10);

  void init(const rclcpp::Time & time);
  bool update(double front_pos, double rear_pos, double front_steering_pos, double rear_steering_pos, const rclcpp::Time & time);
  bool updateFromVelocity(double front_vel, double rear_vel, double front_steering_pos, double rear_steering_pos, const rclcpp::Time & time);
  void updateOpenLoop(double linear_x, double linear_y, double angular, const rclcpp::Time & time);
  void resetOdometry();

  double getX() const { return x_; }
  double getY() const { return y_; }
  double getHeading() const { return heading_; }
  double getLinearX() const { return linear_x_; }
  double getLinearY() const { return linear_y_; }
  double getAngular() const { return angular_; }

  void setWheelParams(double wheel_base, double front_wheel_radius, double rear_wheel_radius);
  void setVelocityRollingWindowSize(size_t velocity_rolling_window_size);

private:
  using RollingMeanAccumulator = rcppmath::RollingMeanAccumulator<double>;

  void integrateRungeKutta2(double linear_x, double linear_y, double angular);
  void integrateExact(double linear_x, double linear_y, double angular);
  void resetAccumulators();

  // Current timestamp:
  rclcpp::Time timestamp_;

  // Current pose:
  double x_;        //   [m]
  double y_;        //   [m]
  double heading_;  // [rad]

  // Current velocity:
  double linear_x_;   //   [m/s]
  double linear_y_;   //   [m/s]
  double angular_;  // [rad/s]

  // Wheel kinematic parameters [m]:
  double wheel_base_;
  double front_wheel_radius_;
  double rear_wheel_radius_;

  // Previous wheel position/state [rad]:
  double front_wheel_old_pos_;
  double rear_wheel_old_pos_;
  double front_steering_old_pos_;
  double rear_steering_old_pos_;

  // Rolling mean accumulators for the linear and angular velocities:
  size_t velocity_rolling_window_size_;
  RollingMeanAccumulator linear_x_accumulator_;
  RollingMeanAccumulator linear_y_accumulator_;
  RollingMeanAccumulator angular_accumulator_;
};

}  // namespace double_steering_drive_controller

#endif  // DOUBLE_STEERING_DRIVE_CONTROLLER__ODOMETRY_HPP_
