# Robotics Maintenance Guild - Fanuc Axis Calibration

## 1. Robot Axis Drift Fault (Error Code: ROB-AXIS-07)

Summary:
Triggered when absolute encoder counts do not align with physical zero markings on Axis 2 (J2) and Axis 3 (J3).

Symptoms
- Position deviation after restart
- TCP offset increases gradually
- Homing procedure fails
- Motion becomes inaccurate

Root Cause
Loss of encoder reference due to battery replacement or mechanical slippage.

Resolution Steps

1. Move the robot to mechanical zero.
2. Verify alignment marks.
3. Open:
Menu → Next → System → Master/Cal
4. Select "Single Axis Master."
5. Calibrate J2.
6. Repeat calibration for J3.
7. Save configuration.
8. Reboot controller.

Verification

Expected Position Error:
Less than ±0.02°

Maintenance Interval
Every 2000 operating hours.
