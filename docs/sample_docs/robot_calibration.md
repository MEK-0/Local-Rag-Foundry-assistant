# Robotics Maintenance Guild - Fanuc Axis Calibration Calibration

## 1. Robot Axis Drift Fault (Error Code: ROB-AXIS-07)
- **Summary**: Triggered when absolute encoder counts do not align with physical zero markings on Axis 2 (J2) and Axis 3 (J3).
- **Step-by-step Guidance**:
  1. Drive the robotic system manually into its mechanical zero calibration position alignment mark.
  2. Navigate on the Teach Pendant unit to: Menu -> Next -> System -> Master/Cal.
  3. Execute 'Single Axis Master' for J2 and J3 modules sequentially, then apply updates.