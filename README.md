# Machine Learning for Smart and Connected Systems — Project
## Project Overview
This repository contains my semester-long project for **Machine Learning for Quantified Self**.
The goal of this repository is to document the full project workflow over the semester:
- problem definition
- data understanding
- preprocessing
- feature engineering
- modeling
- evaluation
- iteration
- final conclusions

---

## Team Members
- Jonah Karstens

---

## Project Question
Can AirPod IMU sensor data (accelerometer, gyroscope, and orientation) be used to classify different foods and distinguish eating from non-eating states?

**Example:**  
Can the chewing patterns captured via in-ear motion sensors be used to identify whether someone is eating an apple, chewing gum, eating yogurt/skyr, or simply sitting still?

---

## Dataset
- **Dataset name:** ChewML – AirPod IMU Food Classification Dataset
- **Source:** Self-recorded using the Sensor Logger iOS app with AirPods Pro
- **Type of data:** Time-series IMU data (accelerometer, gyroscope, pitch, roll, yaw) sampled at ~50 Hz
- **Target variable:** Food class (apple, chewing gum, skyr/yogurt, still)
- **Important features:** magnitude_std, chew_band_power, dominant_chew_freq, yaw_range, rot_x_std, rhythmicity
