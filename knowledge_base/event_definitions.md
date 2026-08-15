# Event Definitions & Severity Ratings

This document outlines the computer-vision-detected events and how they map to academic integrity risk severity levels (Minor, Medium, High, Critical).

---

## 1. Gaze Deviation (`gaze_deviation`)
- **Definition**: The candidate's head orientation or gaze is directed away from the screen for an extended period.
- **Rules & Severity**:
  - **Minor Risk**: Gaze deviation lasting between 2 and 5 seconds, occurring fewer than 5 times during an exam session.
  - **Medium Risk**: Gaze deviation lasting between 5 and 10 seconds, or recurring more than 5 times in a 30-minute window.
  - **High Risk**: Persistent gaze deviation (more than 10 seconds continuous) or frequent patterns suggesting external assistance (e.g., reading from notes placed next to the monitor).

---

## 2. Multiple Faces Detected (`multiple_faces`)
- **Definition**: More than one human face is visible in the video frame at the same time.
- **Rules & Severity**:
  - **High Risk**: Detection of a second face in the background for a brief moment (e.g., someone walking behind the candidate in a public space).
  - **Critical Risk**: Multiple faces visible simultaneously looking at the screen or communicating, indicating direct collaboration.

---

## 3. No Face Detected (`no_face`)
- **Definition**: The candidate's face is completely missing from the camera frame.
- **Rules & Severity**:
  - **Minor Risk**: Temporary absence (under 3 seconds) which could be due to a sudden movement or posture adjustment.
  - **Medium Risk**: Absence between 3 and 10 seconds (e.g., dropping a pen, tying shoelaces).
  - **High Risk**: Absence exceeding 10 seconds, suggesting the candidate left the testing station without authorization.

---

## 4. Mobile Phone Detected (`phone_detected`)
- **Definition**: An object classified as a mobile phone or cell phone is detected in the candidate's hands or immediate proximity.
- **Rules & Severity**:
  - **Critical Risk**: A mobile phone is held, viewed, or operated during the examination. This is an immediate violation of exam conditions.
