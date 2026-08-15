# Incident Summary Style Guide & Examples

This document serves as a style reference for the AI Report Generator. The reports must be objective, factual, chronological, and free of absolute "cheating" verdicts. Use the templates and examples below.

---

## Report Structure Requirements
Each report must contain:
1. **Exam Session Metadata**: Session ID, candidate name, date/time, and duration.
2. **Executive Summary**: A high-level overview of the session, including total flag count, average confidence, and highest severity risk detected.
3. **Chronological Incident Timeline**: Table or list detailing every flagged event, its timestamp, duration, severity, and description.
4. **Severity & Policy Analysis**: A breakdown linking the events to specific policy violations (e.g. mobile phone detected maps to Critical Policy violation).
5. **Conclusion / Next Steps**: Recommended action for human reviewer (e.g., "Recommend full video audit of timestamp 10:15 - 10:20").

---

## Example 1: High/Critical Severity (Phone & Gaze)

**Metadata**
- **Session ID**: exam_math_101_alice
- **Duration**: 60 minutes
- **Total Flags**: 3 events

**Executive Summary**
The exam session proceeded normally for the first 40 minutes. However, in the final third of the exam, the CV pipeline detected a prohibited device (mobile phone) and recurrent gaze deviations. The session is flagged for **Critical Review** based on prohibited device policy.

**Chronological Incident Timeline**
| Timestamp | Event Type | Duration | Severity | Description |
| :--- | :--- | :--- | :--- | :--- |
| 10:14:02 | `gaze_deviation` | 6 seconds | Medium | Candidate's gaze shifted off-screen toward the right side. |
| 10:42:15 | `phone_detected` | 4 seconds | Critical | Object identified as a mobile phone detected in the candidate's hands. |
| 10:42:20 | `gaze_deviation` | 12 seconds | High | Candidate's gaze was directed downwards and away from the screen, corresponding with phone detection. |

**Policy Analysis & Context**
- **Mobile Phone Detection**: The detection of a mobile device at 10:42:15 violates the *Prohibited Devices Policy (Section 2)*. This event is classified as Critical.
- **Gaze Deviation**: The gaze deviations at 10:14:02 and 10:42:20 indicate looking away from the monitor. While the first deviation is medium risk, the second deviation is prolonged (12s) and co-occurs with phone detection, indicating a high risk of external materials usage.

**Recommended Action**
- Review the video segment from 10:41:00 to 10:44:00 to manually verify phone usage and candidate gaze direction.

---

## Example 2: Low/Medium Severity (Gaze & No Face)

**Metadata**
- **Session ID**: exam_chem_201_bob
- **Duration**: 90 minutes
- **Total Flags**: 2 events

**Executive Summary**
The exam session is flagged for **Minor Review**. The pipeline detected minor head-pose deviation and a brief face-absence event. These events appear to be typical of posture adjustment or stretching.

**Chronological Incident Timeline**
| Timestamp | Event Type | Duration | Severity | Description |
| :--- | :--- | :--- | :--- | :--- |
| 09:30:12 | `gaze_deviation` | 3 seconds | Minor | Candidate looked away from the monitor briefly. |
| 10:15:45 | `no_face` | 4 seconds | Medium | Candidate's face was missing from the frame. |

**Policy Analysis & Context**
- **Gaze Deviation**: The gaze deviation was brief (3s) and occurred only once, falling under Minor severity.
- **No Face**: The candidate was out of frame for 4 seconds. Under the *No Face Detected Policy (Section 3)*, this falls into Medium severity. The candidate appeared to lean down (potentially to pick up a fallen pen or stretch), returning to the frame immediately.

**Recommended Action**
- Dismiss or approve after a brief inspection of the 10:15:45 timestamp to confirm the candidate was simply picking up an item.
