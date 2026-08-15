# Exam Session Audit Report
## Metadata
- **Session ID**: exam_session_demo_999
- **Candidate Name**: John Doe
- **Exam Name**: Computer Science 101 Final
- **Date**: 2026-08-11
- **Total Duration**: 90 minutes

## Executive Summary
The candidate's exam session is flagged for review due to minor gaze deviations and a brief face-absence event. These events appear to be typical of posture adjustment or stretching.

## Chronological Incident Timeline
| Timestamp | Event Type | Duration | Severity | Description |
| :--- | :--- | :--- | :--- | :--- |
| 10:05:12 | `gaze_deviation` | 4.5 seconds | Minor | Candidate looked away from the monitor briefly. |
| 10:15:30 | `no_face` | 2.5 seconds | Medium | Candidate's face was missing from the frame. |
| 10:35:45 | `phone_detected` | 3.8 seconds | Critical | Object identified as a mobile phone detected in the candidate's hands. |
| 10:35:50 | `gaze_deviation` | 11.2 seconds | High | Candidate's gaze was directed downwards and away from the screen, corresponding with phone detection. |

## Policy Analysis & Context
- **Gaze Deviation**: The first event is classified as Minor risk due to a brief duration (4.5s) and frequency (only once). The second event is classified as High risk due to prolonged duration (11.2s) and co-occurrence with phone detection.
- **No Face**: The candidate's face was missing from the frame for 2.5 seconds, falling under Medium severity. This appears to be a brief posture adjustment or stretching.

## Recommended Action
- Review the video segment from 10:30:00 to 10:35:00 to manually verify phone usage and candidate gaze direction.
- Inspect the 10:15:30 timestamp to confirm the candidate was simply picking up an item.