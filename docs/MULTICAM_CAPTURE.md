# Multicamera Session research workflow (4.8.0)

1. Probe actual readable video devices; enable 1–3 channels.
2. Perform independent and combined preflight using the exact requested MJPEG / width / height / FPS.
3. Verify ongoing live previews; then start all enabled cameras with shared host monotonic reference.
4. Continuously record MJPEG AVIs and `_metadata/timestamps_*.csv`.
5. Snapshot experimental condition and parameter fields at every phase transition; all cameras remain uninterrupted.
6. Save `session_record.csv` as the human-facing session log; keep technical metadata under `_metadata`.
7. For analysis, read video frames and per-frame timestamps, extract independent ROI features per camera, build state/baseline reference, associate compatible event candidates across camera streams, and summarize stages per camera.
8. Keep reports/evidence exploratory until human validation. Synchronization is host-receipt time, not exposure time.
