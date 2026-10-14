"""
VisionGuard AI - Interactive Web Dashboard
Module: app.py
Streamlit web interface for real-time video analytics, tracking, zone intrusion, and flow monitoring.
"""

import os
import tempfile
import cv2
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from PIL import Image

from src.detector import ObjectDetector
from src.tracker import MultiObjectTracker
from src.analytics import AnalyticsEngine
from src.visualizer import Visualizer
from src.utils import FPSCounter, create_video_writer

# Page Configuration
st.set_page_config(
    page_title="VisionGuard AI | Intelligent Video Analytics",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .metric-card {
        background-color: #1e222b;
        border: 1px solid #30363d;
        border-radius: 8px;
        padding: 15px;
        text-align: center;
    }
    .metric-title {
        color: #8b949e;
        font-size: 0.85rem;
        margin-bottom: 5px;
    }
    .metric-val {
        color: #58a6ff;
        font-size: 1.8rem;
        font-weight: 700;
    }
</style>
""", unsafe_allow_html=True)


def main():
    st.title("🛡️ VisionGuard AI: Intelligent Video Analytics & Surveillance")
    st.markdown("""
    **Real-Time Multi-Object Detection, Multi-Target Tracking, Tripwire Counting & Restricted Zone Intrusion Monitoring.**
    """)

    # Sidebar Controls
    st.sidebar.header("⚙️ Configuration & Inputs")

    input_source = st.sidebar.selectbox(
        "Select Input Source",
        [
            "Built-in Benchmark Video",
            "Static Benchmark Image",
            "Upload Custom Video",
            "Upload Custom Image"
        ]
    )

    st.sidebar.subheader("Detection & Tracking Parameters")
    conf_thresh = st.sidebar.slider("Confidence Threshold", 0.1, 1.0, 0.40, 0.05)
    nms_thresh = st.sidebar.slider("NMS Threshold", 0.1, 0.9, 0.35, 0.05)
    min_area = st.sidebar.slider("Minimum Object Area (px)", 100, 5000, 600, 100)

    st.sidebar.subheader("Visualization Toggles")
    show_trails = st.sidebar.checkbox("Show Movement Trails", value=True)
    show_hud = st.sidebar.checkbox("Show HUD Telemetry Bar", value=True)
    show_line = st.sidebar.checkbox("Enable Tripwire Counting Line", value=True)
    show_zone = st.sidebar.checkbox("Enable Restricted Intrusion Zone", value=True)

    # Initialize components
    detector = ObjectDetector(
        mode="motion",
        confidence_threshold=conf_thresh,
        nms_threshold=nms_thresh,
        min_area=min_area
    )
    tracker = MultiObjectTracker(max_lost=15, iou_threshold=0.25, max_distance=110.0)
    analytics = AnalyticsEngine()
    visualizer = Visualizer(draw_trails=show_trails, draw_hud=show_hud)
    fps_counter = FPSCounter()

    # Setup zones & lines
    if show_line:
        analytics.add_line("Main-Tripwire", (400, 120), (400, 420))
    if show_zone:
        analytics.add_zone("Restricted-Zone", [(80, 140), (300, 140), (300, 240), (80, 240)])

    # Setup Source
    temp_video_path = None
    input_image = None

    if input_source == "Built-in Benchmark Video":
        sample_video_path = "data/samples/sample_traffic.mp4"
        if not os.path.exists(sample_video_path):
            from sample_generator import generate_benchmark_video
            generate_benchmark_video(sample_video_path)
        temp_video_path = sample_video_path

    elif input_source == "Static Benchmark Image":
        sample_img_path = "data/samples/sample_frame.jpg"
        if not os.path.exists(sample_img_path):
            from sample_generator import generate_benchmark_image
            generate_benchmark_image(sample_img_path)
        input_image = cv2.imread(sample_img_path)

    elif input_source == "Upload Custom Video":
        uploaded_file = st.sidebar.file_uploader("Upload MP4 / AVI / MOV", type=["mp4", "avi", "mov"])
        if uploaded_file is not None:
            tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
            tfile.write(uploaded_file.read())
            temp_video_path = tfile.name

    elif input_source == "Upload Custom Image":
        uploaded_img = st.sidebar.file_uploader("Upload Image", type=["jpg", "jpeg", "png", "webp"])
        if uploaded_img is not None:
            file_bytes = np.asarray(bytearray(uploaded_img.read()), dtype=np.uint8)
            input_image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    # UI Metric KPI placeholders
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        metric_active = st.empty()
    with col2:
        metric_total = st.empty()
    with col3:
        metric_cross = st.empty()
    with col4:
        metric_alerts = st.empty()

    metric_active.metric("Active Targets", 0)
    metric_total.metric("Total Unique Detected", 0)
    metric_cross.metric("Line Flow (In / Out)", "0 / 0")
    metric_alerts.metric("Intrusion Alerts", 0)

    # Process Static Image
    if input_image is not None:
        st.subheader("🖼️ Static Image Analytics")
        detections = detector.detect(input_image)
        tracks = tracker.update(detections)
        summary = analytics.process_frame(tracks)

        annotated = input_image.copy()
        if show_zone:
            annotated = visualizer.draw_zones(annotated, analytics.zones)
        if show_line:
            annotated = visualizer.draw_lines(annotated, analytics.lines)
        annotated = visualizer.draw_tracks(annotated, tracks)
        if show_hud:
            annotated = visualizer.draw_hud(annotated, 30.0, len(tracks), summary["total_unique_objects"], len(analytics.event_log))

        rgb_view = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
        st.image(rgb_view, use_container_width=True, caption="Processed Image with Detection & Bounding Overlays")

        metric_active.metric("Active Targets", len(tracks))
        metric_total.metric("Total Unique Detected", summary["total_unique_objects"])
        metric_alerts.metric("Intrusion Alerts", len(analytics.event_log))

        # Show Detections Table
        if detections:
            st.subheader("📋 Detected Objects Table")
            st.dataframe(pd.DataFrame(detections), use_container_width=True)

    # Process Video
    elif temp_video_path is not None:
        st.subheader("📹 Live Video Stream & Analytics Feed")

        start_btn = st.button("▶️ Start Video Analysis", type="primary")

        video_placeholder = st.empty()
        chart_col1, chart_col2 = st.columns(2)
        with chart_col1:
            chart_placeholder1 = st.empty()
        with chart_col2:
            chart_placeholder2 = st.empty()

        log_placeholder = st.empty()

        if start_btn:
            cap = cv2.VideoCapture(temp_video_path)
            fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

            frame_idx = 0
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret or frame is None:
                    break

                frame_idx += 1
                curr_fps = fps_counter.tick()

                # Detection & Tracking
                detections = detector.detect(frame)
                tracks = tracker.update(detections)
                summary = analytics.process_frame(tracks)

                # Overlays
                annotated = frame.copy()
                if show_zone:
                    annotated = visualizer.draw_zones(annotated, analytics.zones)
                if show_line:
                    annotated = visualizer.draw_lines(annotated, analytics.lines)
                annotated = visualizer.draw_tracks(annotated, tracks)
                if show_hud:
                    annotated = visualizer.draw_hud(
                        annotated,
                        curr_fps,
                        len(tracks),
                        summary["total_unique_objects"],
                        len(analytics.event_log)
                    )

                # Render frame in Streamlit
                rgb_frame = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
                video_placeholder.image(rgb_frame, use_container_width=True)

                # Update KPI Metrics
                metric_active.metric("Active Targets", len(tracks))
                metric_total.metric("Total Unique Detected", summary["total_unique_objects"])
                if "Main-Tripwire" in analytics.lines:
                    line = analytics.lines["Main-Tripwire"]
                    metric_cross.metric("Line Flow (In / Out)", f"{line.count_in} / {line.count_out}")
                metric_alerts.metric("Intrusion Alerts", len(analytics.event_log))

                # Update Charts every 10 frames
                if frame_idx % 10 == 0:
                    dist = summary["class_distribution"]
                    if dist:
                        df_dist = pd.DataFrame(list(dist.items()), columns=["Class", "Count"])
                        fig1 = px.bar(
                            df_dist,
                            x="Class",
                            y="Count",
                            title="Object Class Distribution",
                            color="Class",
                            template="plotly_dark"
                        )
                        fig1.update_layout(height=280, margin=dict(l=20, r=20, t=40, b=20))
                        chart_placeholder1.plotly_chart(fig1, use_container_width=True)

                    # Event Log Table
                    df_events = analytics.get_summary_dataframe()
                    if not df_events.empty:
                        log_placeholder.dataframe(df_events.tail(10), use_container_width=True)

            cap.release()
            st.success("🎉 Video Analysis Completed!")

            # Final summary & CSV download
            df_final = analytics.get_summary_dataframe()
            if not df_final.empty:
                csv_data = df_final.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label="📥 Download Event & Audit Log (CSV)",
                    data=csv_data,
                    file_name="visionguard_analytics_report.csv",
                    mime="text/csv"
                )
    else:
        st.info("👈 Please select an input source or upload media from the sidebar to begin.")


if __name__ == "__main__":
    main()
