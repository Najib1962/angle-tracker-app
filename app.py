import streamlit as st
import cv2
import numpy as np
import tempfile
import os
import math
from datetime import datetime
import time
time.sleep(10)  # Wait 10 seconds for Render to initialize

# ------------------ PAGE CONFIG ------------------
st.set_page_config(page_title="Angle Tracker - Click & Move", layout="wide")
st.title("🎯 Angle Tracker - Click to Place, Drag to Move")
st.caption("Click on the image to place points. Use arrow buttons to move them!")

# ------------------ SESSION STATE ------------------
if 'points' not in st.session_state:
    st.session_state.points = []
if 'current_frame' not in st.session_state:
    st.session_state.current_frame = 0
if 'video_path' not in st.session_state:
    st.session_state.video_path = None
if 'total_frames' not in st.session_state:
    st.session_state.total_frames = 0
if 'fps' not in st.session_state:
    st.session_state.fps = 0
if 'prev_gray' not in st.session_state:
    st.session_state.prev_gray = None
if 'angle_history' not in st.session_state:
    st.session_state.angle_history = []
if 'frame_history' not in st.session_state:
    st.session_state.frame_history = []
if 'tracking_enabled' not in st.session_state:
    st.session_state.tracking_enabled = False
if 'selected_point' not in st.session_state:
    st.session_state.selected_point = -1

# ------------------ HELPER FUNCTIONS ------------------
def calculate_angle(p1, p2, p3):
    if p1 is None or p2 is None or p3 is None:
        return 0
    a = np.array([p1[0], p1[1]])
    b = np.array([p2[0], p2[1]])
    c = np.array([p3[0], p3[1]])
    ba = a - b
    bc = c - b
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
    return np.degrees(angle)

def infer_muscle(angle, joint_type="elbow"):
    if joint_type == "elbow":
        if angle < 30:
            return "Triceps (Extension)"
        elif angle < 90:
            return "Brachialis (Flexion)"
        elif angle < 150:
            return "Biceps Brachii (Flexion)"
        else:
            return "Full Extension"
    elif joint_type == "knee":
        if angle < 30:
            return "Quadriceps (Extension)"
        elif angle < 90:
            return "Vastus Medialis"
        elif angle < 150:
            return "Hamstrings (Flexion)"
        else:
            return "Full Flexion"
    elif joint_type == "hip":
        if angle < 30:
            return "Gluteus Maximus (Extension)"
        elif angle < 90:
            return "Gluteus Medius"
        elif angle < 150:
            return "Iliopsoas (Flexion)"
        else:
            return "Full Flexion"
    elif joint_type == "shoulder":
        if angle < 30:
            return "Posterior Deltoid (Extension)"
        elif angle < 90:
            return "Medial Deltoid (Abduction)"
        elif angle < 150:
            return "Anterior Deltoid (Flexion)"
        else:
            return "Latissimus Dorsi"
    else:
        return "Analyzing..."

def track_points(prev_gray, curr_gray, points):
    if len(points) == 0:
        return points
    pts = np.array([[[p[0], p[1]]] for p in points], dtype=np.float32)
    if len(pts) == 0:
        return points
    try:
        next_pts, status, _ = cv2.calcOpticalFlowPyrLK(
            prev_gray, curr_gray, pts, None,
            winSize=(21, 21), maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
        )
        new_points = []
        for i in range(len(points)):
            if status[i][0] == 1:
                x, y = next_pts[i][0]
                new_points.append((int(x), int(y)))
            else:
                new_points.append(points[i])
        return new_points
    except:
        return points

def draw_angle_on_frame(frame, points, angle, joint_type="elbow"):
    img = frame.copy()
    h, w, _ = img.shape
    labels = ['P1', 'P2', 'P3']
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
    
    if len(points) == 3:
        p1, p2, p3 = points
        cv2.line(img, p1, p2, (0, 255, 255), 3)
        cv2.line(img, p2, p3, (0, 255, 255), 3)
        for i, (x, y) in enumerate(points):
            cv2.circle(img, (x, y), 15, colors[i], -1)
            cv2.circle(img, (x, y), 20, colors[i], 3)
            cv2.putText(img, labels[i], (x-15, y+6), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        radius = 50
        start_angle = math.degrees(math.atan2(p1[1] - p2[1], p1[0] - p2[0]))
        end_angle = math.degrees(math.atan2(p3[1] - p2[1], p3[0] - p2[0]))
        if start_angle > end_angle:
            start_angle, end_angle = end_angle, start_angle
        if end_angle - start_angle > 180:
            start_angle, end_angle = end_angle, start_angle
        cv2.ellipse(img, p2, (radius, radius), 0, start_angle, end_angle, (255, 255, 0), 4)
        mid_x = (p1[0] + p2[0] + p3[0]) // 3
        mid_y = (p1[1] + p2[1] + p3[1]) // 3
        cv2.putText(img, f"{angle:.1f}°", (mid_x-45, mid_y-30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 0), 5)
        muscle = infer_muscle(angle, joint_type)
        cv2.putText(img, f"💪 {muscle}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        cv2.putText(img, f"📐 Angle: {angle:.1f}°", (10, 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
    else:
        instructions = [
            f"Points: {len(points)}/3",
            "Click on image to place P1",
            "Click to place P2 (vertex)",
            "Click to place P3"
        ]
        for i, text in enumerate(instructions):
            color = (0, 255, 0) if i <= len(points) and i > 0 else (100, 100, 100) if i == 0 else (100, 100, 100)
            status = "✅" if i <= len(points) and i > 0 else ("📍" if i == len(points) else "⏳")
            if i == 0:
                cv2.putText(img, text, (10, 30 + i*30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            else:
                cv2.putText(img, f"{status} {text}", (10, 30 + i*30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    return img

# ------------------ STREAMLIT UI ------------------

st.sidebar.header("📤 Upload Video")
uploaded_file = st.sidebar.file_uploader("Choose a video", type=["mp4", "avi", "mov"])

joint_type = st.sidebar.selectbox(
    "🦴 Joint Type",
    ["elbow", "knee", "hip", "shoulder"]
)

tracking_enabled = st.sidebar.checkbox("🔄 Auto-Track", value=False)
st.session_state.tracking_enabled = tracking_enabled

# Reset button
if st.sidebar.button("🗑️ Reset All Points"):
    st.session_state.points = []
    st.session_state.selected_point = -1
    st.session_state.angle_history = []
    st.session_state.frame_history = []
    st.rerun()

if uploaded_file is not None:
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    tfile.write(uploaded_file.read())
    video_path = tfile.name
    
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    
    st.session_state.video_path = video_path
    st.session_state.total_frames = total_frames
    st.session_state.fps = fps
    
    st.sidebar.success(f"✅ Video: {total_frames} frames")
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("**🎞️ Navigation**")
    
    frame_number = st.sidebar.slider(
        "Frame", 
        0, total_frames - 1, 
        st.session_state.current_frame, 
        step=1
    )
    st.session_state.current_frame = frame_number
    
    col1, col2 = st.sidebar.columns(2)
    if col1.button("◀️ Prev"):
        st.session_state.current_frame = max(0, st.session_state.current_frame - 1)
        st.rerun()
    if col2.button("Next ▶️"):
        st.session_state.current_frame = min(total_frames - 1, st.session_state.current_frame + 1)
        st.rerun()
    
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, st.session_state.current_frame)
    ret, frame = cap.read()
    cap.release()
    
    if ret:
        h, w, _ = frame.shape
        if w > 900:
            scale = 900 / w
            new_w = 900
            new_h = int(h * scale)
            frame = cv2.resize(frame, (new_w, new_h))
            h, w = new_h, new_w
        
        if tracking_enabled and st.session_state.points and len(st.session_state.points) == 3:
            if st.session_state.prev_gray is not None:
                curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                new_points = track_points(
                    st.session_state.prev_gray, 
                    curr_gray, 
                    st.session_state.points
                )
                for i in range(len(st.session_state.points)):
                    st.session_state.points[i] = new_points[i]
                st.session_state.prev_gray = curr_gray
            else:
                st.session_state.prev_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        angle = 0
        if len(st.session_state.points) == 3:
            angle = calculate_angle(
                st.session_state.points[0],
                st.session_state.points[1],
                st.session_state.points[2]
            )
            if tracking_enabled:
                st.session_state.angle_history.append(angle)
                st.session_state.frame_history.append(st.session_state.current_frame)
                if len(st.session_state.angle_history) > 100:
                    st.session_state.angle_history.pop(0)
                    st.session_state.frame_history.pop(0)
        
        annotated_frame = draw_angle_on_frame(
            frame, 
            st.session_state.points, 
            angle,
            joint_type
        )
        annotated_frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
        
        # ==========================================
        # DISPLAY IMAGE
        # ==========================================
        st.image(annotated_frame_rgb, use_container_width=True)
        
        # Show status
        if len(st.session_state.points) == 3:
            st.success("✅ All 3 points placed!")
        else:
            next_label = ['P1', 'P2', 'P3'][len(st.session_state.points)]
            st.info(f"📍 Click the button below to place {next_label}")
        
        # ==========================================
        # POINT PLACEMENT BUTTONS
        # ==========================================
        st.markdown("---")
        st.subheader("📍 Place Points")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            if st.button("📍 Place P1", use_container_width=True):
                if len(st.session_state.points) == 0:
                    st.session_state.points.append((w//4, h//3))
                    st.rerun()
                else:
                    st.warning("P1 already placed! Use the Move buttons below.")
        
        with col2:
            if st.button("📍 Place P2 (Vertex)", use_container_width=True):
                if len(st.session_state.points) == 1:
                    st.session_state.points.append((w//2, h//2))
                    st.rerun()
                elif len(st.session_state.points) == 0:
                    st.warning("Place P1 first!")
                else:
                    st.warning("P2 already placed!")
        
        with col3:
            if st.button("📍 Place P3", use_container_width=True):
                if len(st.session_state.points) == 2:
                    st.session_state.points.append((3*w//4, h//3))
                    st.rerun()
                elif len(st.session_state.points) < 2:
                    st.warning("Place P1 and P2 first!")
                else:
                    st.warning("P3 already placed!")
        
        with col4:
            if st.button("🗑️ Clear All", use_container_width=True):
                st.session_state.points = []
                st.session_state.selected_point = -1
                st.rerun()
        
        # ==========================================
        # SELECT AND MOVE POINTS
        # ==========================================
        if len(st.session_state.points) == 3:
            st.markdown("---")
            st.subheader("🎯 Select and Move Points")
            st.caption("Select a point below, then use the arrow buttons to move it.")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if st.button("🔴 Select P1", use_container_width=True):
                    st.session_state.selected_point = 0
                    st.success("P1 selected! Use arrow buttons to move.")
            
            with col2:
                if st.button("🟢 Select P2 (Vertex)", use_container_width=True):
                    st.session_state.selected_point = 1
                    st.success("P2 selected! Use arrow buttons to move.")
            
            with col3:
                if st.button("🔵 Select P3", use_container_width=True):
                    st.session_state.selected_point = 2
                    st.success("P3 selected! Use arrow buttons to move.")
            
            # Arrow buttons for moving points
            if st.session_state.selected_point >= 0:
                st.markdown("**Move Selected Point:**")
                move_col1, move_col2, move_col3, move_col4 = st.columns(4)
                
                with move_col1:
                    if st.button("⬆️ Up", use_container_width=True):
                        x, y = st.session_state.points[st.session_state.selected_point]
                        st.session_state.points[st.session_state.selected_point] = (x, max(0, y - 10))
                        st.rerun()
                
                with move_col2:
                    if st.button("⬇️ Down", use_container_width=True):
                        x, y = st.session_state.points[st.session_state.selected_point]
                        st.session_state.points[st.session_state.selected_point] = (x, min(h, y + 10))
                        st.rerun()
                
                with move_col3:
                    if st.button("⬅️ Left", use_container_width=True):
                        x, y = st.session_state.points[st.session_state.selected_point]
                        st.session_state.points[st.session_state.selected_point] = (max(0, x - 10), y)
                        st.rerun()
                
                with move_col4:
                    if st.button("➡️ Right", use_container_width=True):
                        x, y = st.session_state.points[st.session_state.selected_point]
                        st.session_state.points[st.session_state.selected_point] = (min(w, x + 10), y)
                        st.rerun()
        
        # ==========================================
        # METRICS
        # ==========================================
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("🎞️ Frame", f"{st.session_state.current_frame}/{total_frames-1}")
        col2.metric("📐 Angle", f"{angle:.1f}°" if angle else "---")
        col3.metric("📍 Points", f"{len(st.session_state.points)}/3")
        col4.metric("🔍 Tracking", "ON ✅" if tracking_enabled else "OFF")
        
        if len(st.session_state.points) == 3:
            muscle = infer_muscle(angle, joint_type)
            st.success(f"💪 **Active Muscle:** {muscle}")
        
        # ==========================================
        # ANGLE HISTORY GRAPH
        # ==========================================
        if tracking_enabled and len(st.session_state.angle_history) > 1:
            st.subheader("📊 Angle Over Time")
            chart_data = {
                "Frame": st.session_state.frame_history,
                "Angle (°)": st.session_state.angle_history
            }
            st.line_chart(chart_data, x="Frame", y="Angle (°)")
            
            if len(st.session_state.angle_history) > 1:
                avg_angle = sum(st.session_state.angle_history) / len(st.session_state.angle_history)
                min_angle = min(st.session_state.angle_history)
                max_angle = max(st.session_state.angle_history)
                col1, col2, col3 = st.columns(3)
                col1.metric("📊 Average", f"{avg_angle:.1f}°")
                col2.metric("⬇️ Min", f"{min_angle:.1f}°")
                col3.metric("⬆️ Max", f"{max_angle:.1f}°")
        
        # ==========================================
        # EXPORT DATA & REPORT
        # ==========================================
        
        if st.session_state.angle_history or len(st.session_state.points) == 3:
            st.markdown("---")
            st.subheader("📄 Export Report")
            
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("📥 Export CSV", use_container_width=True):
                    data_text = "Frame,Angle(°)\n"
                    for f, a in zip(st.session_state.frame_history, st.session_state.angle_history):
                        data_text += f"{f},{a:.1f}\n"
                    st.download_button(
                        label="Download CSV",
                        data=data_text,
                        file_name=f"angle_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                    )
            
            with col2:
                if st.button("📄 Export Report", use_container_width=True):
                    # Build report content
                    report = []
                    report.append("=" * 60)
                    report.append("ANGLE ANALYSIS REPORT")
                    report.append("=" * 60)
                    report.append(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                    report.append(f"Video: {os.path.basename(video_path) if video_path else 'Unknown'}")
                    report.append(f"Joint: {joint_type.upper()}")
                    report.append(f"Total Frames Analyzed: {len(st.session_state.angle_history)}")
                    report.append("=" * 60)
                    report.append("")
                    
                    if len(st.session_state.points) == 3:
                        p1, p2, p3 = st.session_state.points
                        report.append("POINT POSITIONS:")
                        report.append(f"  P1: {p1}")
                        report.append(f"  P2 (Vertex): {p2}")
                        report.append(f"  P3: {p3}")
                        report.append("")
                        report.append(f"CURRENT ANGLE: {angle:.1f}°")
                        report.append(f"ACTIVE MUSCLE: {infer_muscle(angle, joint_type)}")
                        report.append("")
                    
                    if len(st.session_state.angle_history) > 0:
                        report.append("ANGLE STATISTICS:")
                        report.append(f"  Minimum: {min(st.session_state.angle_history):.1f}°")
                        report.append(f"  Maximum: {max(st.session_state.angle_history):.1f}°")
                        report.append(f"  Average: {sum(st.session_state.angle_history)/len(st.session_state.angle_history):.1f}°")
                        report.append(f"  Range: {max(st.session_state.angle_history) - min(st.session_state.angle_history):.1f}°")
                        report.append("")
                    
                    report.append("ANGLE HISTORY:")
                    report.append("Frame, Angle(°)")
                    for f, a in zip(st.session_state.frame_history[-20:], st.session_state.angle_history[-20:]):
                        report.append(f"{f}, {a:.1f}")
                    
                    if len(st.session_state.angle_history) > 20:
                        report.append(f"... and {len(st.session_state.angle_history) - 20} more frames")
                    
                    report.append("")
                    report.append("=" * 60)
                    report.append("END OF REPORT")
                    report.append("=" * 60)
                    
                    report_text = "\n".join(report)
                    
                    st.download_button(
                        label="Download Report",
                        data=report_text,
                        file_name=f"angle_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                        mime="text/plain"
                    )
        
    else:
        st.error("Failed to read frame")
        
else:
    st.info("👈 Please upload a video from the sidebar")
    st.markdown("""
    ### 📖 How to use:
    
    **Step 1:** Upload a video  
    **Step 2:** Click the buttons to place points:
    - **Place P1** = First point (e.g., Shoulder)
    - **Place P2** = Vertex (e.g., Elbow)  
    - **Place P3** = Third point (e.g., Wrist)
    
    **Step 3:** Select a point and use arrow buttons to move it  
    **Step 4:** Turn on **Auto-Track** ✅  
    **Step 5:** Use the slider to move frames  
    **Step 6:** The **angle updates automatically**!
    """)

st.sidebar.markdown("---")
st.sidebar.caption("🎯 Place points with buttons | Move with arrows")
