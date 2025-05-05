import streamlit as st
import numpy as np
import os
import pickle
import matplotlib.pyplot as plt
import time

st.set_page_config(layout="wide", page_title="Swarm Letter Formation")

# ---------------- Load Paths ----------------
def loadRobotPaths():
    path = "robotPaths.pkl"
    if os.path.exists(path):
        with open(path, "rb") as f:
            return pickle.load(f)
    return {}

# ---------------- Session State ----------------
if "letterCount" not in st.session_state:
    st.session_state.letterCount = 0
if "robotPaths" not in st.session_state:
    st.session_state.robotPaths = {}
if "history" not in st.session_state:
    st.session_state.history = []
if "activeTab" not in st.session_state:
    st.session_state.activeTab = "🏠 Home"

def nav_button(label, tab):
    if st.button(label):
        st.session_state.activeTab = tab

# ---------------- Home ----------------
if st.session_state.activeTab == "🏠 Home":
    st.title("🤖 Swarm Robotics Letter Formation")

    st.markdown("""
        Welcome to the **Swarm Letter Formation Simulator** — a project that demonstrates the capabilities of **multi-agent robotic systems**
        in forming precise letter structures using principles of **swarm intelligence**, **distributed control**, and **collision avoidance**.
        
        In this simulation, each robot is autonomously assigned to a point on the desired letter, moves from a random starting position,
        and collaborates with its neighbors while avoiding collisions. This is a visual and practical demonstration of how coordinated robot behavior
        can achieve complex formations.

        ### 🔍 Features:
        - **Contour-based target extraction** for precision letter formation.
        - **Greedy target assignment** ensuring optimal path allocation.
        - **Repulsion-based inter-robot collision avoidance.**
        - **2D and 3D path visualization.**
        - Easy-to-use **Streamlit interface** to simulate, view, and review formations.

        ---
        👇 **Choose an option below to get started:**
    """)

    col1, col2, col3 = st.columns(3)
    with col1:
        nav_button("🌀 Go to Simulation", "🌀 Simulation")
    with col2:
        nav_button("📊 View Plot History", "📊 Plot History")
    with col3:
        nav_button("📚 Explanation", "📚 Explanation")

# ---------------- Simulation ----------------
elif st.session_state.activeTab == "🌀 Simulation":
    nav_button("🏠 Return to Home", "🏠 Home")
    st.title("🌀 Letter Formation Simulation")
    letter = st.text_input("Enter a single letter:", max_chars=1).upper()

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("▶️ Run Simulation", use_container_width=True) and letter.isalpha():
            gifFile = f"letter_{letter}.gif"
            if os.path.exists(gifFile):
                os.remove(gifFile)

            with st.spinner(f"Simulating letter '{letter}'..."):
                os.system(f"python swarmLetterLogic.py {letter} 30 2.5 {st.session_state.letterCount + 1}")
                timeout = 30
                elapsed = 0
                while (not os.path.exists(gifFile) or os.path.getsize(gifFile) < 1000) and elapsed < timeout:
                    time.sleep(0.5)
                    elapsed += 0.5

            st.session_state.letterCount += 1
            st.session_state.robotPaths = loadRobotPaths()
            if letter not in st.session_state.history:
                st.session_state.history.append(letter)
                if len(st.session_state.history) > 3:
                    st.session_state.history.pop(0)

    if letter in st.session_state.robotPaths:
        gifFile = f"letter_{letter}.gif"
        st.image(gifFile, caption=f"Simulation: '{letter}'", use_column_width=True)
        st.subheader("📈 Robot Paths")

        fig2d, ax2d = plt.subplots()
        for path in st.session_state.robotPaths[letter]:
            path = np.array(path)
            ax2d.plot(path[:, 0], path[:, 1])
        st.pyplot(fig2d)

        fig3d = plt.figure()
        ax3d = fig3d.add_subplot(111, projection='3d')
        for path in st.session_state.robotPaths[letter]:
            path = np.array(path)
            z = np.arange(len(path))
            ax3d.plot(path[:, 0], path[:, 1], z)
        st.pyplot(fig3d)

    st.markdown("---")
    nav_button("📊 Go to Simulation History", "📊 Plot History")

# ---------------- History ----------------
elif st.session_state.activeTab == "📊 Plot History":
    nav_button("🏠 Return to Home", "🏠 Home")
    st.title("📊 Simulation History")
    if not st.session_state.history:
        st.warning("Run some simulations first!")
    else:
        for letter in reversed(st.session_state.history):
            st.markdown(f"### 🔠 Letter: {letter}")
            gifFile = f"letter_{letter}.gif"

            fig2d, ax2d = plt.subplots()
            for path in st.session_state.robotPaths.get(letter, []):
                path = np.array(path)
                ax2d.plot(path[:, 0], path[:, 1])
            ax2d.set_title("2D Path")

            fig3d = plt.figure()
            ax3d = fig3d.add_subplot(111, projection='3d')
            for path in st.session_state.robotPaths.get(letter, []):
                path = np.array(path)
                z = np.arange(len(path))
                ax3d.plot(path[:, 0], path[:, 1], z)
            ax3d.set_title("3D Path")

            col1, col2, col3 = st.columns(3)
            with col1:
                if os.path.exists(gifFile):
                    st.image(gifFile, caption=f"{letter} Simulation", use_column_width=True)
                else:
                    st.write("GIF not found.")
            with col2:
                st.pyplot(fig2d)
            with col3:
                st.pyplot(fig3d)

    st.markdown("---")

# ---------------- Explanation ----------------
elif st.session_state.activeTab == "📚 Explanation":
    nav_button("🏠 Return to Home", "🏠 Home")
    st.title("📚 Project Explanation & Code Walkthrough")

    st.markdown("---")
    st.header("1️⃣ Backend: Multi-Robot Coordination Logic")

    st.markdown("""
    This simulation is powered by key principles from **swarm robotics** and **multi-agent systems**.

    #### 🧱 Step 1: Target Point Generation using Letter Contours
    A large-size letter is rendered using a font, and OpenCV extracts its boundary contour points.
    These points are then sampled based on the number of robots:

    ```python
    font = ImageFont.truetype("arial.ttf", 220)
    draw.text((20, 20), letter, font=font, fill=255)
    contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    sampled = contour[np.linspace(0, len(contour)-1, numRobots).astype(int)]
    ```

    #### 🎯 Step 2: Assigning Targets to Robots
    Robots are assigned to contour points using a greedy method — each robot is matched to its closest available point:

    ```python
    for tIdx in range(len(targetPoints)):
        for rIdx in unassignedRobots:
            dist = np.linalg.norm(robotPositions[rIdx] - targetPoints[tIdx])
    ```

    #### 🚗 Step 3: Movement with Attraction + Repulsion
    Each robot moves toward its target while avoiding others using a simple artificial potential field:

    ```python
    toTarget = self.target - self.position
    desiredVelocity = (toTarget / distance) * robotSpeed
    self.acceleration += steeringForce

    for other in robots:
        if distance < minDist:
            self.acceleration += repulsion_force
    ```

    #### ✅ Step 4: Convergence Check (Stop Early)
    Simulation stops early when all robots reach their targets:

    ```python
    if all(np.linalg.norm(robot.position - robot.target) < threshold for robot in robots):
        break
    ```

    #### 📹 Step 5: Output
    - Each robot's path is stored in `robotPaths.pkl`.
    - A `.gif` is created to visualize the robot movements.
    """)

    st.markdown("---")
    st.header("2️⃣ Frontend: Streamlit Application Overview")

    st.markdown("""
    The frontend is built using **Streamlit** and provides four main views:
    - 🏠 Home
    - 🌀 Simulation
    - 📊 Plot History
    - 📚 Explanation

    #### 📥 Inputs
    - User inputs a letter using a text box.
    - Triggers a subprocess to run the simulation backend.

    #### 🔄 State Management
    Streamlit session state maintains user history and current robot paths:

    ```python
    if "letterCount" not in st.session_state:
        st.session_state.letterCount = 0
    if "robotPaths" not in st.session_state:
        st.session_state.robotPaths = {}
    ```

    #### 🧠 Running Backend from UI
    The simulation is triggered using:

    ```python
    os.system(f"python swarmLetterLogic.py {letter} 30 2.5 {count}")
    ```

    #### 📊 Plotting Robot Paths
    After simulation, 2D and 3D paths are rendered using `matplotlib`:

    ```python
    fig2d, ax2d = plt.subplots()
    ax2d.plot(path[:, 0], path[:, 1])

    ax3d = fig3d.add_subplot(111, projection='3d')
    ax3d.plot(path[:, 0], path[:, 1], z)
    ```

    #### 💡 Plot History
    Stores up to the **last 3 letters** simulated and shows:
    - Simulation GIF
    - 2D path plot
    - 3D time-evolving path plot

    """)

    st.markdown("---")
    st.header("🔮 Future Work & Research Extensions")

    st.markdown("""
    - ✅ Implement **barrier certificates** for formal collision avoidance guarantees.
    - 🔌 Connect to **Robotarium** to test this logic on real robots.
    - 🔋 Add battery constraints, recharge logic, and real-time obstacle updates.
    - 🔠 Multi-letter formation animations (e.g., 'HELLO') with timed transitions.
    - 🌐 Deploy with dynamic frontend controls and real-time status feedback.
    """)