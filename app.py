import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint # add this for pendulum

st.set_page_config(layout="wide")
st.title("🧪 Interactive PDE Physics Lab")

sim = st.sidebar.selectbox("Choose Simulation",
    ["Wave: Double Slit", "Heat Diffusion", "Damped Pendulum", "Quantum Particle in Box"])

# --- 1. WAVE: DOUBLE SLIT ---
if sim == "Wave: Double Slit":
    st.header("Wave Equation + Measurement = Quantum Collapse")

    measure_frame = st.sidebar.slider("Measurement Turns ON at Frame", 50, 200, 100)
    frame = st.sidebar.slider("Current Frame", 0, 150, 0)

    N = 200 # lowered for speed on cloud
    x = np.linspace(-10, 10, N)
    y = np.linspace(-10, 10, N)
    X, Y = np.meshgrid(x, y)
    dx = x[1]-x[0]
    dt = 0.01
    c = 1.0

    # barrier
    barrier = np.ones((N,N))
    barrier[np.abs(Y) < 0.1] = 0
    barrier[(np.abs(Y) < 0.1) & (np.abs(X-1) < 0.3)] = 1
    barrier[(np.abs(Y) < 0.1) & (np.abs(X+1) < 0.3)] = 1

    # run simulation up to current frame
    psi = np.exp(1j * 5 * (Y + 10))
    psi_old = psi.copy()

    for f in range(frame):
        lap = (np.roll(psi,1,0)+np.roll(psi,-1,0)+np.roll(psi,1,1)+np.roll(psi,-1,1)-4*psi)/dx**2
        psi_new = 2*psi - psi_old + (c*dt)**2 * lap
        psi_new *= barrier

        if f > measure_frame:
            noise = np.exp(1j*np.random.uniform(0,2*np.pi,psi.shape))
            psi_new[np.abs(Y) < 1] *= noise[np.abs(Y) < 1]

        psi_old, psi = psi, psi_new

    fig, ax = plt.subplots(figsize=(5,5))
    ax.imshow(np.abs(psi)**2, cmap='inferno', extent=[-10,10,-10,10])
    status = "ON - Particles" if frame > measure_frame else "OFF - Waves"
    ax.set_title(f"Measurement: {status}")
    st.pyplot(fig)

# --- 2. HEAT DIFFUSION ---
elif sim == "Heat Diffusion":
    st.header("Heat Equation: $u_t = \\alpha u_{xx}$")
    alpha = st.sidebar.slider("Diffusivity", 0.1, 2.0, 0.5)
    t_steps = st.sidebar.slider("Time Steps", 10, 200, 100)

    N=200; x=np.linspace(0,1,N); dx=x[1]-x[0]
    u = np.sin(np.pi*x)
    for t in range(t_steps):
        u[1:-1] = u[1:-1] + alpha*dt/dx**2 * (u[2:]-2*u[1:-1]+u[:-2])
    fig, ax = plt.subplots()
    ax.plot(x,u)
    st.pyplot(fig)

# --- 3. DAMPED PENDULUM ODE ---
elif sim == "Damped Pendulum":
    st.header("ODE: $\\theta'' + b\\theta' + g\\sin\\theta = 0$")
    b = st.sidebar.slider("Damping", 0.0, 1.0, 0.1)

    def pend(y, t, b, g=9.81):
        th, w = y
        return [w, -b*w - g*np.sin(th)]

    t = np.linspace(0, 10, 500)
    sol = odeint(pend, [np.pi/3, 0], t, args=(b,))
    fig, ax = plt.subplots()
    ax.plot(t, sol[:,0])
    ax.set_xlabel("Time"); ax.set_ylabel("Angle")
    st.pyplot(fig)

# --- 4. QUANTUM BOX ---
elif sim == "Quantum Particle in Box":
    st.header("Schrödinger PDE: Energy Levels")
    n = st.sidebar.slider("Energy Level n", 1, 5, 1)
    x = np.linspace(0,1,200)
    psi = np.sqrt(2)*np.sin(n*np.pi*x)
    fig, ax = plt.subplots()
    ax.plot(x, psi**2)
    ax.set_title(f"Probability Density for n={n}")
    st.pyplot(fig)
