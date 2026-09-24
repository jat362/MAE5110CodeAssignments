# MAE 5110 - Assignment 2
### Joanna Tan (jat362)

---
header-includes:
  - \usepackage{float}
---

### **1. Sketches**

\begin{center}
\includegraphics[width=0.85\textwidth]{figures/Sketches.png}
\end{center}

### **2. Visualization of RoA**

\begin{center}
\includegraphics[width=0.85\textwidth]{figures/roa_controller.png}
\end{center}

### **3. Choice of Poincaré Section**
The Poincaré section was selected at $\theta = 0$. This corresponds to the stance leg passing through its reference upright orientation and provides a consistent point at which to sample the walker's hybrid dynamics during each step.

The continuous state of the inverted pendulum walker is

$$
x =
\begin{bmatrix}
\theta \\
\dot{\theta}
\end{bmatrix}.
$$

Since $\theta$ is fixed at zero on the section, the state of the walker can be represented by only its angular velocity $\dot{\theta}$, reducing the step-to-step dynamics to a one-dimensional map

$$
\dot{\theta}_{k+1} = P(\dot{\theta}_k, \alpha_k),
$$

This choice of Poincaré section is useful for the step controller because the angle of attack $\alpha$ is selected once per stance phase. At each section crossing, the controller can use the current angular velocity $\dot{\theta}_k$ to select an appropriate $\alpha_k$, then use the Poincaré map to determine the resulting angular velocity on the following step. This process is repeated until the walker reaches the region of attraction of the standing equilibrium, where the continuous-time balancing controller can take over.

### **4. Grid Resolution Verification**

Grid resolution was verified by testing each lookup-table policy against the continuous walker dynamics at 100 fixed off-grid initial velocities between $0.022$ and $4.407$ rad/s. 

**Table header definitions:**

- **n:** resolution
- **Fail Rate:** percentage of trajectories that did not reach the RoA
- **Broken-Promise Rate:** percentage where actual steps exceed predicted steps
- **Exact-Match Rate:** percentage where predicted and actual step counts matched
- **Maximum Error:** largest difference between predicted and actual step counts
- **Worst Actual Steps:** largest number of steps required

| n | Fail Rate | Broken-Promise Rate | Exact-Match Rate | Max Error (steps) | Worst Actual Steps |
|:---:|---:|---:|---:|---:|---:|
| 10  | 0.0% | 60.0% | 40.0% | 2 | 5 |
| 15  | 0.0% | 52.0% | 48.0% | 1 | 4 |
| 20  | 0.0% | 51.0% | 49.0% | 1 | 4 |
| 25  | 0.0% | 48.0% | 52.0% | 1 | 4 |
| 30  | 0.0% | 52.0% | 48.0% | 1 | 4 |
| 35  | 0.0% | 42.0% | 58.0% | 2 | 5 |
| 40  | 0.0% | 28.0% | 72.0% | 2 | 5 |
| 50  | 0.0% | 43.0% | 57.0% | 1 | 4 |
| 60  | 0.0% | 37.0% | 63.0% | 2 | 5 |
| 70  | 0.0% | 12.0% | 88.0% | 1 | 4 |
| 80  | 0.0% | 26.0% | 74.0% | 2 | 5 |
| 90  | 0.0% | 8.0% | 92.0% | 2 | 5 |
| **100** | **0.0%** | **8.0%** | **92.0%** | **1** | **4** |

A final resolution of **n=100** was selected. This had a 0% fail rate, had the lowest broken-promise rate, highest exact-match rate, and only had 1 error. 

### **5. Trajectory Plot**

\begin{center}
\includegraphics[width=0.85\textwidth]{figures/trajectory_comparison.png}
\end{center}

For the initial condition $\dot{\theta}=2.768$ rad/s, the walker requires 4 steps to reach the standing RoA under the fastest-path policy. The maximum-step policy also produces 4 steps, showing that for this initial condition the walker cannot prolong walking beyond four steps before entering the RoA.

### **6. Steps to Standstill Plot**

\begin{center}
\includegraphics[width=0.85\textwidth]{figures/steps_to_stand.png}
\end{center}

The plot shows that the number of steps required to reach the standing RoA increases with the initial angular velocity $\dot{\theta}$. Lower velocities reach standstill in 1 step, intermediate velocities require 2 steps, and higher velocities require 3 steps.