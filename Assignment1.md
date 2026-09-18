**Instructions for running code:** Run A1_plots.py file for plots to generate. Change parameters in the top of A1_rimless_wheel.py file. Estimate time to run: ~6min. 

**Overview of Files:**
* *A1_analysis.py:* Outlines stability analysis. Includes RoA, 1D Poincare return map, Floquet multiplier estimate, and parameter sweeps. 
* *A1_plots.py:* generates four figures into ./figures/
* *A1_rimless_wheel.py:* Outlines model of the continuous dynamics of the rimless wheel. Includes when leg hits the ground and upon impact. 
* *Assignment1.md:* Markdown file

**Figures:**
* *return_map.png*
* *roa_map.png*
* *sweep_gamma.png*
* *sweep_N.png*

**Sanity Checks:**
I expected the wheel to be more stable at larger slope angle (gamma) since a steeper incline provides more potential energy and momentum, making a continuous forward rolling easier to sustain. The RoA plot at gamma = 0.52 rad (or ~30 degrees) confirms this intuition, as shown in the figure below. (RoA plot at a smaller angle: 0.17 rad is below in RoA section).
<img width="900" height="750" alt="roa_map_0 52rad" src="https://github.com/user-attachments/assets/e934d801-560b-475f-92d8-37b7674eb002" />
In addition, I expected that the simulation starting at theta= 0 with zero angular velocity (thetadot) would remain stationary indefinitely. This state corresponds to the pendulum’s unstable equilibrium and the system should stay at [0,0] with no impacts. My model behaves this way, confirming that both the equilibrium and detection logic is accurate.

**RoA:**
<img width="900" height="750" alt="roa_map" src="https://github.com/user-attachments/assets/99bdb195-11c1-40b9-8c7b-7c53e73c25d7" />

**Return-map**
<img width="825" height="825" alt="return_map" src="https://github.com/user-attachments/assets/aae80a65-9ce9-4054-b5b3-9a5002720445" />

**Visualization and discussion of how slope and number of spokes affect RoA and local convergence:**
<img width="1500" height="600" alt="sweep_gamma" src="https://github.com/user-attachments/assets/127f831a-4cad-4572-9b01-06b8f45998f8" />
The Floquet multiplier is flat at ~0.5 across the entire slope sweep (gamma = 0.08 to 0.35 rad). This shows that the slope has no effect on how quickly the limit cycle locally attracts nearby trajectories. However, the slope does control the forward-rolling motion. The rimless wheel is at 0 during small angles since gravity isn't replenishing enough energy per swing. Past a critical slope (0.05 rad), the forward-rolling continues to grow as the angle increases. 

<img width="1500" height="600" alt="sweep_N" src="https://github.com/user-attachments/assets/de245f9e-7916-4058-99c6-edc41af9103b" />
The number of spokes affects both the RoA and local convergence. The Floquet multiplier increases as N increases since each leg is closer together. As a result, the wheel loses less energy every time a leg hits the ground. Similarly, as you increase N, it is easier for the rimless wheel to get into a stable gait. There is no stability at N=6, but the RoA grows quickly and then levels off around N=10–12.