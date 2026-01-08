# Copilot Instructions: Physics Practical Programming 2025

This codebase handles a physical programming exam covering symbol computation (MMA) and numerical simulation (Python).

## Big Picture Architecture
- **Structure**: Organized into `PartX_Type` folders corresponding to exam tasks.
- **Documentation**: Each folder contains its own `Process_TaskX.md` which serves as the "source of truth" for that task's grading criteria (35% on process recording).
- **Tooling**: Python 3.14.2 (Standard scientific stack: NumPy, SciPy, Matplotlib) and Mathematica (MMA).

## Critical Workflows
- **Mathematica Tasks**: Code is stored as `.txt` in `Part1_MMA/`. 
  - Agents should provide snippets from these for users to copy into `.nb` files.
  - **Direct Visualization**: Plots MUST be displayed directly in the notebook (remove semicolon if needed) in addition to being exported.
  - **Optimization**: Use `NSolve` with appropriate constraints or initial seeds to avoid long computation times.
- **Python Simulations**: Standalone scripts in `Part2_Math/`, `Part3_Physics/`, and `Part4_Open/`.
- **Visualization**: All scripts must save visual outputs as `.png` files (e.g., `Export["name.png", plot]` in MMA or `plt.savefig("name.png")` in Python).
- **Process Recording**: Every task edit MUST be accompanied by an update to the respective `Process_TaskX.md` documenting:
  - Theoretical reasoning/Physical principles.
  - Step-by-step implementation logic.
  - Failures and corrections (e.g., Numerical stability boundaries).

## Project-Specific Conventions
### Python Execution
- Use `scipy.integrate.solve_ivp` with `terminal=True` events for ODEs (see `ProjectileMotion.py`).
- For PDEs (Heat Equation), manually check and enforce stability conditions (e.g., $r = \frac{\alpha \Delta t}{(\Delta x)^2} \leq 0.5$).
- Interactive tools use `matplotlib.widgets` instead of IPython widgets for standalone `.py` compatibility.

### Mathematica Patterns
- Explicitly define coordinate systems in `Grad`, `Curl`, `Div` (e.g., `{x, y, z}`).
- Use `FullSimplify` for symbolic verification of identities.

## Data Flow & Integration
- **Inputs**: Defined parameters in scripts (e.g., $L=1.0, \alpha=0.01$ in `HeatConduction.py`).
- **Outputs**: Console logs for numerical results and `.png` images for visual validation.

## AI Interaction Requirements
- **Process Document Sync**: MUST update the corresponding `Process_TaskX.md` in the task folder in EVERY interaction. 
  - **Sections**: Each task document must have sub-sections for `Theoretical Reasoning`, `Step-by-step Implementation`, and `Issues & Resolutions`. 
- Prioritize **Python** for numerical tasks.
- Keep answers short, impersonal, and avoid backticks for file links.
- Always prompt for user testing after completing a discrete task.
