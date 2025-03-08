#!/usr/bin/env python3
"""
plot_bokeh.py

HOW TO RUN:
  1) Install bokeh if needed:  pip install bokeh
  2) In your terminal/cmd:  bokeh serve --show plot_bokeh.py

This will open a browser window with:
  - A slider for the 'omega' parameter
  - A "Run Solver" button
  - A plot that shows 'Pressure' and 'Density' lines, updated whenever the
    slider is changed or the button is clicked.

Currently uses stub functions for:
  - get_initial_state()
  - run_amr_solver_stub()
Replace them with real code from your analytics/numerics modules if you want
a full end-to-end workflow.

"""

import numpy as np
from bokeh.plotting import figure, curdoc
from bokeh.models import ColumnDataSource, Slider, Button
from bokeh.layouts import column

# ---------------- Stub / Example Functions for demonstration ---------------
def get_initial_state(omega):
    """
    Example function to create initial conditions depending on 'omega'.
    In real usage, you might call an analytics function or define boundary
    conditions for your solver with the chosen parameter.
    """
    x_init = np.linspace(0,1,10) 
    U_init = np.zeros((len(x_init), 3))
    for i in range(len(x_init)):
        # Some silly formula that changes density w.r.t. omega:
        rho = 1.0 - (0.9 * x_init[i]) - 0.5*(omega - 0.3)
        if rho < 0.05:
            rho = 0.05
        p = 1e5
        u = 0.0
        # convert to conservative
        # e = p/( (gamma-1)*rho ), for gamma=1.4 => p/(0.4*rho)
        e = p/(0.4*rho)
        E = rho*e + 0.5*rho*u*u
        U_init[i] = [rho, rho*u, E]
    return x_init, U_init

def run_amr_solver_stub(x_init, U_init):
    """
    A stub that simulates running your real solver from numerics_pro
    and returns final x, pressure, density arrays for plotting.
    Replace with actual solver call if you want a real result:
      x_final, U_final, time_end = run_amr_solver(x_init, U_init, ...)
    Then parse U_final to get p, rho, etc.
    """
    x_final = x_init
    density = []
    pressure = []
    for i in range(len(x_init)):
        rho = U_init[i,0]
        E   = U_init[i,2]
        u   = U_init[i,1]/rho if rho>1e-12 else 0
        e_int = (E - 0.5*rho*(u**2))/rho
        # again, for gamma=1.4 => p=0.4*rho*e_int
        p = 0.4*rho*e_int
        density.append(rho)
        pressure.append(p)
    return x_final, pressure, density

# ------------------- Bokeh Visualization Setup ------------------
source = ColumnDataSource(data={'x': [], 'pressure': [], 'density': []})

plot = figure(title="Shock Wave Profile", width=600, height=400)
plot.line('x', 'pressure', source=source, legend_label="Pressure (Pa)",
          line_width=2, color='blue')
plot.line('x', 'density', source=source, legend_label="Density (kg/m³)",
          color='red', line_width=2)
plot.legend.location = "top_left"

omega_slider = Slider(title="Acentric Factor (ω)", start=0.1, end=0.5,
                      value=0.3, step=0.01)
run_button = Button(label="Run Solver", button_type="success")

def update_data():
    """
    1) read slider param
    2) get initial state
    3) run solver stub
    4) update ColumnDataSource
    """
    omega_val = omega_slider.value
    x_init, U_init = get_initial_state(omega_val)
    x_f, p_f, rho_f = run_amr_solver_stub(x_init, U_init)
    source.data = {'x': x_f, 'pressure': p_f, 'density': rho_f}

def on_slider_change(attr, old, new):
    """
    If we want immediate re-run each time slider changes:
      update_data()
    but that might be slow for real solver. For now, let's do nothing
    so user must click the 'Run Solver' button.
    """
    pass

def on_run_button_clicked():
    """
    Actually run the solver & update the plot
    """
    update_data()

omega_slider.on_change('value', on_slider_change)
run_button.on_click(on_run_button_clicked)

# Initial run
update_data()

layout = column(omega_slider, run_button, plot)
curdoc().add_root(layout)
