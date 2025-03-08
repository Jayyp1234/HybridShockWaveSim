#!/usr/bin/env python3
"""
bokeh_plot.py

Interactive visualization of shock wave simulation results.
Supports multiple EOS models, real-time parameter tuning, and validation against experimental data.

Usage:
  python main.py  (automatically starts Bokeh server)
  OR
  bokeh serve --show visualization/bokeh_plot.py  (if running standalone)
"""

import pickle
import os
from bokeh.plotting import figure, curdoc
from bokeh.models import ColumnDataSource, Select, Slider
from bokeh.layouts import column

# Check if data exists
DATA_FILE = "shock_data.pkl"

def load_shock_data():
    """Loads numerical shock wave results for multiple EOS and experimental validation data."""
    if not os.path.exists(DATA_FILE):
        print(f"Error: Data file '{DATA_FILE}' not found! Run 'python main.py' first.")
        return {}

    with open(DATA_FILE, "rb") as f:
        eos_results = pickle.load(f)  # Dictionary format {EOS: {x, rho, p}}

    return eos_results

# Load data
eos_results = load_shock_data()

# Ensure the experimental dataset is present in eos_results
if "experimental" not in eos_results:
    print("Warning: Experimental data missing in 'shock_data.pkl'.")
    eos_results["experimental"] = {"x": [], "rho": [], "p": []}

# Extract experimental data
x_exp = eos_results["experimental"]["x"]
rho_exp = eos_results["experimental"]["rho"]
pressure_exp = eos_results["experimental"]["p"]

# Default EOS selection
default_eos = next(iter(eos_results.keys()), "PR")  # Pick first available EOS
x_num, rho_num, pressure_num = eos_results.get(default_eos, {"x": [], "rho": [], "p": []}).values()

# Create data sources
source_density = ColumnDataSource(data={'x': x_num, 'density': rho_num})
source_pressure = ColumnDataSource(data={'x': x_num, 'pressure': pressure_num})

source_exp_density = ColumnDataSource(data={'x': x_exp, 'density': rho_exp})
source_exp_pressure = ColumnDataSource(data={'x': x_exp, 'pressure': pressure_exp})

# Create Bokeh figures
plot_density = figure(title="Shock Wave Density Profile", width=700, height=350)
plot_density.line('x', 'density', source=source_density, line_width=2, legend_label="Numerical Density", color="blue")
plot_density.scatter('x', 'density', source=source_exp_density, color="red", size=5, legend_label="Experimental Density")
plot_density.legend.location = "top_right"

plot_pressure = figure(title="Shock Wave Pressure Profile", width=700, height=350)
plot_pressure.line('x', 'pressure', source=source_pressure, line_width=2, legend_label="Numerical Pressure", color="green")
plot_pressure.scatter('x', 'pressure', source=source_exp_pressure, color="red", size=5, legend_label="Experimental Pressure")
plot_pressure.legend.location = "top_right"

# Dropdown menu to select EOS
eos_selector = Select(title="Select EOS Model", value=default_eos, options=list(eos_results.keys()))

# Slider for scaling density
density_slider = Slider(title="Density Scale Factor", start=0.5, end=2.0, value=1.0, step=0.1)

# Slider for scaling pressure
pressure_slider = Slider(title="Pressure Scale Factor", start=0.5, end=2.0, value=1.0, step=0.1)

def update_plot(attr, old, new):
    """Updates plots when EOS selection or scale sliders change."""
    selected_eos = eos_selector.value
    scale_density = density_slider.value
    scale_pressure = pressure_slider.value

    if selected_eos in eos_results:
        x_new, rho_new, pressure_new = eos_results[selected_eos]["x"], eos_results[selected_eos]["rho"], eos_results[selected_eos]["p"]
        source_density.data = {'x': x_new, 'density': [r * scale_density for r in rho_new]}
        source_pressure.data = {'x': x_new, 'pressure': [p * scale_pressure for p in pressure_new]}

# Attach event listeners
eos_selector.on_change('value', update_plot)
density_slider.on_change('value', update_plot)
pressure_slider.on_change('value', update_plot)

# Ensure the first update happens to set correct data
update_plot(None, None, None)

# Layout
curdoc().add_root(column(eos_selector, density_slider, plot_density, pressure_slider, plot_pressure))

def start_bokeh_server():
    """Launches the Bokeh server from within Python (for automatic execution in main.py)."""
    os.system("bokeh serve --show visualization/bokeh_plot.py --port 5010")
