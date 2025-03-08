#!/usr/bin/env python3
"""
bokeh_plot.py

Interactive visualization of shock wave simulation results.
Allows real-time parameter tuning and validation against experimental data.

Usage:
  bokeh serve --show bokeh_plot.py
"""

import pickle
import numpy as np
from bokeh.plotting import figure, curdoc
from bokeh.models import ColumnDataSource, Slider, Button
from bokeh.layouts import column, row

# Load solver and experimental data from pickle
def load_shock_data():
    """Loads numerical shock wave results and experimental validation data."""
    with open("shock_data.pkl", "rb") as f:
        x_num, rho_num, pressure_num, x_exp, rho_exp = pickle.load(f)
    return x_num, rho_num, pressure_num, x_exp, rho_exp

# Initial data load
x_num, rho_num, pressure_num, x_exp, rho_exp = load_shock_data()
source = ColumnDataSource(data={'x': x_num, 'density': rho_num, 'pressure': pressure_num})
source_exp = ColumnDataSource(data={'x': x_exp, 'density': rho_exp})

# Create Bokeh figure
plot = figure(title="Shock Wave Profile", width=800, height=450)
plot.line('x', 'pressure', source=source, color='blue', line_width=2, legend_label="Numerical Pressure")
plot.line('x', 'density', source=source, color='red', line_width=2, legend_label="Numerical Density")
plot.scatter('x', 'density', source=source_exp, color='black', size=5, legend_label="Experimental Density")

plot.legend.location = "top_left"
plot.xaxis.axis_label = "Position (x)"
plot.yaxis.axis_label = "Density / Pressure"

# Slider for density scale factor
density_slider = Slider(title="Density Scale Factor", start=0.5, end=2.0, value=1.0, step=0.1)

def update_plot(attr, old, new):
    """Updates density profile when slider changes."""
    scale = density_slider.value
    rho_scaled = [r * scale for r in rho_num]
    source.data = {'x': x_num, 'density': rho_scaled, 'pressure': pressure_num}

# Attach event listener
density_slider.on_change('value', update_plot)

# Layout
layout = column(density_slider, plot)
curdoc().add_root(layout)
