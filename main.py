#!/usr/bin/env python3
"""
main.py

A single script that:
1) Defines a minimal finite-volume + AMR solver with a max cell cap.
2) Uses Bokeh for interactive slider to adjust a parameter (e.g. omega),
3) Re-runs the solver each time the slider changes, and
4) Plots Pressure & Density in real-time in the browser.

Run with:
   bokeh serve --show main.py
"""

import numpy as np
from bokeh.plotting import figure, curdoc
from bokeh.models import ColumnDataSource, Slider, Button
from bokeh.layouts import column, row

# ==================== 1) EOS & Helper Functions =====================
def eos_pressure(rho, e_int, gamma=1.4):
    """Ideal gas: p = (gamma-1)*rho*e. For demonstration."""
    return (gamma - 1)*rho*e_int

def eos_sound_speed(rho, p, gamma=1.4):
    """c = sqrt(gamma * p / rho)."""
    if rho < 1e-12 or p < 0:
        return 0.0
    return np.sqrt(gamma * p / rho)

def cons_to_prim(U, gamma=1.4):
    """
    Convert [rho, rho*u, E] -> (rho, u, p).
    E = rho e + 0.5*rho u^2, p=(gamma-1)*rho e
    """
    rho = U[0]
    if rho<1e-12: rho=1e-12
    u = U[1]/rho
    e_int = (U[2] - 0.5*rho*u*u)/rho
    p = eos_pressure(rho, e_int, gamma=gamma)
    return (rho, u, p)

def prim_to_cons(rho, u, p, gamma=1.4):
    """(rho, u, p)->[rho, rho*u, E]. E=rho e+0.5*rho u^2,e = p/( (gamma-1)*rho )."""
    e_int = p/((gamma-1)*rho)
    E = rho*e_int + 0.5*rho*u*u
    return np.array([rho, rho*u, E])

# ==================== 2) Riemann Solver (Rusanov) =====================
def rusanov_flux(UL, UR, gamma=1.4):
    rhoL,uL,pL = cons_to_prim(UL,gamma=gamma)
    rhoR,uR,pR = cons_to_prim(UR,gamma=gamma)
    FL = np.array([rhoL*uL,
                   rhoL*uL*uL + pL,
                   (UL[2]+pL)*uL])
    FR = np.array([rhoR*uR,
                   rhoR*uR*uR + pR,
                   (UR[2]+pR)*uR])
    cL = eos_sound_speed(rhoL,pL,gamma=gamma)
    cR = eos_sound_speed(rhoR,pR,gamma=gamma)
    s_max= max(abs(uL)+cL, abs(uR)+cR)
    return 0.5*(FL+FR) - 0.5*s_max*(UR-UL)

# ==================== 3) Finite-Volume Step =====================
def finite_volume_step(x,U,dt,gamma=1.4):
    nx = len(x)
    Unew = U.copy()
    # flux array
    F = np.zeros((nx+1, 3))

    UL_bound = U[0]
    UR_bound = U[-1]

    # compute flux at each interface
    for i in range(nx+1):
        if i==0:
            FL_ = UL_bound
            FR_ = U[0]
        elif i==nx:
            FL_ = U[-1]
            FR_ = UR_bound
        else:
            FL_ = U[i-1]
            FR_ = U[i]
        F[i] = rusanov_flux(FL_,FR_, gamma=gamma)

    for i in range(nx):
        if nx>1:
            if i< nx-1:
                dx_i = x[i+1]-x[i]
            else:
                dx_i = (x[i]-x[i-1]) if i>0 else 0.01
        else:
            dx_i= 0.01
        if abs(dx_i)<1e-12:
            dx_i=1e-12

        Unew[i] = U[i] - (dt/dx_i)*(F[i+1]-F[i])
    return Unew

# ==================== 4) AMR refine/coarsen =====================
def refine_coarsen(x,U, refine_thresh=0.2, coarsen_thresh=0.05, gamma=1.4):
    nx=len(x)
    new_x=[]
    new_U=[]
    i=0
    while i<nx:
        if i==0 or i== nx-1:
            new_x.append(x[i])
            new_U.append(U[i])
            i+=1
            continue
        rho_curr, _, _= cons_to_prim(U[i], gamma=gamma)
        rho_prev, _, _= cons_to_prim(U[i-1], gamma=gamma)
        dx_i= x[i] - x[i-1]
        if abs(dx_i)<1e-12: dx_i=1e-12
        grad_rho= abs(rho_curr-rho_prev)/dx_i

        if grad_rho> refine_thresh:
            # refine
            x_left= x[i]-0.25*dx_i
            x_right= x[i]+0.25*dx_i
            halfU=0.5*U[i]
            new_x.append(x_left)
            new_U.append(halfU.copy())
            new_x.append(x_right)
            new_U.append(halfU.copy())
            i+=1
        elif grad_rho< coarsen_thresh and i< nx-1:
            # coarsen
            mergedU= U[i]+ U[i+1]
            mergedX=0.5*(x[i]+ x[i+1])
            new_x.append(mergedX)
            new_U.append(mergedU)
            i+=2
        else:
            new_x.append(x[i])
            new_U.append(U[i])
            i+=1

    x_new= np.array(new_x)
    U_new= np.array(new_U)
    idx= np.argsort(x_new)
    return x_new[idx],U_new[idx]

# ==================== 5) Overall solver with max_cells =====================
def run_amr_solver(x_init,U_init,tmax=0.0002, cfl=0.8,
                   refine_thresh=0.2, coarsen_thresh=0.05, gamma=1.4,
                   max_cells=50000):
    x= x_init
    U= U_init
    time=0.0
    while time< tmax:
        # compute dt
        smax=0.0
        for i in range(len(x)):
            r,u,p= cons_to_prim(U[i], gamma=gamma)
            c= eos_sound_speed(r,p,gamma=gamma)
            speed= abs(u)+c
            if speed> smax:
                smax= speed
        if smax<1e-12:
            dt=1e-6
        else:
            # minimal dx
            min_dx=1e9
            for j in range(len(x)-1):
                dd= abs(x[j+1]-x[j])
                if dd< min_dx:
                    min_dx= dd
            dt= cfl* min_dx / smax

        if time+ dt> tmax:
            dt= tmax- time

        U= finite_volume_step(x,U,dt,gamma=gamma)
        time+= dt

        x,U= refine_coarsen(x,U, refine_thresh, coarsen_thresh,gamma=gamma)

        # cap max cells
        if len(x)> max_cells:
            x= x[:max_cells]
            U= U[:max_cells]
            # or do other approach
    return x, U, time

# ==================== 6) Bokeh Visualization  =====================
from bokeh.models import ColumnDataSource, Slider, Button
from bokeh.layouts import column

source= ColumnDataSource(data={'x':[], 'pressure':[], 'density':[]})
plot= figure(title="Shock Wave w/ AMR", width=700, height=400,
             x_axis_label='x', y_axis_label='Value')
# We plot pressure in one color, density in another
plot.line('x','pressure', source=source, color='blue', legend_label='Pressure(Pa)', line_width=2)
plot.line('x','density', source=source, color='red', legend_label='Density(kg/m^3)', line_width=2)
plot.legend.location="top_left"

omega_slider= Slider(title="Parameter (ω)", start=0.1, end=0.5, value=0.3, step=0.01)
run_button= Button(label="Run AMR Solver", button_type="success")

def run_solver_with_omega(omega):
    """
    We define a small domain or vary initial conditions w.r.t. omega
    Then run solver. Return x, p, rho for bokeh.
    """
    # create initial conditions
    nx0= 20
    x_init= np.linspace(0,1,nx0)
    U_init= np.zeros((nx0,3))
    # e.g. left side: rho=1, p=1e5 minus some fraction of ω
    for i in range(nx0):
        if x_init[i]<0.5:
            rho= 1.0
            p= 1e5
        else:
            rho= 0.125
            p= 1e4
        # slightly vary it with omega
        p*= (1+ 0.2*(omega-0.3))
        # convert
        e_int= p/((1.4-1)*rho)
        E= rho* e_int
        U_init[i]=[rho,0,E]

    x_final,U_final,_= run_amr_solver(x_init,U_init,tmax=0.0002,
                                      cfl=0.8, refine_thresh=0.2,
                                      coarsen_thresh=0.05, gamma=1.4,
                                      max_cells=50000)
    # Convert final state to arrays for pressure, density
    pressure_list=[]
    density_list=[]
    for i in range(len(x_final)):
        r,u,p= cons_to_prim(U_final[i], gamma=1.4)
        pressure_list.append(p)
        density_list.append(r)
    return x_final, pressure_list, density_list

def update_data():
    # read slider
    omega= omega_slider.value
    x_f, p_f, rho_f= run_solver_with_omega(omega)
    source.data= {'x':x_f, 'pressure':p_f, 'density':rho_f}

def on_slider_change(attr, old, new):
    # Could do immediate re-run, or wait for button
    pass

def on_run_button_clicked():
    update_data()

omega_slider.on_change('value', on_slider_change)
run_button.on_click(on_run_button_clicked)

# do an initial run
update_data()

layout= column(omega_slider, run_button, plot)
curdoc().add_root(layout)
