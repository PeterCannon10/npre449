'''
Peter Cannon - 11/11/26
NPRE 449 Simple CP
'''

import numpy as np
import matplotlib.pyplot as plt
import CoolProp.CoolProp as CP
import os

"""
In this computer project, temperatures and pressures in the hot channel of a PWR and a BWR are calculated.
In both cases, the linear heat generation rate changes in z as: q'(z) = q'_0*sin(pi*z/H), 0 <= z <= H, where H is the fuel height

Assumptions:
Steady-state
No internal heat generation in the fluid
Axial conduction is negligible
Single phase properties are constant
Saturation properties (temperature, liquid saturation enthalpy, vapor saturation enthalpy) are a function of pressure. All other properties can be treated as constant (based on inlet pressure)
Changes in fuel and clad properties with temperature are negligible
"""


reactor_type = "PWR"
if reactor_type == "PWR":
    """
    PWR constants
    """
    #geometry
    h = 4.1 # m
    d_rod = 0.94e-2 # m
    pitch = 1.25e-2 # m
    d_fuel = 0.81e-2 # m
    gap_thickness = 0.005e-2 # m
    k_gap = 0.25 # W/m-C
    k_fuel = 3.6 # W/m-C
    k_clad = 21.5 # W/m-C
    A_f = pitch**2 - 0.25*np.pi*d_rod**2 # m2
    perimeter = np.pi * d_rod # m

    #conditions
    G = 3800 # kg/m2-s
    q0 = 380*100 # W/m
    inlet_pressure = 15e6 # Pa
    inlet_fluid_temperature = 277 # C

    #frequently used properties
    c_p = CP.PropsSI('C', 'T', inlet_fluid_temperature + 273.15, 'P', inlet_pressure, 'Water') # J/kg-K
    g = 9.81 # m/s2
    inlet_fluid_density = CP.PropsSI('D', 'T', inlet_fluid_temperature + 273.15, 'P', inlet_pressure, 'Water') # kg/m3

def fluid_temperature(z):
    """
    Calculates fluid temperature as a function of z -- Complete?
    """
    h_g = CP.PropsSI('H', 'T', inlet_fluid_temperature + 273.15, 'Q', 1, 'Water') # J/kg
    h_f = CP.PropsSI('H', 'T', inlet_fluid_temperature + 273.15, 'Q', 0, 'Water') # J/kg
    h_fg = h_g - h_f
    T_sat = CP.PropsSI('T', 'P', inlet_pressure, 'Q', 0, 'Water') - 273.15
    return (equilibrium_quality(z)*h_fg/c_p + T_sat)

def saturation_temperature(z):
    """
    Calculates saturation temperature as a function of z -- Complete?
    """
    return CP.PropsSI('T', 'P', pressure(z), 'Q', 0, 'Water') - 273.15

def clad_temperature(z):
    """
    Calculates clad temperature as a function of z -- Complete?
    """
    Re = G * d_rod / (CP.PropsSI('V', 'T', inlet_fluid_temperature + 273.15, 'P', inlet_pressure, 'Water'))
    Pr = CP.PropsSI('PRANDTL', 'T', inlet_fluid_temperature + 273.15, 'P', inlet_pressure, 'Water')
    d_h = 4*A_f / perimeter # m
    k_fluid = CP.PropsSI('L', 'T', inlet_fluid_temperature + 273.15, 'P', inlet_pressure, 'Water') # W/m-K
    h_f = k_fluid / d_h * 0.023 * Re**0.8 * Pr**0.4
    return fluid_temperature(z) + q0 * np.sin(np.pi*z/h) / (np.pi * d_rod) / h_f

def centerline_temperature(z):
    """
    Calculates centerline temperature as a function of z -- Complete?
    """
    r_ci = d_fuel/2 + gap_thickness # m
    r_co = d_rod/2 # m
    r_fo = d_fuel/2 # m
    return clad_temperature(z) + q0 * np.sin(np.pi*z/h) * (np.log(r_co/r_ci)/(2*np.pi*k_clad) + np.log(r_ci/r_fo)/(2*np.pi*k_gap) + 1/(4*np.pi*k_fuel))

def pressure(z):
    """
    Calculates pressure as a function of z -- Complete
    """
    Re = G * d_rod / (CP.PropsSI('V', 'T', inlet_fluid_temperature + 273.15, 'P', inlet_pressure, 'Water'))
    if Re > 4000: #Re is much higher than 4000, this is turbulent flow
        f = 0.3164 * Re**(-0.25) # Blasius correlation

    change_in_pressure = f*G**2*perimeter/(8*inlet_fluid_density*A_f)*z + inlet_fluid_density*g*z
    return inlet_pressure - change_in_pressure

def equilibrium_quality(z):
    """
    Calculates equilibrium quality as a function of z -- Complete?
    """
    h_0 = CP.PropsSI('H', 'P', inlet_pressure, 'T', inlet_fluid_temperature, 'Water')
    h_z = h_0 + (q0*h)/(np.pi*G*A_f) * (1-np.cos(np.pi*z/h))
    h_f = CP.PropsSI('H', 'P', pressure(z), 'Q', 0, 'Water')
    h_g = CP.PropsSI('H', 'P', pressure(z), 'Q', 1, 'Water')
    h_fg = h_g - h_f
    return (h_z-h_f)/(h_fg)

def sensitivity():
    G_range = np.linspace(2000, 6000, 100)
    q0_range = np.linspace(100*100, 500*100, 100)
    P0_range = np.linspace(5e6, 25e6, 100)
    T0_range = np.linspace(220, 330, 100)
    d_rod_range = np.linspace(0.75e-2, 1.25e-2, 100)
    pitch_range = np.linspace(1e-2, 1.5e-2, 100)
    h_range = np.linspace(2, 6, 100)

    def _outlet_eq_quality(H, d_rod, pitch, G, q0, P0, T0):
        A_f = pitch**2 - np.pi*d_rod**2 / 4
        P_outlet = P0
        h_0 = CP.PropsSI('H', 'P', P0, 'T', T0+273.15, 'Water')
        h_outlet = h_0 + (2*q0*H)/(np.pi*G*A_f)
        h_f = CP.PropsSI('H', 'P', P_outlet, 'Q', 0, 'Water')
        h_g = CP.PropsSI('H', 'P', P_outlet, 'Q', 1, 'Water')
        h_fg = h_g - h_f

        return (h_outlet - h_f) / (h_g - h_f)

    def _max_centerline_temperature(H, d_rod, pitch, G, q0, P0, T0):
        A_f = pitch**2 - np.pi*d_rod**2 / 4
        perimeter = np.pi*d_rod
        d_h = 4*A_f/perimeter

        T_inlet = T0 + 273.15
        mu = CP.PropsSI('V', 'T', T_inlet, 'P', P0, 'Water')
        Pr = CP.PropsSI('PRANDTL', 'T', T_inlet, 'P', P0, 'Water')
        k_f = CP.PropsSI('L', 'T', T_inlet, 'P', P0, 'Water')

        Re = G*d_h/mu
        h_conv = 0.023*k_f/d_h * Re**0.8 * Pr**0.4

        r_ci = d_fuel/2 + gap_thickness
        r_co = d_rod/2
        r_fo = d_fuel/2

        resistance = np.log(r_co/r_ci)/(2*np.pi*k_clad) + np.log(r_ci/r_fo)/(2*np.pi*k_gap) + 1/(4*np.pi*k_fuel)

        z = np.linspace(0, H, 100)
        h_0 = CP.PropsSI('H', 'P', P0, 'T', T_inlet, 'Water')

        T_center = []
        for i in z:
            h_z = h_0 + (q0*H)/(np.pi*G*A_f)*(1-np.cos(np.pi*i/H))

            T_fluid = CP.PropsSI('T', 'P', P0, 'H', h_z, 'Water')
            q_z = q0*np.sin(np.pi*i/h)
            T_clad = T_fluid + q_z/(np.pi*d_rod*h_conv)

            T_fuel = T_clad + q_z*resistance

            T_center.append(T_fuel)

        return np.max(T_center) - 273.15

    """
    _outlet_eq_quality(h, d_rod, pitch, G, q0, inlet_pressure, inlet_fluid_temperature)
    _max_centerline_temperature(h, d_rod, pitch, G, q0, inlet_pressure, inlet_fluid_temperature)

    plt.savefig(os.path.join('CP Figures', '_graph.png'))
    plt.close()
    plt.savefig(os.path.join('CP Figures', 'temperature_graphs.png'))
    """

    #Varying G
    #Eq Quality
    plt.figure()
    plt.plot(G_range, _outlet_eq_quality(h, d_rod, pitch, G_range, q0, inlet_pressure, inlet_fluid_temperature))
    plt.grid()
    plt.xlabel("G (kg/m$^{2}$-s)")
    plt.ylabel("Outlet Equilibrium Quality")
    plt.axvline(G, ls='--', color='black')
    plt.title("G vs Quality")
    plt.savefig(os.path.join('CP Figures', 'G_x_graph.png'))
    plt.close()

    #Max Centerline Temperature
    cl_temps = []
    for Gi in G_range:
        cl_temps.append(_max_centerline_temperature(h, d_rod, pitch, Gi, q0, inlet_pressure, inlet_fluid_temperature))
    cl_temps = np.array(cl_temps)
    plt.figure()
    plt.plot(G_range, cl_temps)
    plt.grid()
    plt.xlabel("G (kg/m$^{2}$-s)")
    plt.ylabel("Max Centerline Temperature (°C)")
    plt.axvline(G, ls='--', color='black')
    plt.title("G vs Max Centerline Temperature")
    plt.savefig(os.path.join('CP Figures', 'G_T_graph.png'))
    plt.close()

    #Varying q0
    #Eq Quality
    plt.figure()
    plt.plot(q0_range, _outlet_eq_quality(h, d_rod, pitch, G, q0_range, inlet_pressure, inlet_fluid_temperature))
    plt.grid()
    plt.xlabel("q'0 (W/m)")
    plt.ylabel("Outlet Equilibrium Quality")
    plt.axvline(q0, ls='--', color='black')
    plt.title("q'0 vs Quality")
    plt.savefig(os.path.join('CP Figures', 'q0_x_graph.png'))
    plt.close()

    #Max Centerline Temperature
    cl_temps = []
    for q0i in q0_range:
        cl_temps.append(_max_centerline_temperature(h, d_rod, pitch, G, q0i, inlet_pressure, inlet_fluid_temperature))
    cl_temps = np.array(cl_temps)
    plt.figure()
    plt.plot(q0_range, cl_temps)
    plt.grid()
    plt.xlabel("q'0 (W/m)")
    plt.ylabel("Max Centerline Temperature (°C)")
    plt.axvline(q0, ls='--', color='black')
    plt.title("q'0 vs Max Centerline Temperature")
    plt.savefig(os.path.join('CP Figures', 'q0_T_graph.png'))
    plt.close()

    #Varying P0
    #Eq Quality
    plt.figure()
    plt.plot(P0_range, _outlet_eq_quality(h, d_rod, pitch, G, q0, P0_range, inlet_fluid_temperature))
    plt.grid()
    plt.xlabel("P0 (Pa)")
    plt.ylabel("Outlet Equilibrium Quality")
    plt.axvline(inlet_pressure, ls='--', color='black')
    plt.title("P0 vs Quality")
    plt.savefig(os.path.join('CP Figures', 'P0_x_graph.png'))
    plt.close()

    #Max Centerline Temperature
    cl_temps = []
    for P0i in P0_range:
        cl_temps.append(_max_centerline_temperature(h, d_rod, pitch, G, q0, P0i, inlet_fluid_temperature))
    cl_temps = np.array(cl_temps)
    plt.figure()
    plt.plot(P0_range, cl_temps)
    plt.grid()
    plt.xlabel("P0 (Pa)")
    plt.ylabel("Max Centerline Temperature (°C)")
    plt.axvline(inlet_pressure, ls='--', color='black')
    plt.title("P0 vs Max Centerline Temperature")
    plt.savefig(os.path.join('CP Figures', 'P0_T_graph.png'))
    plt.close()

    #Varying T0
    #Eq Quality
    plt.figure()
    plt.plot(T0_range, _outlet_eq_quality(h, d_rod, pitch, G, q0, inlet_pressure, T0_range))
    plt.grid()
    plt.xlabel("T0 (°C)")
    plt.ylabel("Outlet Equilibrium Quality")
    plt.axvline(inlet_fluid_temperature, ls='--', color='black')
    plt.title("T0 vs Quality")
    plt.savefig(os.path.join('CP Figures', 'T0_x_graph.png'))
    plt.close()

    #Max Centerline Temperature
    cl_temps = []
    for T0i in T0_range:
        cl_temps.append(_max_centerline_temperature(h, d_rod, pitch, G, q0, inlet_pressure, T0i))
    cl_temps = np.array(cl_temps)
    plt.figure()
    plt.plot(T0_range, cl_temps)
    plt.grid()
    plt.xlabel("T0 (°C)")
    plt.ylabel("Max Centerline Temperature (°C)")
    plt.axvline(inlet_fluid_temperature, ls='--', color='black')
    plt.title("T0 vs Max Centerline Temperature")
    plt.savefig(os.path.join('CP Figures', 'T0_T_graph.png'))
    plt.close()

    #Varying d_rod
    #Eq Quality
    plt.figure()
    plt.plot(d_rod_range, _outlet_eq_quality(h, d_rod_range, pitch, G, q0, inlet_pressure, inlet_fluid_temperature))
    plt.grid()
    plt.xlabel("Rod Diameter (m)")
    plt.ylabel("Outlet Equilibrium Quality")
    plt.axvline(d_rod, ls='--', color='black')
    plt.title("Rod Diameter vs Quality")
    plt.savefig(os.path.join('CP Figures', 'd_x_graph.png'))
    plt.close()

    #Max Centerline Temperature
    cl_temps = []
    for d_rodi in d_rod_range:
        cl_temps.append(_max_centerline_temperature(h, d_rodi, pitch, G, q0, inlet_pressure, inlet_fluid_temperature))
    cl_temps = np.array(cl_temps)
    plt.figure()
    plt.plot(d_rod_range, cl_temps)
    plt.grid()
    plt.xlabel("Rod Diameter (m)")
    plt.ylabel("Max Centerline Temperature (°C)")
    plt.axvline(d_rod, ls='--', color='black')
    plt.title("Rod Diameter vs Max Centerline Temperature")
    plt.savefig(os.path.join('CP Figures', 'd_T_graph.png'))
    plt.close()

    #Varying pitch
    #Eq Quality
    plt.figure()
    plt.plot(pitch_range, _outlet_eq_quality(h, d_rod, pitch_range, G, q0, inlet_pressure, inlet_fluid_temperature))
    plt.grid()
    plt.xlabel("Pitch (m)")
    plt.ylabel("Outlet Equilibrium Quality")
    plt.axvline(pitch, ls='--', color='black')
    plt.title("Pitch vs Quality")
    plt.savefig(os.path.join('CP Figures', 'pitch_x_graph.png'))
    plt.close()

    #Max Centerline Temperature
    cl_temps = []
    for pitchi in pitch_range:
        cl_temps.append(_max_centerline_temperature(h, d_rod, pitchi, G, q0, inlet_pressure, inlet_fluid_temperature))
    cl_temps = np.array(cl_temps)
    plt.figure()
    plt.plot(pitch_range, cl_temps)
    plt.grid()
    plt.xlabel("Pitch (m)")
    plt.ylabel("Max Centerline Temperature (°C)")
    plt.axvline(pitch, ls='--', color='black')
    plt.title("Pitch vs Max Centerline Temperature")
    plt.savefig(os.path.join('CP Figures', 'pitch_T_graph.png'))
    plt.close()

    #Varying h
    #Eq Quality
    plt.figure()
    plt.plot(h_range, _outlet_eq_quality(h_range, d_rod, pitch, G, q0, inlet_pressure, inlet_fluid_temperature))
    plt.grid()
    plt.xlabel("Fuel Height (m)")
    plt.ylabel("Outlet Equilibrium Quality")
    plt.axvline(h, ls='--', color='black')
    plt.title("Fuel Height vs Quality")
    plt.savefig(os.path.join('CP Figures', 'h_x_graph.png'))
    plt.close()

    #Max Centerline Temperature
    cl_temps = []
    for hi in h_range:
        cl_temps.append(_max_centerline_temperature(hi, d_rod, pitch, G, q0, inlet_pressure, inlet_fluid_temperature))
    cl_temps = np.array(cl_temps)
    plt.figure()
    plt.plot(h_range, cl_temps)
    plt.grid()
    plt.xlabel("Fuel Height (m)")
    plt.ylabel("Max Centerline Temperature (°C)")
    plt.axvline(h, ls='--', color='black')
    plt.title("Fuel Height vs Max Centerline Temperature")
    plt.savefig(os.path.join('CP Figures', 'h_T_graph.png'))
    plt.close()


def plotting():
    """
    Plots Question 1 plots
    """
    colors = ['#261cba', '#308a0c', '#ad3d0c', '#ad0c9d', '#000000']
    z = np.linspace(0, h, 100)

    #temperature plot
    plt.figure()
    plt.plot(fluid_temperature(z), z, label='Fluid Temperature', color=colors[0])
    plt.plot(saturation_temperature(z), z, label='Saturation Temperature', color=colors[1])
    plt.plot(clad_temperature(z), z, label='Clad Temperature', color=colors[2])
    plt.grid()
    plt.ylabel("z (m)")
    plt.xlabel("Temperature (°C)")
    plt.title("Temperature vs. z")
    plt.ylim(0, h)
    plt.legend()
    plt.savefig(os.path.join('CP Figures', 'temperature_graphs.png'))
    plt.close()

    #centerline temperature plot
    plt.figure()
    plt.plot(centerline_temperature(z), z, label='Centerline Temperature', color=colors[3])
    plt.grid()
    plt.ylabel("z (m)")
    plt.xlabel("Temperature (°C)")
    plt.title("Centerline Temperature vs. z")
    plt.ylim(0, h)
    plt.savefig(os.path.join('CP Figures', 'centerline_temperature_graph.png'))
    plt.close()

    #equilibrium quality plot
    plt.figure()
    plt.plot(equilibrium_quality(z), z, color='blue')
    plt.grid()
    plt.ylabel("z (m)")
    plt.xlabel("Equilibrium Quality")
    plt.title("Equilibrium quality vs. z")
    plt.ylim(0, h)
    plt.savefig(os.path.join('CP Figures', 'equilibrium_quality_graph.png'))
    plt.close()

    #pressure plot
    plt.figure()
    plt.plot(pressure(z) /1e6, z, color='red')
    plt.grid()
    plt.ylabel("z (m)")
    plt.xlabel("Pressure (MPa)")
    plt.title("Pressure vs. z")
    plt.ylim(0, h)
    plt.savefig(os.path.join('CP Figures', 'pressure_graph.png'))
    plt.close()

plotting()
sensitivity()