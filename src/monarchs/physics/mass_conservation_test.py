import numpy as np

# Constants
rho_solid = 917  # kg/m^3

# Lid: 20 layers, each 1 m thick, solid fraction = 1
lid_dz = np.ones(20) * 1.0
sfrac_lid = np.ones(20)

# Column: 500 layers, each 2 m thick, varying solid fraction
column_dz = np.ones(500) * 0.015
sfrac_column = np.linspace(0.2, 0.8, 500)

# Combine into full profile
dz_full = np.concatenate((lid_dz, column_dz))  # Length 520
sfrac_full = np.concatenate((sfrac_lid, sfrac_column))  # Length 520

# Depth edges of full profile (top at 0)
z_edges_full = np.concatenate(([0], np.cumsum(dz_full)))  # Length 521
z_centers_full = 0.5 * (z_edges_full[:-1] + z_edges_full[1:])

# Total depth
total_depth = np.sum(dz_full)

# New grid: 500 layers
num_layers_new = 500
dz_new = np.full(num_layers_new, total_depth / num_layers_new)
z_edges_new = np.linspace(0, total_depth, num_layers_new + 1)
z_centers_new = 0.5 * (z_edges_new[:-1] + z_edges_new[1:])

# Solid mass per layer in old grid
mass_old = sfrac_full * dz_full * rho_solid  # length 520

# Create mass function to integrate
mass_profile = np.zeros_like(z_edges_full)
mass_profile[1:] = np.cumsum(mass_old)

# Interpolate cumulative mass to new layer edges
mass_interp_edges = np.interp(z_edges_new, z_edges_full, mass_profile)

# New solid mass per layer
mass_new = np.diff(mass_interp_edges)  # length 500

# Recover new solid fraction
sfrac_new = mass_new / (dz_new * rho_solid)

mass_initial = np.sum(mass_old)
mass_final = np.sum(sfrac_new * dz_new * rho_solid)

print(f"Initial solid mass: {mass_initial:.3f} kg/m²")
print(f"Final solid mass:   {mass_final:.3f} kg/m²")
print(f"Difference:         {mass_final - mass_initial:.3e} kg/m²")
