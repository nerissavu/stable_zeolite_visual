import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import os
from ase import Atoms
from ase.data import chemical_symbols
from ase.units import Hartree, Bohr

from ase.data import chemical_symbols, covalent_radii
from scipy.spatial.distance import pdist, squareform


# Update the extract_data_from_npz function to include atomic_numbers
def extract_data_from_npz(npz_file):
    """Extract energy, stress, and forces from the npz file."""
    data = np.load(npz_file)
    
    # Get the most stable state (min energy)
    stable_state_index = np.argmin(data['energy'])
    numbers = data['numbers']
    positions = data['xyz'][stable_state_index]
    cell = data['lattice'][stable_state_index]
    energy = data['energy'][stable_state_index] * Hartree * 1000
    forces = data['gradients'][stable_state_index] * 51422.1  # [number_of_atoms, 3]
    stress = data['stress'][stable_state_index] * 294210100  # [3, 3]
    charge = data['charges']
    
    # Create ASE Atoms object
    atoms = Atoms(numbers=numbers, positions=positions, cell=cell, pbc=True)
    # Derive zeolite name from file name (or manually)
    zeolite_name = os.path.splitext(os.path.basename(npz_file))[0]
    
    # Store relevant information
    return {
        'Zeolite': zeolite_name,  # Store the zeolite name
        'number_of_atoms': len(atoms),
        'chemical_formula': atoms.get_chemical_formula(),
        'energy (mev)': energy,  # Energy in meV
        'force (meV/Å)': forces.tolist(),  # Convert force matrix
        'stress_tensor (meV/Å³)': stress.tolist(),  # Convert stress tensor 
        'charge': charge.tolist(),  # Convert charge to a list
        'positions': positions.tolist(),  # Add positions for 3D visualization
        'atomic_numbers': numbers.tolist()  # Add atomic numbers for element identification
    }

@st.cache_data
def load_data(npz_dir):
    data_list = []
    for filename in os.listdir(npz_dir):
        if filename.endswith('.npz'):
            npz_file = os.path.join(npz_dir, filename)
            if os.path.exists(npz_file):
                data = extract_data_from_npz(npz_file)
                data_list.append(data)
    return pd.DataFrame(data_list)



def plot_zeolite_structure_with_info(positions, atomic_numbers, forces, charges):
    unique_atoms = sorted(set(atomic_numbers))
    if len(unique_atoms) != 2:
        raise ValueError("Expected exactly two types of atoms in the zeolite structure")
    
    color_map = {unique_atoms[0]: 'blue', unique_atoms[1]: 'red'}
    
    # Calculate distances between all pairs of atoms
    distances = squareform(pdist(positions))
    
    # Determine bonds based on covalent radii
    bonds = []
    for i in range(len(atomic_numbers)):
        for j in range(i+1, len(atomic_numbers)):
            # Sum of covalent radii with 20% tolerance
            bond_length = (covalent_radii[atomic_numbers[i]] + 
                           covalent_radii[atomic_numbers[j]]) * 1.2
            if distances[i, j] <= bond_length:
                bonds.append((i, j))
    
    # Create traces for atoms
    atom_traces = []
    for atom_type in unique_atoms:
        mask = [num == atom_type for num in atomic_numbers]
        atom_positions = positions[mask]
        atom_forces = forces[mask]
        atom_charges = charges[mask]
        atom_symbol = chemical_symbols[atom_type]
        
        hover_text = []
        for (fx, fy, fz), charge in zip(atom_forces, atom_charges):
            text = (
                f"{atom_symbol} ({atom_type})<br>"
                f"Force (meV/Å): ({fx:.2f}, {fy:.2f}, {fz:.2f})<br>"
                f"Charge: {charge:.2f}"
            )
            hover_text.append(text)
        
        trace = go.Scatter3d(
            x=atom_positions[:, 0],
            y=atom_positions[:, 1],
            z=atom_positions[:, 2],
            mode='markers',
            marker=dict(
                size=5,
                color=color_map[atom_type],
                opacity=0.8
            ),
            text=hover_text,
            hoverinfo='text',
            name=f"{atom_symbol} ({atom_type})"
        )
        atom_traces.append(trace)
    
    # Create traces for bonds
    bond_traces = []
    for bond in bonds:
        start, end = positions[bond[0]], positions[bond[1]]
        trace = go.Scatter3d(
            x=[start[0], end[0]],
            y=[start[1], end[1]],
            z=[start[2], end[2]],
            mode='lines',
            line=dict(color='gray', width=2),
            hoverinfo='none',
            showlegend=False
        )
        bond_traces.append(trace)
    
    # Combine all traces
    fig = go.Figure(data=atom_traces + bond_traces)
    
    # Update layout
    fig.update_layout(
        title="Zeolite Structure with Atom Information",
        scene=dict(
            xaxis_title="X",
            yaxis_title="Y",
            zaxis_title="Z",
            aspectmode="data"
        ),
        legend_title="Atom Types",
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01
        )
    )
    
    return fig

# Main app
def main():
    st.title("Zeolite Data Visualization")

    # # Load data
    # npz_dir = st.sidebar.text_input("NPZ Directory", value="npz")
    # if not os.path.exists(npz_dir):
    #     st.error(f"Directory '{npz_dir}' does not exist.")
    #     return

    df_zeolites = load_data("npz")

    # Sidebar
    st.sidebar.header("Filters")
    min_atoms, max_atoms = st.sidebar.slider(
        "Number of atoms", 
        min_value=int(df_zeolites['number_of_atoms'].min()),
        max_value=int(df_zeolites['number_of_atoms'].max()),
        value=(int(df_zeolites['number_of_atoms'].min()), int(df_zeolites['number_of_atoms'].max()))
    )

    # Filter data
    filtered_df = df_zeolites[(df_zeolites['number_of_atoms'] >= min_atoms) & (df_zeolites['number_of_atoms'] <= max_atoms)]

    # Main content
    st.header("Energy Distribution")
    fig = px.scatter(filtered_df, 
                    x='energy (mev)', 
                    y='number_of_atoms',
                    hover_name='Zeolite',
                    hover_data={'energy (mev)': ':.2f', 'number_of_atoms': True},
                    labels={'energy (mev)': 'Energy (meV)', 'number_of_atoms': 'Number of Atoms'},
                    title="Zeolite Energy Distribution")

    # Customize the layout
    fig.update_traces(marker=dict(size=10))
    fig.update_layout(
        xaxis_title="Energy (meV)",
        yaxis_title="Number of Atoms",
        hovermode="closest"
    )
    st.plotly_chart(fig)


    # Detailed information
    st.header("Detailed Zeolite Information")
    selected_zeolite = st.selectbox("Select a Zeolite", filtered_df['Zeolite'])
    zeolite_data = filtered_df[filtered_df['Zeolite'] == selected_zeolite].iloc[0]

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Basic Information")
        st.write(f"Number of atoms: {zeolite_data['number_of_atoms']}")
        st.write(f"Chemical formula: {zeolite_data['chemical_formula']}")
        st.write(f"Energy: {zeolite_data['energy (mev)']:.2f} meV")

    with col2:
        st.subheader("Stress Tensor (meV/Å³)")
        stress_df = pd.DataFrame(zeolite_data['stress_tensor (meV/Å³)'], 
                                 columns=['x', 'y', 'z'])
        st.write(stress_df)

    st.subheader("Zeolite Structure Visualization with Atom Information")
    structure_fig = plot_zeolite_structure_with_info(
        np.array(zeolite_data['positions']),
        np.array(zeolite_data['atomic_numbers']),
        np.array(zeolite_data['force (meV/Å)']),
        np.array(zeolite_data['charge'])
    )
    st.plotly_chart(structure_fig)


if __name__ == "__main__":
    main()