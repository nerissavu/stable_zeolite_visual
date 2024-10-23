import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import os
from ase import Atoms
from ase.data import chemical_symbols, covalent_radii
from ase.units import Hartree, Bohr
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

def analyze_energy_per_atom(df_zeolites, energy_col='energy (mev)', atoms_col='number_of_atoms', name_col='Zeolite'):
    """
    Create comprehensive energy per atom analysis and plots
    """
    df_zeolites = df_zeolites.copy()
    df_zeolites['energy_per_atom'] = df_zeolites[energy_col] / df_zeolites[atoms_col]
    
    scatter_fig = px.scatter(df_zeolites,
                           x='energy_per_atom',
                           y=atoms_col,
                           hover_name=name_col,
                           hover_data={
                               'energy_per_atom': ':.4f',
                               energy_col: ':.2f',
                               atoms_col: True
                           },
                           labels={
                               'energy_per_atom': 'Energy per Atom (meV/atom)',
                               atoms_col: 'Number of Atoms'
                           },
                           title="Zeolite Energy per Atom Distribution")
    
    scatter_fig.update_layout(
        hovermode='closest',
        xaxis_title="Energy per Atom (meV/atom)",
        yaxis_title="Number of Atoms"
    )
    
    multi_fig = make_subplots(rows=2, cols=2,
                             subplot_titles=("Scatter Plot", "Histogram",
                                           "Box Plot", "Violin Plot"))
    
    multi_fig.add_trace(
        go.Scatter(x=df_zeolites['energy_per_atom'], y=df_zeolites[atoms_col],
                  mode='markers', name='Zeolites',
                  hovertemplate="Energy/Atom: %{x:.4f}<br>" +
                               "Atoms: %{y}<br>" +
                               "<extra></extra>"),
        row=1, col=1
    )
    
    multi_fig.add_trace(
        go.Histogram(x=df_zeolites['energy_per_atom'], name='Distribution'),
        row=1, col=2
    )
    
    multi_fig.add_trace(
        go.Box(y=df_zeolites['energy_per_atom'], name='Energy/Atom',
               boxpoints='all', jitter=0.3, pointpos=-1.8),
        row=2, col=1
    )
    
    multi_fig.add_trace(
        go.Violin(y=df_zeolites['energy_per_atom'], name='Energy/Atom',
                 box_visible=True, meanline_visible=True),
        row=2, col=2
    )
    
    multi_fig.update_layout(
        height=800,
        title_text="Energy per Atom Analysis",
        showlegend=False
    )
    
    stats = {
        'mean': df_zeolites['energy_per_atom'].mean(),
        'median': df_zeolites['energy_per_atom'].median(),
        'std': df_zeolites['energy_per_atom'].std(),
        'min': df_zeolites['energy_per_atom'].min(),
        'max': df_zeolites['energy_per_atom'].max(),
        'n_samples': len(df_zeolites)
    }
    
    return {
        'scatter_plot': scatter_fig,
        'multi_plot': multi_fig,
        'stats': stats,
        'data': df_zeolites
    }

def analyze_forces(df):
    """
    Analyze forces from DataFrame where forces are stored as lists in 'force (meV/Å)' column
    
    Parameters:
    df: pandas DataFrame containing force data
    
    Returns:
    dict containing processed data and figures
    """
    # Initialize lists to store processed data
    all_flattened_forces = []
    all_force_magnitudes = []
    zeolite_names = []
    
    # Process each zeolite's forces
    for idx, row in df.iterrows():
        forces = np.array(row['force (meV/Å)'])
        zeolite_name = row['Zeolite']
        num_atoms = row['number_of_atoms']
        
        # Calculate force magnitudes for each atom
        magnitudes = np.sqrt(np.sum(forces**2, axis=1))
        
        # Store data
        all_flattened_forces.extend(forces.flatten())
        all_force_magnitudes.extend(magnitudes)
        zeolite_names.extend([zeolite_name] * len(magnitudes))
    
    # Create figures
    # 1. Distribution of all force components
    flat_fig = go.Figure()
    flat_fig.add_trace(go.Histogram(
        x=all_flattened_forces,
        name='Force Components',
        nbinsx=200,
        histnorm='probability'
    ))
    flat_fig.update_layout(
        title='Distribution of Force Components',
        xaxis_title='Force (meV/Å)',
        yaxis_title='Probability',
        showlegend=False
    )
    
    # 2. Distribution of force magnitudes
    mag_fig = go.Figure()
    mag_fig.add_trace(go.Histogram(
        x=all_force_magnitudes,
        name='Force Magnitudes',
        nbinsx=50,
        histnorm='probability'
    ))
    mag_fig.update_layout(
        title='Distribution of Force Magnitudes',
        xaxis_title='Force Magnitude (meV/Å)',
        yaxis_title='Probability',
        showlegend=False
    )
    
    # 3. Create comprehensive analysis figure
    multi_fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Force Components Distribution',
                       'Force Magnitudes Distribution',
                       'Force Components Box Plot',
                       'Force Magnitudes Box Plot')
    )
    
    # Add traces
    multi_fig.add_trace(
        go.Histogram(x=all_flattened_forces, name='Components'),
        row=1, col=1
    )
    multi_fig.add_trace(
        go.Histogram(x=all_force_magnitudes, name='Magnitudes'),
        row=1, col=2
    )
    multi_fig.add_trace(
        go.Box(y=all_flattened_forces, name='Components'),
        row=2, col=1
    )
    multi_fig.add_trace(
        go.Box(y=all_force_magnitudes, name='Magnitudes'),
        row=2, col=2
    )
    
    multi_fig.update_layout(
        height=800,
        title_text='Comprehensive Force Analysis',
        showlegend=False
    )
    
    # Calculate statistics
    stats = {
        'force_components': {
            'mean': np.mean(all_flattened_forces),
            'median': np.median(all_flattened_forces),
            'std': np.std(all_flattened_forces),
            'min': np.min(all_flattened_forces),
            'max': np.max(all_flattened_forces)
        },
        'force_magnitudes': {
            'mean': np.mean(all_force_magnitudes),
            'median': np.median(all_force_magnitudes),
            'std': np.std(all_force_magnitudes),
            'min': np.min(all_force_magnitudes),
            'max': np.max(all_force_magnitudes)
        }
    }
    
    return {
        'flattened_plot': flat_fig,
        'magnitude_plot': mag_fig,
        'multi_plot': multi_fig,
        'stats': stats,
        'flattened_forces': all_flattened_forces,
        'force_magnitudes': all_force_magnitudes
    }

def analyze_force_components(df):
    """
    Analyze x, y, z components of forces separately
    """
    fx_all = []
    fy_all = []
    fz_all = []
    
    for _, row in df.iterrows():
        forces = np.array(row['force (meV/Å)'])
        fx = forces[:, 0]
        fy = forces[:, 1]
        fz = forces[:, 2]
        
        fx_all.extend(fx)
        fy_all.extend(fy)
        fz_all.extend(fz)
    
    fig = make_subplots(rows=2, cols=1,
                       subplot_titles=('Force Components Distributions',
                                     'Force Components Box Plots'))
    
    # Add histograms
    fig.add_trace(
        go.Histogram(x=fx_all, name='Fx', nbinsx=50, 
                    histnorm='probability', opacity=0.6),
        row=1, col=1
    )
    fig.add_trace(
        go.Histogram(x=fy_all, name='Fy', nbinsx=50,
                    histnorm='probability', opacity=0.6),
        row=1, col=1
    )
    fig.add_trace(
        go.Histogram(x=fz_all, name='Fz', nbinsx=50,
                    histnorm='probability', opacity=0.6),
        row=1, col=1
    )
    
    # Add box plots
    fig.add_trace(
        go.Box(y=fx_all, name='Fx', boxpoints='outliers'),
        row=2, col=1
    )
    fig.add_trace(
        go.Box(y=fy_all, name='Fy', boxpoints='outliers'),
        row=2, col=1
    )
    fig.add_trace(
        go.Box(y=fz_all, name='Fz', boxpoints='outliers'),
        row=2, col=1
    )
    
    fig.update_layout(
        height=800,
        title_text='Analysis of Force Components by Direction',
        barmode='overlay',
        showlegend=True,
        legend=dict(
            x=1.0,
            y=0.5,
            yanchor='middle',
            xanchor='left'
        )
    )
    
    # Calculate statistics
    stats = {
        'Fx': {
            'mean': np.mean(fx_all),
            'median': np.median(fx_all),
            'std': np.std(fx_all),
            'min': np.min(fx_all),
            'max': np.max(fx_all)
        },
        'Fy': {
            'mean': np.mean(fy_all),
            'median': np.median(fy_all),
            'std': np.std(fy_all),
            'min': np.min(fy_all),
            'max': np.max(fy_all)
        },
        'Fz': {
            'mean': np.mean(fz_all),
            'median': np.median(fz_all),
            'std': np.std(fz_all),
            'min': np.min(fz_all),
            'max': np.max(fz_all)
        }
    }
    
    return fig, stats

def analyze_stress_tensor(df):
    """
    Analyze stress tensors from zeolite data
    
    Parameters:
    df: DataFrame containing 'stress_tensor (meV/Å³)' column
    
    Returns:
    dict containing figures and analysis results
    """
    # Lists to store analyzed properties
    principal_stresses = []
    von_mises_stresses = []
    hydrostatic_pressures = []
    shear_stresses = []
    zeolite_names = []
    
    for _, row in df.iterrows():
        # Get stress tensor
        stress = np.array(row['stress_tensor (meV/Å³)'])
        zeolite_name = row['Zeolite']
        
        # Calculate principal stresses (eigenvalues)
        eigenvals = np.linalg.eigvals(stress)
        principal_stresses.append(eigenvals)
        
        # Calculate von Mises stress
        # σvm = √[(σ1-σ2)² + (σ2-σ3)² + (σ3-σ1)²]/√2
        s1, s2, s3 = np.sort(eigenvals)
        von_mises = np.sqrt(((s1-s2)**2 + (s2-s3)**2 + (s3-s1)**2)/2)
        von_mises_stresses.append(von_mises)
        
        # Calculate hydrostatic pressure (mean stress)
        # P = -(σxx + σyy + σzz)/3
        pressure = -np.trace(stress)/3
        hydrostatic_pressures.append(pressure)
        
        # Calculate maximum shear stress
        # τmax = (σ1 - σ3)/2
        max_shear = (s1 - s3)/2
        shear_stresses.append(max_shear)
        
        zeolite_names.append(zeolite_name)
    
    # Convert to numpy arrays
    principal_stresses = np.array(principal_stresses)
    
    # Create visualization
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Principal Stresses Distribution',
                       'Von Mises Stress vs Hydrostatic Pressure',
                       'Maximum Shear Stress Distribution',
                       'Stress State Analysis')
    )
    
    # 1. Principal stresses distribution
    for i in range(3):
        fig.add_trace(
            go.Histogram(x=principal_stresses[:,i],
                        name=f'σ{i+1}',
                        nbinsx=30,
                        opacity=0.6),
            row=1, col=1
        )
    
    # 2. Von Mises vs Hydrostatic pressure
    fig.add_trace(
        go.Scatter(x=hydrostatic_pressures,
                  y=von_mises_stresses,
                  mode='markers',
                  text=zeolite_names,
                  name='Stress State'),
        row=1, col=2
    )
    
    # 3. Maximum shear stress distribution
    fig.add_trace(
        go.Histogram(x=shear_stresses,
                    name='Max Shear',
                    nbinsx=30),
        row=2, col=1
    )
    
    # 4. Box plots for all stress measures
    fig.add_trace(
        go.Box(y=von_mises_stresses,
               name='Von Mises',
               boxpoints='outliers'),
        row=2, col=2
    )
    fig.add_trace(
        go.Box(y=hydrostatic_pressures,
               name='Hydrostatic',
               boxpoints='outliers'),
        row=2, col=2
    )
    fig.add_trace(
        go.Box(y=shear_stresses,
               name='Max Shear',
               boxpoints='outliers'),
        row=2, col=2
    )
    
    # Update layout
    fig.update_layout(
        height=800,
        title_text='Stress Tensor Analysis',
        showlegend=True
    )
    
    # Calculate stability metrics
    stability_metrics = {
        'principal_stresses': {
            'mean': np.mean(principal_stresses, axis=0),
            'std': np.std(principal_stresses, axis=0)
        },
        'von_mises': {
            'mean': np.mean(von_mises_stresses),
            'std': np.std(von_mises_stresses)
        },
        'hydrostatic_pressure': {
            'mean': np.mean(hydrostatic_pressures),
            'std': np.std(hydrostatic_pressures)
        },
        'max_shear': {
            'mean': np.mean(shear_stresses),
            'std': np.std(shear_stresses)
        }
    }
    
    # Create a summary DataFrame
    summary_df = pd.DataFrame({
        'Zeolite': zeolite_names,
        'Von_Mises_Stress': von_mises_stresses,
        'Hydrostatic_Pressure': hydrostatic_pressures,
        'Max_Shear_Stress': shear_stresses,
        'Principal_Stress_1': principal_stresses[:,0],
        'Principal_Stress_2': principal_stresses[:,1],
        'Principal_Stress_3': principal_stresses[:,2]
    })
    
    return {
        'figure': fig,
        'metrics': stability_metrics,
        'summary': summary_df
    }

def main():
    st.title("Zeolite Data Visualization")

    # Load data
    df_zeolites = load_data("npz")

    # Sidebar
    st.sidebar.header("Filters")
    
    # Number of atoms filter
    min_atoms, max_atoms = st.sidebar.slider(
        "Number of atoms", 
        min_value=int(df_zeolites['number_of_atoms'].min()),
        max_value=int(df_zeolites['number_of_atoms'].max()),
        value=(int(df_zeolites['number_of_atoms'].min()), int(df_zeolites['number_of_atoms'].max()))
    )

    # Add navigation to sidebar
    st.sidebar.markdown("---")  # Separator
    st.sidebar.header("Navigation")
    page = st.sidebar.radio("", ["Introduction", "Energy Analysis", "Force Analysis","Stress Analysis", "Detailed Information"])

    # Filter data
    filtered_df = df_zeolites[
        (df_zeolites['number_of_atoms'] >= min_atoms) & 
        (df_zeolites['number_of_atoms'] <= max_atoms)
    ]
    if page == "Introduction":
        st.header("Welcome to the Zeolite Data Visualization Tool")

        # Dataset Information
        st.subheader("About the Dataset")
        st.markdown("""
        This application visualizes data from the Zeo-1 dataset, a comprehensive collection of zeolite structures and their properties. 
        
        📚 **Dataset Citation**:
        ```
        Komissarov, L., Verstraelen, T. Zeo-1, a computational data set of zeolite structures. 
        Sci Data 9, 91 (2022). https://doi.org/10.1038/s41597-022-01160-5
        ```
        """)

        # Data Processing
        st.subheader("Data Processing")
        st.markdown("""
        The application processes raw zeolite data to:
        * Extract properties from the most stable frame of each zeolite structure
        * Calculate and organize key properties including:
            * Energy values
            * Force vectors
            * Stress tensors
            * Atomic charges
        """)

        # Features and Capabilities
        st.subheader("Features")
        
        st.markdown("""
        **1. Energy Analysis**
        * Visualize energy distribution across different structures
        * Examine relationship between energy and number of atoms
        * Analyze energy per atom statistics
        
        **2. Force Analysis**
        * View force component distributions
        * Analyze force magnitudes
        * Examine force vectors for individual atoms
        
        **3. Stress Analysis**
        * Analyze stress tensor components
        * View stress distributions and relationships
        * Examine structural stability metrics
        
        **4. Detailed Information**  
        For any selected zeolite structure, you can:
        * View 3D molecular structure visualization
        * Examine structural properties:
            * Chemical formula
            * Number of atoms
            * Total energy
            * Stress tensor
        * Interactive atom inspection:
            * Hover over atoms to see:
                * Atom type
                * Atomic charges
                * Force vectors
        """)

        # Navigation Guide
        st.subheader("Getting Started")
        st.markdown("""
        1. Use the sidebar to navigate between different analysis pages
        2. Adjust the "Number of atoms" filter to focus on specific structure sizes
        3. Select individual zeolites for detailed inspection
        4. Interact with visualizations to explore the data
        """)
    # Display selected page
    elif page == "Energy Analysis":
        st.header("Energy Analysis")
        st.markdown("""
        ### Understanding Energy Analysis in Zeolites
        
        Energy calculations provide fundamental insights into zeolite stability and properties:
        
        1. **Total Energy**:
           - Represents the total electronic energy of the zeolite structure
           - Lower energy generally indicates greater stability
           - Units: meV (millielectron volts)
        
        2. **Energy per Atom**:
           - Total energy normalized by number of atoms
           - Allows fair comparison between structures of different sizes
           - Calculated as: E_per_atom = E_total / N_atoms
           - Units: meV/atom
        
        Key Analysis Components:
        
        - **Distribution Analysis**:
           - Shows the spread of energies across different structures
           - Helps identify outliers and typical energy ranges
           - Can indicate relative stability of different frameworks
        
        - **Statistical Measures**:
           - Mean: Average energy indicating typical stability
           - Standard Deviation: Spread in energy values
           - Min/Max: Range of structural energies
           - Number of Samples: Total structures analyzed
        
        Applications:
        - Structure stability assessment
        - Framework comparison
        - Quality check of calculations
        - Identification of unusual structures
        """)
        # Total Energy Distribution
        st.subheader("Total Energy Distribution")
        energy_dist_fig = px.scatter(filtered_df, 
                        x='energy (mev)', 
                        y='number_of_atoms',
                        hover_name='Zeolite',
                        hover_data={'energy (mev)': ':.2f', 'number_of_atoms': True},
                        labels={'energy (mev)': 'Energy (meV)', 'number_of_atoms': 'Number of Atoms'})
        
        energy_dist_fig.update_traces(marker=dict(size=10))
        energy_dist_fig.update_layout(
            xaxis_title="Energy (meV)",
            yaxis_title="Number of Atoms",
            hovermode="closest"
        )
        st.plotly_chart(energy_dist_fig)
        
        # Energy per Atom Analysis
        st.subheader("Energy per Atom Analysis")
        
        # Get energy analysis results
        energy_analysis = analyze_energy_per_atom(filtered_df)
        
        # Custom CSS for smaller font
        st.markdown("""
            <style>
            [data-testid="stMetricValue"] {
                font-size: 16px;
            }
            [data-testid="stMetricLabel"] {
                font-size: 14px;
            }
            </style>
            """, unsafe_allow_html=True)
            
        st.subheader("Statistical Summary")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric(
                "Mean Energy per Atom",
                f"{energy_analysis['stats']['mean']:.2f} meV/atom"
            )
            st.metric(
                "Standard Deviation",
                f"{energy_analysis['stats']['std']:.2f} meV/atom"
            )
        with col2:
            st.metric(
                "Median Energy per Atom",
                f"{energy_analysis['stats']['median']:.2f} meV/atom"
            )
            st.metric(
                "Number of Samples",
                f"{energy_analysis['stats']['n_samples']}"
            )
        with col3:
            st.metric(
                "Min Energy per Atom",
                f"{energy_analysis['stats']['min']:.2f} meV/atom"
            )
            st.metric(
                "Max Energy per Atom",
                f"{energy_analysis['stats']['max']:.2f} meV/atom"
            )
        
        st.subheader("Distribution Plots")
        st.plotly_chart(energy_analysis['multi_plot'], use_container_width=True)
    elif page == "Force Analysis":
        st.header("Force Analysis")
        
                # Add explanation section
        st.markdown("""
        ### Understanding Zeolite Forces
        
        Forces in zeolite structures represent the atomic forces acting on each atom in the crystal structure. These forces are important indicators of:
        - Structural stability
        - Energy minimization
        - Quality of crystal structure optimization
        
        #### Dimensions and Units
        - Forces are measured in **meV/Å** (millielectron volts per Angstrom)
        - Each atom has force components in three dimensions (x, y, z)
        - A perfectly optimized structure should have forces close to zero
        
        #### Calculations
        1. **Force Components**:
           - Individual x, y, z components of forces on each atom
           - Can be positive (repulsive) or negative (attractive)
           - Ideal value: 0 meV/Å
        
        2. **Force Magnitude**:
           - Calculated as: $\\sqrt{F_x^2 + F_y^2 + F_z^2}$
           - Always positive
           - Represents the total force intensity on each atom
           - Lower values indicate better structural optimization
        
        The following analysis shows the distribution of both force components and magnitudes across all atoms in the selected zeolite structures.
        """)
        # Custom CSS for smaller font sizes
        st.markdown("""
            <style>
            [data-testid="stMetricValue"] {
                font-size: 14px !important;
            }
            [data-testid="stMetricLabel"] {
                font-size: 12px !important;
            }
            [data-testid="stMarkdownContainer"] > div:first-child {
                font-size: 13px !important;
            }
            </style>
        """, unsafe_allow_html=True)
        
# Get force analysis results
        force_results = analyze_forces(filtered_df)
        component_fig, component_stats = analyze_force_components(filtered_df)
        
        # Display statistics in three columns
        st.subheader("Statistical Summary")
        
        # Custom CSS for smaller font sizes
        st.markdown("""
            <style>
            [data-testid="stMetricValue"] {
                font-size: 14px !important;
            }
            [data-testid="stMetricLabel"] {
                font-size: 12px !important;
            }
            [data-testid="stMarkdownContainer"] > div:first-child {
                font-size: 13px !important;
            }
            </style>
        """, unsafe_allow_html=True)
        
        # Three columns for statistics
        col1, col2, col3 = st.columns(3)
        
        # Force Components (x, y, z)
        with col1:
            st.markdown("**Force Components Statistics**")
            directions = ['Fx', 'Fy', 'Fz']
            for direction in directions:
                stats = component_stats[direction]
                st.metric(f"{direction} Mean", f"{stats['mean']:.2f} meV/Å")
                st.metric(f"{direction} Std Dev", f"{stats['std']:.2f} meV/Å")
        
        # Overall Force Components
        with col2:
            st.markdown("**Overall Force Components**")
            stats = force_results['stats']['force_components']
            st.metric("Mean", f"{stats['mean']:.2f} meV/Å")
            st.metric("Median", f"{stats['median']:.2f} meV/Å")
            st.metric("Std Dev", f"{stats['std']:.2f} meV/Å")
            st.metric("Min", f"{stats['min']:.2f} meV/Å")
            st.metric("Max", f"{stats['max']:.2f} meV/Å")
        
        # Force Magnitudes
        with col3:
            st.markdown("**Force Magnitudes**")
            stats = force_results['stats']['force_magnitudes']
            st.metric("Mean", f"{stats['mean']:.2f} meV/Å")
            st.metric("Median", f"{stats['median']:.2f} meV/Å")
            st.metric("Std Dev", f"{stats['std']:.2f} meV/Å")
            st.metric("Min", f"{stats['min']:.2f} meV/Å")
            st.metric("Max", f"{stats['max']:.2f} meV/Å")

        # Display force component analysis
        st.subheader("Force Components Analysis")
        st.plotly_chart(component_fig, use_container_width=True)
        
        # Display overall force analysis
        st.subheader("Overall Force Analysis")
        st.plotly_chart(force_results['multi_plot'], use_container_width=True)
    elif page == "Stress Analysis":
        st.header("Stress Analysis")
        
       # Enhanced explanation of stress analysis
        st.markdown("""
        ### Understanding Stress Tensor Analysis
        
        The stress tensor in zeolites is a 3×3 matrix representing the internal forces acting on the crystal structure:
        
        ```
        σ = [σxx  σxy  σxz]
            [σyx  σyy  σyz]
            [σzx  σzy  σzz]
        ```
        
        where:
        - Diagonal elements (σxx, σyy, σzz): Normal stresses acting perpendicular to faces
        - Off-diagonal elements: Shear stresses acting parallel to faces
        
        Key stress measures analyzed:
        
        1. **Principal Stresses (σ₁, σ₂, σ₃)**:
           - Eigenvalues of the stress tensor
           - Represent normal stresses in the principal directions
           - Ordered such that σ₁ ≥ σ₂ ≥ σ₃
        
        2. **Von Mises Stress**:
           - Single value representing the overall stress state
           - Calculated as: σᵥₘ = √[(σ₁-σ₂)² + (σ₂-σ₃)² + (σ₃-σ₁)²]/√2
           - Important for assessing structural stability
        
        3. **Hydrostatic Pressure**:
           - Average of normal stresses: P = -(σxx + σyy + σzz)/3
           - Represents the uniform pressure component
           - Negative values indicate compression
        
        4. **Maximum Shear Stress**:
           - Calculated as: τₘₐₓ = (σ₁ - σ₃)/2
           - Indicates the maximum shearing effect in the structure
        
        Units:
        - All stress measures are in meV/Å³ (millielectron volts per cubic Angstrom)
        - This unit is common in quantum mechanical calculations
        - Can be converted to GPa (gigapascals) for comparison with experimental values
        
        The analysis helps evaluate:
        - Structural stability
        - Mechanical properties
        - Deformation behavior
        - Potential phase transitions
        """)
        
        # Get stress analysis results
        stress_results = analyze_stress_tensor(filtered_df)
        
        # Display stress metrics
        st.subheader("Statistical Summary")
        
        # Custom CSS for smaller font sizes
        st.markdown("""
            <style>
            [data-testid="stMetricValue"] {
                font-size: 14px !important;
            }
            [data-testid="stMetricLabel"] {
                font-size: 12px !important;
            }
            </style>
        """, unsafe_allow_html=True)
        
        # Create three columns for metrics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("**Principal Stresses**")
            for i, (mean, std) in enumerate(zip(
                stress_results['metrics']['principal_stresses']['mean'],
                stress_results['metrics']['principal_stresses']['std']
            )):
                st.metric(f"σ₍{i+1}₎ Mean", f"{mean:.2f} meV/Å³")
                st.metric(f"σ₍{i+1}₎ Std", f"{std:.2f} meV/Å³")

        with col2:
            st.markdown("**Von Mises & Hydrostatic**")
            vm_stats = stress_results['metrics']['von_mises']
            hp_stats = stress_results['metrics']['hydrostatic_pressure']
            st.metric("Von Mises Mean", f"{vm_stats['mean']:.2f} meV/Å³")
            st.metric("Von Mises Std", f"{vm_stats['std']:.2f} meV/Å³")
            st.metric("Hydrostatic Mean", f"{hp_stats['mean']:.2f} meV/Å³")
            st.metric("Hydrostatic Std", f"{hp_stats['std']:.2f} meV/Å³")

        with col3:
            st.markdown("**Maximum Shear**")
            shear_stats = stress_results['metrics']['max_shear']
            st.metric("Max Shear Mean", f"{shear_stats['mean']:.2f} meV/Å³")
            st.metric("Max Shear Std", f"{shear_stats['std']:.2f} meV/Å³")

        # Display main visualization
        st.subheader("Stress Analysis Plots")
        st.plotly_chart(stress_results['figure'], use_container_width=True)
        
        # Display top 5 zeolites by Von Mises stress
        st.subheader("Top 5 Zeolites by Von Mises Stress")
        top_5_df = stress_results['summary'].nlargest(5, 'Von_Mises_Stress')[['Zeolite', 'Von_Mises_Stress']]
        st.dataframe(top_5_df.style.format({'Von_Mises_Stress': '{:.2f}'}))
        
    else:  # Detailed Information page
        st.header("Detailed Zeolite Information")
        selected_zeolite = st.selectbox("Select a Zeolite", filtered_df['Zeolite'])
        zeolite_data = filtered_df[filtered_df['Zeolite'] == selected_zeolite].iloc[0]

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Basic Information")
            st.write(f"Number of atoms: {zeolite_data['number_of_atoms']}")
            st.write(f"Chemical formula: {zeolite_data['chemical_formula']}")
            st.write(f"Energy: {zeolite_data['energy (mev)']:.2f} meV")
            st.write(f"Energy per atom: {zeolite_data['energy (mev)']/zeolite_data['number_of_atoms']:.4f} meV/atom")

        with col2:
            st.subheader("Stress Tensor (meV/Å³)")
            stress_df = pd.DataFrame(zeolite_data['stress_tensor (meV/Å³)'], 
                                   columns=['x', 'y', 'z'])
            st.write(stress_df)

        st.subheader("Zeolite Structure Visualization")
        structure_fig = plot_zeolite_structure_with_info(
            np.array(zeolite_data['positions']),
            np.array(zeolite_data['atomic_numbers']),
            np.array(zeolite_data['force (meV/Å)']),
            np.array(zeolite_data['charge'])
        )
        st.plotly_chart(structure_fig)

if __name__ == "__main__":
    main()