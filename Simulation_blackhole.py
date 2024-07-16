import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from scipy import stats
from mpl_toolkits.mplot3d import Axes3D
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit_ibm_runtime import QiskitRuntimeService, Sampler
from qiskit import transpile
import numpy as np
import matplotlib.pyplot as plt
import json
import h5py
# Initialize the QiskitRuntimeService
service = QiskitRuntimeService(channel="ibm_quantum", token="")

def create_black_hole_collision_circuit(num_qubits, mass_ratio, spin_parameter):
    qr = QuantumRegister(num_qubits, 'q')
    cr = ClassicalRegister(num_qubits, 'c')
    circuit = QuantumCircuit(qr, cr)

    # Encode black hole properties
    for i in range(num_qubits // 2):
        circuit.ry(mass_ratio * np.pi, qr[i])
        circuit.rz(spin_parameter * np.pi, qr[i])

    # Simulate gravitational interaction
    for i in range(num_qubits // 2, num_qubits):
        circuit.cx(qr[i-num_qubits//2], qr[i])
        circuit.rz(mass_ratio * np.pi / 2, qr[i])

    # Model spacetime distortion
    circuit.h(qr[0])
    for i in range(1, num_qubits):
        circuit.cp(np.pi/2**(i), qr[0], qr[i])

    circuit.measure(qr, cr)
    return circuit

# Simulation parameters
num_qubits = 8
mass_ratio = 0.8  # Ratio of masses of the two black holes
spin_parameter = 0.6  # Spin of the black holes
shots = 4000

# Create and run the circuit
collision_circuit = create_black_hole_collision_circuit(num_qubits, mass_ratio, spin_parameter)
transpiled_circuit = transpile(collision_circuit, backend)

# Use Sampler primitive with QiskitRuntimeService
backend = service.get_backend('ibm_brisbane')  # or any other available backend
sampler = Sampler(backend=backend)
job = sampler.run(transpiled_circuit, shots=shots)
result = job.result()


# Process and visualize results
counts = result.quasi_dists[0]
counts = {format(int(state), f'0{num_qubits}b'): prob for state, prob in counts.items()}


df = pd.DataFrame(list(counts.items()), columns=['State', 'Probability'])
df['Decimal'] = df['State'].apply(lambda x: int(x, 2))

df.to_csv('black_hole_collision_data.csv', index=False)

np.save('black_hole_collision_data.npy', df.values)


with open('black_hole_collision_data.json', 'w') as f:
    json.dump(counts, f)

# Create a structured array
dtype = [('State', 'S8'), ('Probability', 'f8'), ('Decimal', 'i4')]
structured_data = np.array(list(zip(df['State'].values, df['Probability'].values, df['Decimal'].values)), dtype=dtype)

# Save to HDF5
with h5py.File('black_hole_collision_simulation.h5', 'w') as hf:
    hf.create_dataset('state_probabilities', data=structured_data)
    hf.create_dataset('parameters/num_qubits', data=num_qubits)
    hf.create_dataset('parameters/mass_ratio', data=mass_ratio)
    hf.create_dataset('parameters/spin_parameter', data=spin_parameter)


# 1. CDF Plot
plt.figure(figsize=(12, 6))
stats.cumfreq(df['Probability'], numbins=50)
plt.plot(stats.cumfreq(df['Probability'], numbins=50)[0])
plt.title('Cumulative Distribution Function of State Probabilities')
plt.xlabel('Probability')
plt.ylabel('Cumulative Frequency')
plt.savefig('black_hole_collision_cdf.png')
plt.show()

# 2. Heat Map
plt.figure(figsize=(16, 8))
heatmap_data = df.pivot_table(index=df['Decimal'] // 16, 
                              columns=df['Decimal'] % 16, 
                              values='Probability', 
                              fill_value=0)
sns.heatmap(heatmap_data, cmap='YlOrRd')

plt.title('Heat Map of State Probabilities')
plt.savefig('black_hole_collision_heatmap.png')
plt.show()

# 3. 3D Surface Plot
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')
x = df['Decimal'] // 16
y = df['Decimal'] % 16
z = df['Probability']
ax.plot_trisurf(x, y, z, cmap='viridis')
plt.title('3D Surface Plot of State Probabilities')
plt.savefig('black_hole_collision_3d_surface.png')
plt.show()

# 4. Pie Chart of Top 10 States
top_10 = df.nlargest(10, 'Probability')
plt.figure(figsize=(10, 10))
plt.pie(top_10['Probability'], labels=top_10['State'], autopct='%1.1f%%')
plt.title('Top 10 Most Probable States')
plt.savefig('black_hole_collision_pie_chart.png')
plt.show()

# 5. Box Plot
plt.figure(figsize=(12, 6))
df['Qubit Sum'] = df['State'].apply(lambda x: sum(int(bit) for bit in x))
sns.boxplot(x='Qubit Sum', y='Probability', data=df)
plt.title('Distribution of Probabilities by Number of Active Qubits')
plt.savefig('black_hole_collision_box_plot.png')
plt.show()

# Printable Data
print("Summary Statistics:")
print(df['Probability'].describe())
print("\nTop 10 Most Probable States:")
print(top_10[['State', 'Probability']])
print(f"\nEntropy of the System: {stats.entropy(df['Probability'])}")

# Correlation between qubit states
correlation_matrix = df['State'].apply(lambda x: pd.Series(list(x))).astype(int).corr()
print("\nCorrelation Matrix between Qubit States:")
print(correlation_matrix)


# Analyze results
high_energy_states = sum(counts[state] for state in counts if state.count('1') > num_qubits // 2)
print(f"Probability of high-energy outcomes: {high_energy_states:.2%}")
