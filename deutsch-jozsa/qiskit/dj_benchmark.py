"""
Deutsch-Jozsa Benchmark Program - Qiskit
"""
import argparse
import sys
import time

import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister

sys.path[1:1] = [ "_common", "_common/qiskit" ]
sys.path[1:1] = [ "../../_common", "../../_common/qiskit" ]
import execute as ex
import metrics as metrics

import math
from collections import Counter
from pyqrack import QrackSimulator


np.random.seed(0)

verbose = False

# saved circuits for display
QC_ = None
C_ORACLE_ = None
B_ORACLE_ = None

############### Circuit Definition

# Create a constant oracle, appending gates to given circuit
def constant_oracle (input_size, num_qubits):
    #Initialize first n qubits and single ancilla qubit
    qc = QuantumCircuit(num_qubits, name=f"Uf")

    output = np.random.randint(2)
    if output == 1:
        qc.x(input_size)

    global C_ORACLE_
    if C_ORACLE_ == None or num_qubits <= 6:
        if num_qubits < 9: C_ORACLE_ = qc

    return qc

# Create a balanced oracle.
# Perform CNOTs with each input qubit as a control and the output bit as the target.
# Vary the input states that give 0 or 1 by wrapping some of the controls in X-gates.
def balanced_oracle (input_size, num_qubits):
    #Initialize first n qubits and single ancilla qubit
    qc = QuantumCircuit(num_qubits, name=f"Uf")

    b_str = "10101010101010101010"              # permit input_string up to 20 chars
    for qubit in range(input_size):
        if b_str[qubit] == '1':
            qc.x(qubit)

    qc.barrier()

    for qubit in range(input_size):
        qc.cx(qubit, input_size)

    qc.barrier()

    for qubit in range(input_size):
        if b_str[qubit] == '1':
            qc.x(qubit)

    global B_ORACLE_
    if B_ORACLE_ == None or num_qubits <= 6:
        if num_qubits < 9: B_ORACLE_ = qc

    return qc
# Create benchmark circuit
def DeutschJozsa (num_qubits, type):
    
    # Size of input is one less than available qubits
    input_size = num_qubits - 1

    # allocate qubits
    qr = QuantumRegister(num_qubits); cr = ClassicalRegister(input_size); qc = QuantumCircuit(qr, cr, name="main")

    for qubit in range(input_size):
        qc.h(qubit)
    qc.x(input_size)
    qc.h(input_size)
    
    qc.barrier()
    
    # Add a constant or balanced oracle function
    if type == 0: Uf = constant_oracle(input_size, num_qubits)
    else: Uf = balanced_oracle(input_size, num_qubits)
    qc.append(Uf, qr)

    qc.barrier()
    
    for qubit in range(num_qubits):
        qc.h(qubit)
    
    # uncompute ancilla qubit, not necessary for algorithm
    qc.x(input_size)
    
    qc.barrier()
    
    for i in range(input_size):
        qc.measure(i, i)
    
    # save smaller circuit and oracle subcircuit example for display
    global QC_
    if QC_ == None or num_qubits <= 6:
        if num_qubits < 9: QC_ = qc

    # return a handle to the circuit
    return qc

############### Result Data Analysis

# Analyze and print measured results
# Expected result is always the type, so fidelity calc is simple
def analyze_and_print_result (qc, result, num_qubits, type, num_shots):
    
    # Size of input is one less than available qubits
    input_size = num_qubits - 1

    # obtain counts from the result object
    counts = result.get_counts(qc)
    if verbose: print(f"For type {type} measured: {counts}")
    
    # create the key that is expected to have all the measurements (for this circuit)
    if type == 0: key = '0'*input_size
    else: key = '1'*input_size
    
    # correct distribution is measuring the key 100% of the time
    correct_dist = {key: 1.0}

    # use our polarization fidelity rescaling
    fidelity = metrics.polarization_fidelity(counts, correct_dist)

    return counts, fidelity

################ Benchmark Loop

# Execute program with default parameters
def run (min_qubits=3, max_qubits=8, max_circuits=3, num_shots=100,
        backend_id='qasm_simulator', provider_backend=None,
        hub="ibm-q", group="open", project="main", exec_options=None):

    print("Deutsch-Jozsa Benchmark Program - Qiskit")

    # validate parameters (smallest circuit is 3 qubits)
    max_qubits = max(3, max_qubits)
    min_qubits = min(max(3, min_qubits), max_qubits)
    #print(f"min, max qubits = {min_qubits} {max_qubits}")
    
    # Initialize metrics module
    metrics.init_metrics()

    # Define custom result handler
    def execution_handler (qc, result, num_qubits, type, num_shots):  
     
        # determine fidelity of result set
        num_qubits = int(num_qubits)
        counts, fidelity = analyze_and_print_result(qc, result, num_qubits, int(type), num_shots)
        metrics.store_metric(num_qubits, type, 'fidelity', fidelity)

    # Initialize execution module using the execution result handler above and specified backend_id
    ex.init_execution(execution_handler)
    ex.set_execution_target(backend_id, provider_backend=provider_backend,
            hub=hub, group=group, project=project, exec_options=exec_options)
    
    # Execute Benchmark Program N times for multiple circuit sizes
    # Accumulate metrics asynchronously as circuits complete
    for num_qubits in range(min_qubits, max_qubits + 1):
    
        input_size = num_qubits - 1
        
        # determine number of circuits to execute for this group
        num_circuits = min(2, max_circuits)
        
        print(f"************\nExecuting [{num_circuits}] circuits with num_qubits = {num_qubits}")
        
        # loop over only 2 circuits
        for type in range( num_circuits ):
            
            # create the circuit for given qubit size and secret string, store time metric
            ts = time.time()
            qc = DeutschJozsa(num_qubits, type)
            metrics.store_metric(num_qubits, type, 'create_time', time.time()-ts)

            # collapse the sub-circuit levels used in this benchmark (for qiskit)
            qc2 = qc.decompose()

            # submit circuit for execution on target (simulator, cloud simulator, or hardware)
            ex.submit_circuit(qc2, num_qubits, type, num_shots)
        
        # Wait for some active circuits to complete; report metrics when groups complete
        ex.throttle_execution(metrics.finalize_group)
    
    # Wait for all active circuits to complete; report metrics when groups complete
    ex.finalize_execution(metrics.finalize_group)

    # print a sample circuit
    print("Sample Circuit:"); print(QC_ if QC_ != None else "  ... too large!")
    print("\nConstant Oracle 'Uf' ="); print(C_ORACLE_ if C_ORACLE_ != None else " ... too large or not used!")
    print("\nBalanced Oracle 'Uf' ="); print(B_ORACLE_ if B_ORACLE_ != None else " ... too large or not used!")

    # Plot metrics for all circuit sizes
    #metrics.plot_metrics("Benchmark Results - Deutsch-Jozsa - Qiskit")
    return metrics.extract_data()

# Hypothetically, the register_width might be known from any permutation output,
# but there might be cases where that wouldn't be true.
# Also, for convenience, PyQrack can interpret a Qiskit circuit,
# (with or without its Qiskit "Provider,"" or a PyZX circuit, or Cirq with its plugin, only)
def fidelities_from_measurement_results(results, qiskit_circuit, register_width, ideal_shots = 1024):
    
    sim = QrackSimulator(qubitCount=register_width, qiskitCircuit = qiskit_circuit)
    ideal_result = sim.measure_shots(list(range(register_width)), ideal_shots)

    fidelity_list = []
    for _, measurement_list in results.items():
        # This is a logically-grouped batch of qubit measurement results, as a list of "permutations."
        histogram = Counter(measurement_list)
        shot_count = sum(histogram.values())
        histogram = dict(histogram)
        fidelity = 0
        for qubit_permutation in histogram.keys():
            ideal_normalized_frequency = ideal_result[qubit_permutation] if qubit_permutation in ideal_result else 0
            normalized_frequency = histogram[qubit_permutation]
            fidelity += math.sqrt(ideal_normalized_frequency * normalized_frequency)
        fidelity *= fidelity
        # See https://github.com/SRI-International/QC-App-Oriented-Benchmarks/blob/master/_doc/POLARIZATION_FIDELITY.md
        normalized_fidelity = (fidelity - 1) / (1 - (1 << register_width)) + 1
        fidelity = 1 - fidelity
        fidelity_list.append((fidelity, normalized_fidelity))
                               
    return fidelity_list

# if main, execute method
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Run benchmarking")

    parser.add_argument("-backend_id", default="qasm_simulator", help="Backend simulator or hardware string", type=str)
    parser.add_argument("-min_qubits", default=2, help="Minimum number of qubits.", type=int)
    parser.add_argument("-max_qubits", default=8, help="Maximum number of qubits", type=int)
    parser.add_argument("-num_shots", default=100, help="Number of shots.", type=int)

    args = parser.parse_args()

    backend_id = args.backend_id
    min_qubits = args.min_qubits
    max_qubits = args.max_qubits
    num_shots = args.max_qubits

    #print(run(backend_id=backend_id, min_qubits=min_qubits, max_qubits=max_qubits, num_shots=num_shots))
    # QUANTINUUM
    from api_wrappers import QuantinuumAPI as QAPI
    from qiskit import transpile

    basis_gates = ['u1', 'u2', 'u3', 'cx', 'u']

    num_qubits = 3
    qc = DeutschJozsa(num_qubits, 1)
    qc = transpile(qc, basis_gates=basis_gates)
    openqasm = qc.qasm()

    machine = 'H1-2E'
    shots = 1_000
    # Submit circuit to the emulator
    qapi = QAPI(machine=machine)
    job_id = qapi.submit_job(openqasm,
                             shots=shots,
                             machine=machine, 
                             name='circuit emulation')
   
    status = qapi.retrieve_job_status(job_id)
    
    print(status)
    
    results = qapi.retrieve_job(job_id)
    results = results["results"]
    print(fidelities_from_measurement_results(results=results, qiskit_circuit=qc, register_width=num_qubits, ideal_shots=shots))
    # QUANTINUUM