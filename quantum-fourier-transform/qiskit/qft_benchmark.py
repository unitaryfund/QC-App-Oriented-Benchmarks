"""
Quantum Fourier Transform Benchmark Program - Qiskit
"""
import argparse
import math
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

verbose = True

# saved circuits for display
num_gates = 0
depth = 0
QC_ = None
QFT_ = None
QFTI_ = None

############### Circuit Definition

def QuantumFourierTransform (num_qubits, secret_int, method=1):
    global num_gates, depth
    # Size of input is one less than available qubits
    input_size = num_qubits
    num_gates = 0
    depth = 0
    
    # allocate qubits
    qr = QuantumRegister(num_qubits); cr = ClassicalRegister(num_qubits); qc = QuantumCircuit(qr, cr, name="main")

    if method==1:

        # Perform X on each qubit that matches a bit in secret string
        s = ('{0:0'+str(input_size)+'b}').format(secret_int)
        for i_qubit in range(input_size):
            if s[input_size-1-i_qubit]=='1':
                qc.x(qr[i_qubit])
                num_gates += 1

        depth += 1

        qc.barrier()

        # perform QFT on the input
        qc.append(qft_gate(input_size).to_instruction(), qr)

        # End with Hadamard on all qubits (to measure the z rotations)
        ''' don't do this unless NOT doing the inverse afterwards
        for i_qubit in range(input_size):
             qc.h(qr[i_qubit])

        qc.barrier()
        '''

        qc.barrier()
        
        # some compilers recognize the QFT and IQFT in series and collapse them to identity;
        # perform a set of rotations to add one to the secret_int to avoid this collapse
        for i_q in range(0, num_qubits):
            divisor = 2 ** (i_q)
            qc.rz( 1 * math.pi / divisor , qr[i_q])
            num_gates+=1
        
        qc.barrier()

        # to revert back to initial state, apply inverse QFT
        qc.append(inv_qft_gate(input_size).to_instruction(), qr)

        qc.barrier()

    elif method == 2:

        for i_q in range(0, num_qubits):
            qc.h(qr[i_q])
            num_gates += 1

        for i_q in range(0, num_qubits):
            divisor = 2 ** (i_q)
            qc.rz(secret_int * math.pi / divisor, qr[i_q])
            num_gates += 1

        depth += 1

        qc.append(inv_qft_gate(input_size).to_instruction(), qr)

    # This method is a work in progress
    elif method==3:

        for i_q in range(0, secret_int):
            qc.h(qr[i_q])
            num_gates+=1

        for i_q in range(secret_int, num_qubits):
            qc.x(qr[i_q])
            num_gates+=1
            
        depth += 1
        
        qc.append(inv_qft_gate(input_size).to_instruction(), qr)
        
    else:
        exit("Invalid QFT method")

    # measure all qubits
    # COMMENTING OUT DUE TO QUANDELA
    qc.measure(qr, cr)
    num_gates += num_qubits
    depth += 1

    # save smaller circuit example for display
    global QC_    
    if QC_ == None or num_qubits <= 5:
        if num_qubits < 9: QC_ = qc
        
    # return a handle on the circuit
    return qc

############### QFT Circuit

def qft_gate(input_size):
    global QFT_, num_gates, depth
    qr = QuantumRegister(input_size); qc = QuantumCircuit(qr, name="qft")
    
    # Generate multiple groups of diminishing angle CRZs and H gate
    for i_qubit in range(0, input_size):
    
        # start laying out gates from highest order qubit (the hidx)
        hidx = input_size - i_qubit - 1
        
        # if not the highest order qubit, add multiple controlled RZs of decreasing angle
        if hidx < input_size - 1:   
            num_crzs = i_qubit
            for j in range(0, num_crzs):
                divisor = 2 ** (num_crzs - j)
                qc.crz( math.pi / divisor , qr[hidx], qr[input_size - j - 1])
                num_gates += 1
                depth += 1
            
        # followed by an H gate (applied to all qubits)
        qc.h(qr[hidx])
        num_gates += 1
        depth += 1
        
        qc.barrier()
    
    if QFT_ == None or input_size <= 5:
        if input_size < 9: QFT_ = qc
        
    return qc

############### Inverse QFT Circuit

def inv_qft_gate(input_size):
    global QFTI_, num_gates, depth
    qr = QuantumRegister(input_size); qc = QuantumCircuit(qr, name="inv_qft")
    
    # Generate multiple groups of diminishing angle CRZs and H gate
    for i_qubit in reversed(range(0, input_size)):
    
        # start laying out gates from highest order qubit (the hidx)
        hidx = input_size - i_qubit - 1
        
        # precede with an H gate (applied to all qubits)
        qc.h(qr[hidx])
        num_gates += 1
        depth += 1
        
        # if not the highest order qubit, add multiple controlled RZs of decreasing angle
        if hidx < input_size - 1:   
            num_crzs = i_qubit
            for j in reversed(range(0, num_crzs)):
                divisor = 2 ** (num_crzs - j)
                qc.crz( -math.pi / divisor , qr[hidx], qr[input_size - j - 1])
                num_gates += 1
                depth += 1
            
        qc.barrier()  
    
    if QFTI_ == None or input_size <= 5:
        if input_size < 9: QFTI_= qc
        
    return qc
    
# Define expected distribution calculated from applying the iqft to the prepared secret_int state
def expected_dist(num_qubits, secret_int, counts):
    dist = {}
    s = num_qubits - secret_int
    for key in counts.keys():
        if key[(num_qubits-secret_int):] == ''.zfill(secret_int):
            dist[key] = 1/(2**s)
    return dist

############### Result Data Analysis

# Analyze and print measured results
def analyze_and_print_result (qc, result, num_qubits, secret_int, num_shots, method):

    # obtain counts from the result object
    counts = result.get_counts(qc)

    # For method 1, expected result is always the secret_int
    if method==1:
        
        # add one to the secret_int to compensate for the extra rotations done between QFT and IQFT
        secret_int_plus_one = (secret_int + 1) % (2 ** num_qubits)

        # create the key that is expected to have all the measurements (for this circuit)
        key = format(secret_int_plus_one, f"0{num_qubits}b")

        # correct distribution is measuring the key 100% of the time
        correct_dist = {key: 1.0}
        
    # For method 2, expected result is always the secret_int
    elif method==2:

        # create the key that is expected to have all the measurements (for this circuit)
        key = format(secret_int, f"0{num_qubits}b")

        # correct distribution is measuring the key 100% of the time
        correct_dist = {key: 1.0}
    
    # For method 3, correct_dist is a distribution with more than one value
    elif method==3:

        # correct_dist is from the expected dist
        correct_dist = expected_dist(num_qubits, secret_int, counts)
            
    # use our polarization fidelity rescaling
    fidelity = metrics.polarization_fidelity(counts, correct_dist)

    if verbose: print(f"For secret int {secret_int} measured: {counts} fidelity: {fidelity}")

    return counts, fidelity

################ Benchmark Loop

# Execute program with default parameters
def run (min_qubits = 2, max_qubits = 8, max_circuits = 3, num_shots = 100,
        method=1, 
        backend_id='qasm_simulator', provider_backend=None,
        hub="ibm-q", group="open", project="main", exec_options=None):

    print("Quantum Fourier Transform Benchmark Program - Qiskit")
    print(f"... using circuit method {method}")

    # validate parameters (smallest circuit is 2 qubits)
    max_qubits = max(2, max_qubits)
    min_qubits = min(max(2, min_qubits), max_qubits)
    #print(f"min, max qubits = {min_qubits} {max_qubits}")
    
    # Initialize metrics module
    metrics.init_metrics()

    # Define custom result handler
    def execution_handler (qc, result, input_size, s_int, num_shots):  
     
        # determine fidelity of result set
        num_qubits = int(input_size)
        counts, fidelity = analyze_and_print_result(qc, result, num_qubits, int(s_int), num_shots, method)
        metrics.store_metric(input_size, s_int, 'fidelity', fidelity)

    # Initialize execution module using the execution result handler above and specified backend_id
    ex.init_execution(execution_handler)
    ex.set_execution_target(backend_id, provider_backend=provider_backend,
            hub=hub, group=group, project=project, exec_options=exec_options)

    # Execute Benchmark Program N times for multiple circuit sizes
    # Accumulate metrics asynchronously as circuits complete
    for input_size in range(min_qubits, max_qubits + 1):
        
        # reset random seed 
        np.random.seed(0)

        num_qubits = input_size

        # determine number of circuits to execute for this group
        # and determine range of secret strings to loop over
        if method == 1 or method == 2:
            num_circuits = min(2 ** (input_size), max_circuits)
        
            if 2**(input_size) <= max_circuits:
                s_range = list(range(num_circuits))
            else:
                s_range = np.random.choice(2**(input_size), num_circuits, False)
         
        elif method == 3:
            num_circuits = min(input_size, max_circuits)

            if input_size <= max_circuits:
                s_range = list(range(num_circuits))
            else:
                s_range = np.random.choice(range(input_size), num_circuits, False)
        
        else:
            sys.exit("Invalid QFT method")

        print(f"************\nExecuting [{num_circuits}] circuits with num_qubits = {num_qubits}")
        
        # loop over limited # of secret strings for this
        for s_int in s_range:

            # create the circuit for given qubit size and secret string, store time metric
            ts = time.time()
            qc = QuantumFourierTransform(num_qubits, s_int, method=method)
            metrics.store_metric(input_size, s_int, 'create_time', time.time()-ts)

            # collapse the sub-circuits used in this benchmark (for qiskit)
            qc2 = qc.decompose()
            
            # submit circuit for execution on target (simulator, cloud simulator, or hardware)

            ex.submit_circuit(qc2, input_size, s_int, num_shots)
        
        print(f"... number of gates, depth = {num_gates}, {depth}")
        
        # Wait for some active circuits to complete; report metrics when groups complete
        ex.throttle_execution(metrics.finalize_group)

    # Wait for all active circuits to complete; report metrics when groups complete
    ex.finalize_execution(metrics.finalize_group)
    
    # print a sample circuit created (if not too large)
    print("Sample Circuit:"); print(QC_ if QC_ != None else "  ... too large!")
    if method==1:
        print("\nQFT Circuit ="); print(QFT_)
    print("\nInverse QFT Circuit ="); print(QFTI_)
     
    # Plot metrics for all circuit sizes
    metrics.plot_metrics(f"Benchmark Results - Quantum Fourier Transform ({method}) - Qiskit")
    return metrics.extract_data()


# Hypothetically, the register_width might be known from any permutation output,
# but there might be cases where that wouldn't be true.
# Also, for convenience, PyQrack can interpret a Qiskit circuit,
# (with or without its Qiskit "Provider,"" or a PyZX circuit, or Cirq with its plugin, only)
def fidelities_from_measurement_results(results, qiskit_circuit, register_width, ideal_shots = 1024):
    
    perm_count = 1 << register_width
    
    sim = QrackSimulator(qubitCount=register_width, qiskitCircuit = qiskit_circuit)
    ideal_result = sim.measure_shots(list(range(register_width)), ideal_shots)
    ideal_result = dict(Counter(ideal_result))
    for key, value in ideal_result.items():
        ideal_result[key] = value / ideal_shots

    fidelity_list = []
    for _, measurement_list in results.items():
        # This is a logically-grouped batch of qubit measurement results, as a list of "permutations."
        t_histogram = Counter(measurement_list)
        shot_count = sum(t_histogram.values())
        t_histogram = dict(t_histogram)
        
        histogram = {}
        for key, value in t_histogram.items():
            histogram[int(key, 2)] = value / shot_count
        
        fidelity = 0
        for qubit_permutation in histogram.keys():
            ideal_normalized_frequency = ideal_result[qubit_permutation] if qubit_permutation in ideal_result else 0
            normalized_frequency = histogram[qubit_permutation]
            fidelity += math.sqrt(ideal_normalized_frequency * normalized_frequency)
        fidelity *= fidelity
        # See https://github.com/SRI-International/QC-App-Oriented-Benchmarks/blob/master/_doc/POLARIZATION_FIDELITY.md
        s = (perm_count - 1) / (perm_count * (1 - len(ideal_result) / perm_count))
        normalized_fidelity = s * (fidelity - 1) + 1
        fidelity_list.append((fidelity, normalized_fidelity))
                               
    return fidelity_list
    

# if main, execute method 1
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
    #exit()

    # # QUANTINUUM
    import os
    from qiskit_quantinuum import QuantinuumProvider
    from qiskit import execute
    provider = QuantinuumProvider()
    provider.save_account(os.environ.get("QUANTINUUM_USERNAME"))
    backends = provider.backends()
    backend = provider.get_backend("H1-1")
    print(run(max_qubits=8, max_circuits=3, method=1, num_shots=1000, provider_backend=backend))
    exit()
    Quantinuum.save_account(os.environ.get("QUANTINUUM_USERNAME"))
    #Quantinuum.load_account()

    backends = Quantinuum.backends()
    backend = Quantinuum.get_backend("H1-2E")
    print(run(provider_backend=backend, min_qubits=min_qubits, max_qubits=max_qubits, num_shots=num_shots))
    exit()

    qc = QuantumCircuit(2, 2)
    qc.h(0)
    qc.cx(0, 1)
    qc.measure([0,1], [0,1])
    result = execute(qc, backend).result()
    print(result.get_counts())


    # from api_wrappers import QuantinuumAPI as QAPI
    # from qiskit import transpile

    # basis_gates = ['u1', 'u2', 'u3', 'cx', 'u']

    # num_qubits = 3
    # qc = QuantumFourierTransform(num_qubits, 1).decompose(reps=99)
    # qc = transpile(qc, basis_gates=basis_gates)
    # openqasm = qc.qasm()

    # machine = 'H1-2E'
    # shots = 1_000
    # # Submit circuit to the emulator
    # import os
    # from qiskit.providers.honeywell import Honeywell
    # USERNAME = os.environ.get("QUANTINUUM_USERNAME")
    # PASSWORD = os.environ.get("QUANTINUUM_PASSWORD")
    # Honeywell.save_account(USERNAME, 
    #                        proxies = { 'urls': {'http':
    #                        f'http://{USERNAME}:{PASSWORD}@qapi.quantinuum.com/',
    #                        'https':
    #                        f'http://{USERNAME}:{PASSWORD}@qapi.quantinuum.com/'
    #                        } })

    # backends = Honeywell.backends()
    # backend = Honeywell.get_backend(machine)
    # qapi = QAPI(machine=machine)
    # job_id = qapi.submit_job(openqasm,
    #                          shots=shots,
    #                          machine=machine, 
    #                          name='circuit emulation')
   
    # status = qapi.retrieve_job_status(job_id)
    
    # print(status)
    
    # results = qapi.retrieve_job(job_id)
    # results = results["results"]
    # print(fidelities_from_measurement_results(results=results, qiskit_circuit=qc, register_width=num_qubits, ideal_shots=shots))
    # # QUANTINUUM

    # QUANDELA
        
    # NOTE MUST COMMENT OUT THESE LINES ABOVE TO RUN ON QUANDELA:
    #####
    # measure all qubits
    # COMMENTING OUT DUE TO QUANDELA
    # qc.measure(qr, cr)

    # import qiskit
    # import perceval as pcvl
    # from perceval.converters import QiskitConverter
    # from perceval.components import catalog

    # token_qcloud = '_T_eyJhbGciOiJIUzUxMiIsImlhdCI6MTY3OTk1NTIxMywiZXhwIjoxNjgyNTQ3MjEzfQ.eyJpZCI6MjgwfQ.Mcc5erbyzhfBT8SYugGpExNkhfl0oEU1GvjDI5-u0hK5LCEq--m4c581Ak4unTMIV5OKLU29uuX9wKen-0AvlA'

    # # Then convert the Quantum Circuit with Perceval QiskitConvertor
    # qiskit_converter = QiskitConverter(catalog, backend_name="Naive")
    # quantum_processor = qiskit_converter.convert(qc, use_postselection=False)
    # pcvl.pdisplay(quantum_processor, recursive=True)
    # QUANDELA

    #print(run(backend_id=backend_id, min_qubits=min_qubits, max_qubits=max_qubits, num_shots=num_shots))
