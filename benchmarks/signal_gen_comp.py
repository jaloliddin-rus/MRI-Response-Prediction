#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Physics-based Signal Generation Benchmarking Script
Compatible with Python 3.7 and VirtualMRI environment

Run this first to benchmark signal generation times.
Results are saved to pickle files for the deep learning benchmarking script.

Author: Analysis Script
Created: 2024
"""

import os
import numpy as np
import pickle
import time
import argparse
from tqdm import tqdm
import traceback
import logging
from statistics import mean, stdev

# Import VirtualMRI components
import VirtualMRI as vmri
from VascGraph.GraphIO import ReadPajek
import VascGraph as vg

def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark physics-based signal generation times.")
    parser.add_argument('--data_dir', type=str, default='data',
                        help='Directory containing graph data')
    parser.add_argument('--num_samples', type=int, default=50,
                        help='Number of samples to benchmark for signal generation')
    parser.add_argument('--output_dir', type=str, default='timing_results',
                        help='Directory to save timing results')
    parser.add_argument('--output_file', type=str, default='signal_generation_times.pkl',
                        help='Output pickle file for timing results')
    return parser.parse_args()

def get_signal(g, B0=None, TE=None, delta_small=None, delta_big=None, ASL=None, b=None):
    """Extract diffusion spin echo signals"""
    configpath = "../virtualMRI/config.txt"
    exp = vmri.MRI.DiffusionExp(configpath=configpath, n_protons=1e5)
    
    if B0 is not None:
        exp.config.set('MRI', 'B0', str(B0))
    if TE is not None:
        exp.config.set('MRI', 'TE', str(TE))
    if delta_small is not None:
        exp.config.set('MRI', 'delta_small', str(delta_small))
    if delta_big is not None:
        exp.config.set('MRI', 'delta_big', str(delta_big))
    if ASL is not None:
        exp.config.set('MRI', 'with_spin_labeling', 'yes' if ASL else 'no')
    if b is not None:
        exp.config.set('Gradients', 'b_values', f'0, {b}')
    
    exp.Run(g)
    return exp.Exp, exp.Exp.signals[0]

def preprocess_graph(g, scale=0.5):
    """Preprocessing of oct vascular graphs"""
    print('--Refining radius...')
    g.RefineRadiusOnSegments(rad_mode='median')
    rad = np.array(g.GetRadii())
    rad[np.isnan(rad)] = 2.0
    for i, r in zip(g.GetNodes(), rad):
        g.node[i]['r'] = r
    
    # flip and fix minus node positions
    pos = np.array(g.GetNodesPos())
    minp = np.min(pos, axis=0)
    minp[minp < 0] = minp[minp < 0] * -1
    minp[minp < 1] = minp[minp < 1] + 1
    for i, p in zip(g.GetNodes(), pos):
        g.node[i]['pos'] = np.array([p[1], p[2], p[0]]) + minp
    
    # scaling the domain
    pos = np.array(g.GetNodesPos())
    pos = pos * scale
    g.SetNodesPos(pos)
    
    # crop # sphere # isotropic
    pos = np.array(g.GetNodesPos())
    maxp = np.max(pos, axis=0)
    remove = []
    for i, p in zip(g.GetNodes(), pos):
        if p[2] > maxp[2] * 0.66:
            remove.append(i)
    remove = list(set(remove))
    g.remove_nodes_from(remove)
    g = vg.Tools.CalcTools.fixG(g)
    
    return g

def validate_delta_values(delta_small, delta_big):
    """Validate delta parameter values"""
    if delta_small >= delta_big:
        raise ValueError(f"delta_small ({delta_small}) must be less than delta_big ({delta_big})")
    if delta_small <= 0 or delta_big <= 0:
        raise ValueError(f"delta_small ({delta_small}) and delta_big ({delta_big}) must be positive")

def benchmark_signal_generation(data_dir, num_samples, output_dir, output_file):
    """Benchmark physics-based signal generation time"""
    print(f"=== Physics-based Signal Generation Benchmarking ===")
    print(f"Benchmarking signal generation for {num_samples} samples...")
    print(f"Python version: {os.sys.version}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Parameters from your generate_signal.py
    B0 = 7
    b_values = [50, 100, 500]
    delta_pairs = [(1, 4), (1.5, 5), (2, 7)]
    
    generation_times = []
    successful_samples = []
    failed_samples = []
    
    # Find available graph files
    graph_files = []
    for animal in os.listdir(data_dir):
        animalpath = os.path.join(data_dir, animal)
        if os.path.isdir(animalpath):
            for chunk in os.listdir(animalpath):
                chunkpath = os.path.join(animalpath, chunk)
                graphpath = os.path.join(chunkpath, 'chunk.pajek')
                if os.path.exists(graphpath):
                    graph_files.append({
                        'path': graphpath,
                        'animal': animal,
                        'chunk': chunk
                    })
                    if len(graph_files) >= num_samples:
                        break
            if len(graph_files) >= num_samples:
                break
    
    print(f"Found {len(graph_files)} graph files")
    print(f"Using first {min(num_samples, len(graph_files))} samples")
    
    # Benchmark each sample
    for i, graph_info in enumerate(tqdm(graph_files[:num_samples], desc="Generating signals")):
        try:
            graphpath = graph_info['path']
            animal = graph_info['animal']
            chunk = graph_info['chunk']
            
            print(f"\nProcessing sample {i+1}/{min(num_samples, len(graph_files))}: {animal}/{chunk}")
            
            start_time = time.time()
            
            # Load and preprocess graph
            print("  - Loading graph...")
            g = ReadPajek(graphpath, mode='di').GetOutput()
            
            print("  - Preprocessing graph...")
            g = preprocess_graph(g)
            
            # Assign oxygen quantities
            print("  - Assigning PO2/SO2...")
            oxyg = vmri.Graphing.OxyGraph(g)
            oxyg.Update()
            g = oxyg.GetOuput()
            g = vg.Tools.CalcTools.fixG(g)
            
            # Generate signals for all parameter combinations
            print("  - Generating MRI signals...")
            signals_generated = 0
            for b in b_values:
                for delta_small, delta_big in delta_pairs:
                    validate_delta_values(delta_small, delta_big)
                    exp, spin_echo_signal = get_signal(g, B0=B0, b=b, 
                                                     delta_small=delta_small, 
                                                     delta_big=delta_big)
                    signals_generated += len(exp.signals)  # Count all signals generated
            
            end_time = time.time()
            generation_time = end_time - start_time
            generation_times.append(generation_time)
            
            successful_samples.append({
                'animal': animal,
                'chunk': chunk,
                'time': generation_time,
                'signals_generated': signals_generated
            })
            
            print(f"  ✓ Completed in {generation_time:.2f} seconds ({signals_generated} signals)")
            
        except Exception as e:
            error_msg = f"Error processing {graphpath}: {str(e)}"
            print(f"  ✗ {error_msg}")
            logging.error(error_msg)
            logging.debug(traceback.format_exc())
            failed_samples.append({
                'path': graphpath,
                'animal': animal,
                'chunk': chunk,
                'error': str(e)
            })
            continue
    
    # Calculate statistics
    if generation_times:
        avg_time = mean(generation_times)
        std_time = stdev(generation_times) if len(generation_times) > 1 else 0
        min_time = min(generation_times)
        max_time = max(generation_times)
        
        print(f"\n=== SIGNAL GENERATION RESULTS ===")
        print(f"Successful samples: {len(successful_samples)}")
        print(f"Failed samples: {len(failed_samples)}")
        print(f"Average time per sample: {avg_time:.2f} ± {std_time:.2f} seconds")
        print(f"Min time: {min_time:.2f} seconds")
        print(f"Max time: {max_time:.2f} seconds")
        print(f"Total time: {sum(generation_times):.2f} seconds")
        
        # Save results to pickle file
        results = {
            'generation_times': generation_times,
            'successful_samples': successful_samples,
            'failed_samples': failed_samples,
            'statistics': {
                'mean': avg_time,
                'std': std_time,
                'min': min_time,
                'max': max_time,
                'total': sum(generation_times),
                'count': len(generation_times)
            },
            'parameters': {
                'B0': B0,
                'b_values': b_values,
                'delta_pairs': delta_pairs,
                'num_samples': num_samples
            },
            'metadata': {
                'python_version': os.sys.version,
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
            }
        }
        
        output_path = os.path.join(output_dir, output_file)
        with open(output_path, 'wb') as f:
            pickle.dump(results, f)
        
        print(f"\nResults saved to: {output_path}")
        print(f"Next: Run the deep learning benchmarking script in your Python 3.10 environment")
        
        return True
    else:
        print(f"\n❌ ERROR: No samples were successfully processed!")
        print("Check your data directory and ensure graph files exist.")
        return False

def main():
    args = parse_args()
    
    print("=== Physics-based Signal Generation Benchmarking ===")
    print(f"Data directory: {args.data_dir}")
    print(f"Number of samples: {args.num_samples}")
    print(f"Output directory: {args.output_dir}")
    print(f"Output file: {args.output_file}")

    # Ensure output directory exists before setting up logging/FileHandler
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(args.output_dir, 'signal_generation_log.txt')),
            logging.StreamHandler()
        ]
    )
    
    success = benchmark_signal_generation(
        args.data_dir, 
        args.num_samples, 
        args.output_dir, 
        args.output_file
    )
    
    if success:
        print("\n✓ Signal generation benchmarking completed successfully!")
        print("You can now run the deep learning benchmarking script in your Python 3.10 environment.")
    else:
        print("\n❌ Signal generation benchmarking failed!")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())