"""
Results Table Generator for MRI Signal Prediction Models

This script generates comprehensive evaluation and performance metrics tables from model 
training and testing results. It processes metrics files from the results directory and 
creates both CSV and LaTeX formatted tables.

Main functionality:
- Generates evaluation metrics tables (MSE, MAE, R², EVS, RMSE, R² >= 80%)
- Generates performance metrics tables (training time, RAM/GPU usage, prediction time)
- Creates "full" tables with all model configurations
- Creates "best" tables showing only the top-performing model per architecture
- Formats LaTeX tables with bold (best) and underlined (second-best) values
- Supports optional command-line flags to generate specific metric types

Usage:
    python gen_results.py                    # Generate both evaluation and performance tables
    python gen_results.py --eval_metrics     # Generate only evaluation metrics tables
    python gen_results.py --perf_metrics     # Generate only performance metrics tables

Output files created in results/:
    - evaluation_metrics_full.csv/tex
    - evaluation_metrics_best_full.csv/tex
    - evaluation_metrics_best.csv/tex
    - performance_metrics_full.csv/tex
    - performance_metrics_best_full.csv/tex
    - performance_metrics_best.csv/tex
"""

import os
import pandas as pd
import argparse
import numpy as np

def parse_args():
    parser = argparse.ArgumentParser(description="Generate results tables")
    parser.add_argument('--eval_metrics', action='store_true', help='Generate evaluation metrics table')
    parser.add_argument('--perf_metrics', action='store_true', help='Generate performance metrics table')
    return parser.parse_args()

def generate_eval_metrics():
    # Define the directories
    metrics_dir = "results/metrics"
    
    # Initialize an empty list to collect data for each experiment
    data = []
    
    # Define required columns for evaluation metrics
    required_columns = [
        "Architecture", "Loss Function", "Optimizer",
        "MSE ± SD", "MAE ± SD", "R²", "EVS", "RMSE", "R² >= 80%"
    ]
    
    # Loop through each file in the metrics directory
    for file_name in os.listdir(metrics_dir):
        if file_name.endswith(".txt"):
            # Extract model details from file name
            model_name = file_name.replace(".txt", "")
            
            # Initialize a dictionary to hold the values for this experiment
            experiment_data = {}
            
            # Read metrics file and extract relevant values
            metrics_path = os.path.join(metrics_dir, file_name)
            with open(metrics_path, "r") as f:
                # Initialize raw value variables
                mse = mae = sd_mse = sd_mae = rmse = r2 = evs = r2_percentage = None
                for line in f:
                    line = line.strip()
                    if line.startswith("Architecture:"):
                        experiment_data["Architecture"] = line.split(":", 1)[1].strip()
                    elif line.startswith("Loss Function:"):
                        experiment_data["Loss Function"] = line.split(":", 1)[1].strip()
                    elif line.startswith("Optimizer:"):
                        experiment_data["Optimizer"] = line.split(":", 1)[1].strip()
                    elif line.startswith("MSE:") and "SD" not in line:
                        try:
                            mse = float(line.split(":", 1)[1].strip())
                        except ValueError:
                            mse = None
                    elif line.startswith("SD of MSE:"):
                        try:
                            sd_mse = float(line.split(":", 1)[1].strip())
                        except ValueError:
                            sd_mse = None
                    elif line.startswith("MAE:") and "SD" not in line:
                        try:
                            mae = float(line.split(":", 1)[1].strip())
                        except ValueError:
                            mae = None
                    elif line.startswith("SD of MAE:"):
                        try:
                            sd_mae = float(line.split(":", 1)[1].strip())
                        except ValueError:
                            sd_mae = None
                    elif line.startswith("RMSE:"):
                        try:
                            rmse = float(line.split(":", 1)[1].strip())
                        except ValueError:
                            rmse = None
                    elif line.startswith("Overall R2 Score:"):
                        try:
                            r2 = float(line.split(":", 1)[1].strip())
                        except ValueError:
                            r2 = None
                    elif line.startswith("Explained Variance:"):
                        try:
                            evs = float(line.split(":", 1)[1].strip())
                        except ValueError:
                            evs = None
                    elif line.startswith("Percentage of samples with R² >= 80%:"):
                        try:
                            r2_percentage = float(line.split(":", 1)[1].strip().replace("%", ""))
                        except ValueError:
                            r2_percentage = None
            
            # Store raw values for ranking later
            experiment_data["MSE_raw"] = mse
            experiment_data["MAE_raw"] = mae
            experiment_data["RMSE_raw"] = rmse
            experiment_data["R²_raw"] = r2
            experiment_data["EVS_raw"] = evs
            experiment_data["R²_percentage_raw"] = r2_percentage
            
            # Format the metrics with updated rounding
            if mse is not None and sd_mse is not None:
                experiment_data["MSE ± SD"] = f"{mse:.4f} $\\pm$ {sd_mse:.4f}"
            else:
                experiment_data["MSE ± SD"] = "N/A"
            
            if mae is not None and sd_mae is not None:
                experiment_data["MAE ± SD"] = f"{mae:.4f} $\\pm$ {sd_mae:.4f}"
            else:
                experiment_data["MAE ± SD"] = "N/A"
            
            if rmse is not None:
                experiment_data["RMSE"] = f"{rmse:.4f}"
            else:
                experiment_data["RMSE"] = "N/A"
            
            if r2 is not None:
                experiment_data["R²"] = f"{r2:.2f}"
            else:
                experiment_data["R²"] = "N/A"
                experiment_data["R²_raw"] = -999  # A value to ensure N/A sorts last
                
            if evs is not None:
                experiment_data["EVS"] = f"{evs:.2f}"
            else:
                experiment_data["EVS"] = "N/A"
            
            if r2_percentage is not None:
                experiment_data["R² >= 80%"] = f"{r2_percentage:.2f}\\%"
            else:
                experiment_data["R² >= 80%"] = "N/A"
            
            # Append the collected data for this experiment to the list
            data.append(experiment_data)
    
    # Convert the list of dictionaries into a DataFrame
    df = pd.DataFrame(data)
    
    # Ensure all required columns are present and in the right order
    for col in required_columns:
        if col not in df.columns:
            df[col] = "N/A"
    
    # Include raw columns for sorting plus the display columns
    all_columns = required_columns + ["MSE_raw", "MAE_raw", "RMSE_raw", "R²_raw", "EVS_raw", "R²_percentage_raw"]
    df = df[all_columns]
    
    # Save the full dataframe (without raw columns)
    df_for_saving = df[required_columns].copy()
    df_for_saving.to_csv("results/evaluation_metrics_full.csv", index=False)
    
    # Generate a LaTeX table for all experiments
    generate_latex_table(df_for_saving, "results/evaluation_metrics_full.tex", "full_evaluation")
    
    # Create the best models dataframe
    best_models = pd.DataFrame()
    for arch in df['Architecture'].unique():
        # Get rows for this architecture and sort by R² (higher is better)
        arch_rows = df[df['Architecture'] == arch].sort_values('R²_raw', ascending=False)
        if not arch_rows.empty:
            best_models = pd.concat([best_models, arch_rows.iloc[:1]])
    
    # Sort by architecture for consistent ordering
    best_models = best_models.sort_values('Architecture').reset_index(drop=True)
    
    # Create best models table with reduced columns (no Loss Function or Optimizer)
    best_models_reduced = best_models.copy()
    
    # Save both full and reduced best models dataframes (without raw columns)
    best_models[required_columns].to_csv("results/evaluation_metrics_best_full.csv", index=False)
    
    best_reduced_columns = [col for col in required_columns if col not in ["Loss Function", "Optimizer"]]
    best_models_reduced = best_models_reduced[best_reduced_columns + ["MSE_raw", "MAE_raw", "RMSE_raw", "R²_raw", "EVS_raw", "R²_percentage_raw"]]
    best_models_reduced[best_reduced_columns].to_csv("results/evaluation_metrics_best.csv", index=False)
    
    # Generate LaTeX tables for best models (with and without Loss Function and Optimizer)
    generate_latex_table(best_models[required_columns], 
                        "results/evaluation_metrics_best_full.tex", 
                        "best_evaluation_full",
                        raw_df=best_models)
    
    generate_latex_table(best_models_reduced[best_reduced_columns], 
                        "results/evaluation_metrics_best.tex", 
                        "best_evaluation", 
                        include_loss_optimizer=False,
                        raw_df=best_models_reduced)
    
    print("Evaluation metrics tables generated successfully.")
    return df, best_models

def generate_perf_metrics(best_eval_models=None):
    # Define the directories
    metrics_dir = "results/metrics"
    training_summary_dir = "results/training_summary"
    
    # Initialize an empty list to collect data for each experiment
    data = []
    
    # Define required columns for performance metrics
    required_columns = [
        "Architecture", "Loss Function", "Optimizer",
        "Training Time", "Training RAM Usage", "Training GPU Memory Usage",
        "Prediction Time per Sample", "Testing GPU Memory Usage", "Testing RAM Usage"
    ]
    
    # Loop through each file in the metrics directory
    for file_name in os.listdir(metrics_dir):
        if file_name.endswith(".txt"):
            # Extract model details from file name
            model_name = file_name.replace(".txt", "")
            
            # Initialize a dictionary to hold the values for this experiment
            experiment_data = {}
            
            # Read metrics file and extract relevant values
            metrics_path = os.path.join(metrics_dir, file_name)
            with open(metrics_path, "r") as f:
                for line in f:
                    line = line.strip()
                    if line.startswith("Architecture:"):
                        experiment_data["Architecture"] = line.split(":", 1)[1].strip()
                    elif line.startswith("Loss Function:"):
                        experiment_data["Loss Function"] = line.split(":", 1)[1].strip()
                    elif line.startswith("Optimizer:"):
                        experiment_data["Optimizer"] = line.split(":", 1)[1].strip()
                    elif line.startswith("Average Prediction Time per Sample:"):
                        try:
                            prediction_time = float(line.split(":", 1)[1].strip().replace(" seconds", ""))
                            experiment_data["Prediction Time per Sample"] = f"{prediction_time:.4f}"
                            experiment_data["Prediction_Time_raw"] = prediction_time
                        except ValueError:
                            experiment_data["Prediction Time per Sample"] = "N/A"
                            experiment_data["Prediction_Time_raw"] = None
                    elif line.startswith("Average GPU Memory Usage:"):
                        try:
                            testing_gpu_memory = float(line.split(":", 1)[1].strip().replace(" MB", ""))
                            experiment_data["Testing GPU Memory Usage"] = f"{testing_gpu_memory:.2f}"
                            experiment_data["Testing_GPU_Memory_raw"] = testing_gpu_memory
                        except ValueError:
                            experiment_data["Testing GPU Memory Usage"] = "N/A"
                            experiment_data["Testing_GPU_Memory_raw"] = None
                    elif line.startswith("Average RAM Usage:"):
                        try:
                            testing_ram_usage = float(line.split(":", 1)[1].strip().replace(" MB", ""))
                            experiment_data["Testing RAM Usage"] = f"{testing_ram_usage:.2f}"
                            experiment_data["Testing_RAM_Usage_raw"] = testing_ram_usage
                        except ValueError:
                            experiment_data["Testing RAM Usage"] = "N/A"
                            experiment_data["Testing_RAM_Usage_raw"] = None
            
            # Look for training metrics in the training summary file
            # Extract architecture, loss function, and optimizer from the model_name
            parts = model_name.split("_")
            if len(parts) >= 3:
                arch = parts[1]
                loss_fn = parts[2]
                opt = parts[3]
                summary_file = f"training_summary_{arch}_{loss_fn}_{opt}.txt"
            else:
                summary_file = f"training_summary_{model_name}.txt"
                
            summary_path = os.path.join(training_summary_dir, summary_file)
            
            if os.path.exists(summary_path):
                with open(summary_path, "r") as f:
                    for line in f:
                        line = line.strip()
                        if line.startswith("Total Training Time:"):
                            try:
                                training_time = float(line.split(":", 1)[1].strip().replace(" seconds", ""))
                                experiment_data["Training Time"] = f"{training_time:.2f}"
                                experiment_data["Training_Time_raw"] = training_time
                            except ValueError:
                                experiment_data["Training Time"] = "N/A"
                                experiment_data["Training_Time_raw"] = None
                        elif line.startswith("Average RAM Usage:"):
                            try:
                                training_ram = float(line.split(":", 1)[1].strip().replace(" MB", ""))
                                experiment_data["Training RAM Usage"] = f"{training_ram:.2f}"
                                experiment_data["Training_RAM_Usage_raw"] = training_ram
                            except ValueError:
                                experiment_data["Training RAM Usage"] = "N/A"
                                experiment_data["Training_RAM_Usage_raw"] = None
                        elif line.startswith("Average GPU Memory Usage:"):
                            try:
                                training_gpu_memory = float(line.split(":", 1)[1].strip().replace(" MB", ""))
                                experiment_data["Training GPU Memory Usage"] = f"{training_gpu_memory:.2f}"
                                experiment_data["Training_GPU_Memory_raw"] = training_gpu_memory
                            except ValueError:
                                experiment_data["Training GPU Memory Usage"] = "N/A"
                                experiment_data["Training_GPU_Memory_raw"] = None
            else:
                print(f"Warning: Training summary file not found: {summary_path}")
                experiment_data["Training Time"] = "N/A"
                experiment_data["Training RAM Usage"] = "N/A"
                experiment_data["Training GPU Memory Usage"] = "N/A"
                experiment_data["Training_Time_raw"] = None
                experiment_data["Training_RAM_Usage_raw"] = None
                experiment_data["Training_GPU_Memory_raw"] = None
            
            # Append the collected data for this experiment to the list
            data.append(experiment_data)
    
    # Convert the list of dictionaries into a DataFrame
    df = pd.DataFrame(data)
    
    # Ensure all required columns are present and in the right order
    for col in required_columns:
        if col not in df.columns:
            df[col] = "N/A"
    
    # Save the full dataframe
    df[required_columns].to_csv("results/performance_metrics_full.csv", index=False)
    
    # Generate a LaTeX table for all experiments
    generate_latex_table(df[required_columns], "results/performance_metrics_full.tex", "full_performance")
    
    # Create best models performance table based on evaluation metrics
    if best_eval_models is not None and not best_eval_models.empty:
        best_perf_models = pd.DataFrame()

        print(f"Looking for best performance metrics for {len(best_eval_models)} models")
        print("Available performance metrics architectures:", df['Architecture'].unique())
        
        for _, best_row in best_eval_models.iterrows():
            # Find matching performance metrics
            mask = (df['Architecture'] == best_row['Architecture']) & \
                   (df['Loss Function'] == best_row['Loss Function']) & \
                   (df['Optimizer'] == best_row['Optimizer'])
            
            matching_rows = df[mask]
            if not matching_rows.empty:
                best_perf_models = pd.concat([best_perf_models, matching_rows])
        
        # Sort by architecture for consistent ordering
        best_perf_models = best_perf_models.sort_values('Architecture').reset_index(drop=True)
        
        # Create version with reduced columns
        best_perf_reduced = best_perf_models.copy()
        best_reduced_columns = [col for col in required_columns if col not in ["Loss Function", "Optimizer"]]
        
        # Save both dataframes
        best_perf_models[required_columns].to_csv("results/performance_metrics_best_full.csv", index=False)
        best_perf_reduced[best_reduced_columns].to_csv("results/performance_metrics_best.csv", index=False)
        
        # Generate LaTeX tables
        generate_latex_table(best_perf_models[required_columns], 
                            "results/performance_metrics_best_full.tex", 
                            "best_performance_full",
                            raw_df=best_perf_models)
        
        generate_latex_table(best_perf_reduced[best_reduced_columns], 
                            "results/performance_metrics_best.tex", 
                            "best_performance", 
                            include_loss_optimizer=False,
                            raw_df=best_perf_reduced)
    
    print("Performance metrics tables generated successfully.")
    return df

def format_best_values(df, raw_df, metric_col, raw_col, best_better="lower"):
    """Format a column to bold the best value and underline the second best"""
    formatted_values = []
    
    # For perf metrics, lower is better. For eval metrics, higher is better (except for MSE, MAE, RMSE)
    if best_better == "lower":
        # For metrics where lower is better (MSE, MAE, RMSE, training/prediction times, memory usage)
        raw_values = raw_df[raw_col].dropna().values
        if len(raw_values) < 2:  # Need at least 2 values to have a best and second best
            return df[metric_col].tolist()
        
        best_idx = np.argmin(raw_values)
        # Make a copy without the best value to find second best
        tmp_values = np.delete(raw_values, best_idx)
        second_best_idx = np.argmin(tmp_values)
        
        # Need to convert back to original index since we removed an element
        orig_indices = list(range(len(raw_values)))
        orig_indices.pop(best_idx)
        second_best_idx = orig_indices[second_best_idx]
        
        # Format values
        for i, val in enumerate(df[metric_col]):
            if i == best_idx:
                formatted_values.append(f"\\textbf{{{val}}}")
            elif i == second_best_idx:
                formatted_values.append(f"\\underline{{{val}}}")
            else:
                formatted_values.append(val)
    else:
        # For metrics where higher is better (R², EVS, R² ≥ 80%)
        raw_values = raw_df[raw_col].dropna().values
        if len(raw_values) < 2:  # Need at least 2 values to have a best and second best
            return df[metric_col].tolist()
        
        best_idx = np.argmax(raw_values)
        # Make a copy without the best value to find second best
        tmp_values = np.delete(raw_values, best_idx)
        second_best_idx = np.argmax(tmp_values)
        
        # Need to convert back to original index since we removed an element
        orig_indices = list(range(len(raw_values)))
        orig_indices.pop(best_idx)
        second_best_idx = orig_indices[second_best_idx]
        
        # Format values
        for i, val in enumerate(df[metric_col]):
            if i == best_idx:
                formatted_values.append(f"\\textbf{{{val}}}")
            elif i == second_best_idx:
                formatted_values.append(f"\\underline{{{val}}}")
            else:
                formatted_values.append(val)
    
    return formatted_values

def generate_latex_table(df, filename, metrics_type="evaluation", include_loss_optimizer=True, raw_df=None):
    # Start the LaTeX table
    latex_content = []
    latex_content.append("\\begin{table}[htbp]")
    latex_content.append("\\centering")
    
    # Set appropriate caption based on metrics type
    if "full" in metrics_type:
        if "evaluation" in metrics_type:
            caption = "Full Evaluation Metrics for All Neural Network Models"
        else:
            caption = "Full Performance Metrics for All Neural Network Models"
    else:
        if "evaluation" in metrics_type:
            caption = "Evaluation Metrics for Best Neural Network Model Configurations"
        else:
            caption = "Performance Metrics for Best Neural Network Model Configurations"
    
    latex_content.append("\\caption{" + caption + "}")
    
    # Create column specification
    if "evaluation" in metrics_type:
        if include_loss_optimizer:
            latex_content.append("\\begin{tabular}{lllcccccc}")
            header = "\\textbf{Architecture} & \\textbf{Loss Function} & \\textbf{Optimizer} & \\textbf{MSE $\\pm$ SD} & \\textbf{MAE $\\pm$ SD} & \\textbf{R²} & \\textbf{EVS} & \\textbf{RMSE} & \\textbf{R² $\\geq$ 80\\%} \\\\"
        else:
            latex_content.append("\\begin{tabular}{lcccccc}")
            header = "\\textbf{Architecture} & \\textbf{MSE $\\pm$ SD} & \\textbf{MAE $\\pm$ SD} & \\textbf{R²} & \\textbf{EVS} & \\textbf{RMSE} & \\textbf{R² $\\geq$ 80\\%} \\\\"
    else:  # performance metrics
        if include_loss_optimizer:
            latex_content.append("\\begin{tabular}{lllccccccc}")
            header = "\\textbf{Architecture} & \\textbf{Loss Function} & \\textbf{Optimizer} & \\textbf{Training Time (s)} & \\textbf{Training RAM (MB)} & \\textbf{Training GPU (MB)} & \\textbf{Pred. Time (s)} & \\textbf{Testing GPU (MB)} & \\textbf{Testing RAM (MB)} \\\\"
        else:
            latex_content.append("\\begin{tabular}{lcccccc}")
            header = "\\textbf{Architecture} & \\textbf{Training Time (s)} & \\textbf{Training RAM (MB)} & \\textbf{Training GPU (MB)} & \\textbf{Pred. Time (s)} & \\textbf{Testing GPU (MB)} & \\textbf{Testing RAM (MB)} \\\\"
    
    latex_content.append("\\toprule")
    latex_content.append(header)
    latex_content.append("\\midrule")
    
    # Group rows if full table, don't group for best models table
    if "full" in metrics_type:
        # Group by architecture, loss function and optimizer if including them, else just by architecture
        if include_loss_optimizer:
            grouped = df.groupby(['Architecture', 'Loss Function', 'Optimizer'])
        else:
            grouped = df.groupby(['Architecture'])
        
        # Process each group and add to table
        last_arch = None
        
        for group_key, group in grouped:
            # Add a line between different architectures
            if include_loss_optimizer:
                arch = group_key[0]
            else:
                arch = group_key
                
            if last_arch is not None and last_arch != arch:
                latex_content.append("\\midrule")
            
            row = group.iloc[0]  # Get the first row of this group
            
            # Format the row data
            if "evaluation" in metrics_type:
                if include_loss_optimizer:
                    row_data = [
                        arch,
                        group_key[1] if include_loss_optimizer else "",  # Loss Function
                        group_key[2] if include_loss_optimizer else "",  # Optimizer
                        row["MSE ± SD"],
                        row["MAE ± SD"],
                        row["R²"],
                        row["EVS"],
                        row["RMSE"],
                        row["R² >= 80%"]
                    ]
                else:
                    row_data = [
                        arch,
                        row["MSE ± SD"],
                        row["MAE ± SD"],
                        row["R²"],
                        row["EVS"],
                        row["RMSE"],
                        row["R² >= 80%"]
                    ]
            else:  # performance metrics
                if include_loss_optimizer:
                    row_data = [
                        arch,
                        group_key[1] if include_loss_optimizer else "",  # Loss Function
                        group_key[2] if include_loss_optimizer else "",  # Optimizer
                        row["Training Time"],
                        row["Training RAM Usage"],
                        row["Training GPU Memory Usage"],
                        row["Prediction Time per Sample"],
                        row["Testing GPU Memory Usage"],
                        row["Testing RAM Usage"]
                    ]
                else:
                    row_data = [
                        arch,
                        row["Training Time"],
                        row["Training RAM Usage"],
                        row["Training GPU Memory Usage"],
                        row["Prediction Time per Sample"],
                        row["Testing GPU Memory Usage"],
                        row["Testing RAM Usage"]
                    ]
            
            # Replace any NaN values with "N/A"
            row_data = [str(val) if val is not None and str(val) != "nan" else "N/A" for val in row_data]
            
            # Add the row to the table
            latex_content.append(" & ".join(row_data) + " \\\\")
            
            last_arch = arch
    else:
        # For best models table, apply formatting for best and second best values
        if raw_df is not None:
            # For evaluation metrics
            if "evaluation" in metrics_type:
                # Format metrics where lower is better
                mse_formatted = format_best_values(df, raw_df, "MSE ± SD", "MSE_raw", "lower")
                mae_formatted = format_best_values(df, raw_df, "MAE ± SD", "MAE_raw", "lower")
                rmse_formatted = format_best_values(df, raw_df, "RMSE", "RMSE_raw", "lower")
                
                # Format metrics where higher is better
                r2_formatted = format_best_values(df, raw_df, "R²", "R²_raw", "higher")
                evs_formatted = format_best_values(df, raw_df, "EVS", "EVS_raw", "higher")
                r2_pct_formatted = format_best_values(df, raw_df, "R² >= 80%", "R²_percentage_raw", "higher")
                
                # Add rows to the table
                for i, row in df.iterrows():
                    arch = row["Architecture"]
                    
                    if include_loss_optimizer:
                        row_data = [
                            arch,
                            row["Loss Function"],
                            row["Optimizer"],
                            mse_formatted[i],
                            mae_formatted[i],
                            r2_formatted[i],
                            evs_formatted[i],
                            rmse_formatted[i],
                            r2_pct_formatted[i]
                        ]
                    else:
                        row_data = [
                            arch,
                            mse_formatted[i],
                            mae_formatted[i],
                            r2_formatted[i],
                            evs_formatted[i],
                            rmse_formatted[i],
                            r2_pct_formatted[i]
                        ]
                    
                    latex_content.append(" & ".join(row_data) + " \\\\")
            else:  # performance metrics
                # Format all metrics (lower is better for performance metrics)
                training_time_formatted = format_best_values(df, raw_df, "Training Time", "Training_Time_raw", "lower")
                training_ram_formatted = format_best_values(df, raw_df, "Training RAM Usage", "Training_RAM_Usage_raw", "lower")
                training_gpu_formatted = format_best_values(df, raw_df, "Training GPU Memory Usage", "Training_GPU_Memory_raw", "lower")
                pred_time_formatted = format_best_values(df, raw_df, "Prediction Time per Sample", "Prediction_Time_raw", "lower")
                testing_gpu_formatted = format_best_values(df, raw_df, "Testing GPU Memory Usage", "Testing_GPU_Memory_raw", "lower")
                testing_ram_formatted = format_best_values(df, raw_df, "Testing RAM Usage", "Testing_RAM_Usage_raw", "lower")
                
                # Add rows to the table
                for i, row in df.iterrows():
                    arch = row["Architecture"]
                    
                    if include_loss_optimizer:
                        row_data = [
                            arch,
                            row["Loss Function"],
                            row["Optimizer"],
                            training_time_formatted[i],
                            training_ram_formatted[i],
                            training_gpu_formatted[i],
                            pred_time_formatted[i],
                            testing_gpu_formatted[i],
                            testing_ram_formatted[i]
                        ]
                    else:
                        row_data = [
                            arch,
                            training_time_formatted[i],
                            training_ram_formatted[i],
                            training_gpu_formatted[i],
                            pred_time_formatted[i],
                            testing_gpu_formatted[i],
                            testing_ram_formatted[i]
                        ]
                    
                    latex_content.append(" & ".join(row_data) + " \\\\")
        else:
            # If raw_df not provided, just iterate through each row without highlighting
            for _, row in df.iterrows():
                arch = row["Architecture"]
                
                # Format the row data
                if "evaluation" in metrics_type:
                    if include_loss_optimizer:
                        row_data = [
                            arch,
                            row["Loss Function"],
                            row["Optimizer"],
                            row["MSE ± SD"],
                            row["MAE ± SD"],
                            row["R²"],
                            row["EVS"],
                            row["RMSE"],
                            row["R² >= 80%"]
                        ]
                    else:
                        row_data = [
                            arch,
                            row["MSE ± SD"],
                            row["MAE ± SD"],
                            row["R²"],
                            row["EVS"],
                            row["RMSE"],
                            row["R² >= 80%"]
                        ]
                else:  # performance metrics
                    if include_loss_optimizer:
                        row_data = [
                            arch,
                            row["Loss Function"],
                            row["Optimizer"],
                            row["Training Time"],
                            row["Training RAM Usage"],
                            row["Training GPU Memory Usage"],
                            row["Prediction Time per Sample"],
                            row["Testing GPU Memory Usage"],
                            row["Testing RAM Usage"]
                        ]
                    else:
                        row_data = [
                            arch,
                            row["Training Time"],
                            row["Training RAM Usage"],
                            row["Training GPU Memory Usage"],
                            row["Prediction Time per Sample"],
                            row["Testing GPU Memory Usage"],
                            row["Testing RAM Usage"]
                        ]
                
                # Replace any NaN values with "N/A"
                row_data = [str(val) if val is not None and str(val) != "nan" else "N/A" for val in row_data]
                
                # Add the row to the table
                latex_content.append(" & ".join(row_data) + " \\\\")
    
    # Finish the LaTeX table
    latex_content.append("\\bottomrule")
    latex_content.append("\\end{tabular}")
    latex_content.append("\\label{tab:" + metrics_type.replace(" ", "_") + "}")
    latex_content.append("\\end{table}")
    
    # Write to file
    with open(filename, "w") as f:
        f.write("\n".join(latex_content))

def main():
    args = parse_args()
    
    # Track the best evaluation models to use for performance metrics
    best_eval_models = None
    
    if args.eval_metrics or not (args.eval_metrics or args.perf_metrics):
        _, best_eval_models = generate_eval_metrics()
    
    if args.perf_metrics or not (args.eval_metrics or args.perf_metrics):
        generate_perf_metrics(best_eval_models)
    
    if not (args.eval_metrics or args.perf_metrics):
        print("No arguments provided, generating both evaluation and performance metrics tables.")

if __name__ == "__main__":
    main()