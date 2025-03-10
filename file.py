def generate_shock_experiment_csv(filename="shock_experiment.csv"):
    """
    Creates a CSV file in the current directory with
    synthetic shock wave experimental data.
    """
    csv_content = """x_exp,density_exp
0.0000,1.0123
0.0500,1.0031
0.1000,0.9952
0.1500,0.9836
0.2000,0.9720
0.2500,0.9675
0.3000,0.9569
0.3200,0.9458
0.3500,0.4292
0.3600,0.2801
0.3700,0.1954
0.4000,0.1602
0.5000,0.1529
0.7000,0.1401
0.9000,0.1267
1.0000,0.1298
"""

    with open(filename, "w") as f:
        f.write(csv_content)
    print(f"'{filename}' has been generated with synthetic shock experiment data.")

# Example usage:
generate_shock_experiment_csv()
