# File: optuna_tuning/compare_studies.py
import os
import optuna
import pandas as pd
import plotly.express as px
from optuna.visualization import plot_optimization_history, plot_param_importances, plot_parallel_coordinate
import scipy.stats as stats

# Configuration
STUDY_NAMES = {
    "pfo": "Polar Fox Optimization",
    "ga": "Genetic Algorithm",
    "pso": "Particle Swarm Optimization",
    "gwo": "Grey Wolf Optimizer",
    "aco": "Ant Colony Optimization",
    "bat": "Bat Algorithm",
    "cs": "Cuckoo Search",
    "de": "Differential Evolution",
    "fa": "Firefly Algorithm",
    "hs": "Harmony Search",
    "ica": "Imperialist Competitive Algorithm",
    "sa": "Simulated Annealing",
    "woa": "Whale Optimization Algorithm"
}

def load_studies(storage="sqlite:///optuna.db"):
    studies = {}
    for short_name, full_name in STUDY_NAMES.items():
        try:
            study = optuna.load_study(
                study_name=f"{short_name}_study",
                storage=storage
            )
            studies[short_name] = {
                "study": study,
                "name": full_name
            }
        except KeyError:
            print(f"Warning: Study {short_name} not found in database")
    return studies

def generate_comparison_report(studies):
    # 1. Optimization History Comparison
    fig_history = plot_optimization_history(
        [s["study"] for s in studies.values()],
        target_names=[s["name"] for s in studies.values()]
    )
    fig_history.update_layout(title="Optimization History Comparison")
    fig_history.write_html("comparison/optimization_history.html")
    
    # 2. Hyperparameter Importance
    for algo, data in studies.items():
        fig = plot_param_importances(data["study"])
        fig.update_layout(title=f"Hyperparameter Importance: {data['name']}")
        fig.write_html(f"comparison/param_importance_{algo}.html")

    # 3. Best Trial Metrics Comparison
    metrics = []
    for algo, data in studies.items():
        trial = data["study"].best_trial
        metrics.append({
            "Algorithm": data["name"],
            "Best Fitness": trial.value,
            "Average SINR (dB)": trial.user_attrs["average_sinr"],
            "Average Throughput (Mbps)": trial.user_attrs["average_throughput"],
            "Fairness Index": trial.user_attrs["fairness"],
            "Load Variance": trial.user_attrs["load_variance"],
            "Execution Time (s)": trial.user_attrs["execution_time"]
        })
    
    df = pd.DataFrame(metrics).sort_values("Best Fitness", ascending=False)
    fig = px.bar(df, x="Algorithm", y="Best Fitness", color="Algorithm",
                 title="Algorithm Performance Comparison")
    fig.write_html("comparison/performance_bars.html")
    
    # 4. Statistical Significance Test
    rewards = {data["name"]: [t.value for t in data["study"].trials] 
               for algo, data in studies.items()}
    h_stat, p_val = stats.kruskal(*rewards.values())
    
    # 5. Parallel Coordinate Plot
    all_trials = []
    for algo, data in studies.items():
        for trial in data["study"].trials:
            params = trial.params.copy()
            params["Algorithm"] = data["name"]
            params["Fitness"] = trial.value
            all_trials.append(params)
    
    df_params = pd.DataFrame(all_trials)
    fig_parallel = plot_parallel_coordinate(
        df_params, 
        params=list(df_params.columns[:-2]),  # Exclude Algorithm and Fitness
        target_name="Fitness"
    )
    fig_parallel.write_html("comparison/parallel_coordinate.html")

    # Save results to CSV
    df.to_csv("comparison/algorithm_comparison.csv", index=False)
    
    # Print Statistical Results
    print(f"\nStatistical Test Results (Kruskal-Wallis):")
    print(f"H-statistic: {h_stat:.2f}, p-value: {p_val:.4f}")
    if p_val < 0.05:
        print("Significant difference exists between algorithms (p < 0.05)")
    else:
        print("No significant difference detected between algorithms")

    return df

    # Add this to generate_comparison_report() function

    # 6. Metric Progression Over Trials
    metric_names = ["average_sinr", "average_throughput", "fairness", "load_variance"]

    for metric in metric_names:
        fig = px.line(title=f"{metric.replace('_', ' ').title()} Progression")
        
        for algo, data in studies.items():
            # Get all trials sorted by number
            trials = sorted(data["study"].trials, key=lambda x: x.number)
            metric_values = [t.user_attrs[metric] for t in trials]
            
            fig.add_scatter(
                x=list(range(len(trials))),
                y=metric_values,
                mode='lines',
                name=data["name"],
                line=dict(width=2)
            )
        
        fig.update_layout(
            xaxis_title="Trial Number",
            yaxis_title=metric.replace('_', ' ').title(),
            showlegend=True
        )
        fig.write_html(f"comparison/{metric}_progression.html")

    # 7. Multi-Metric Parallel Plot
    fig = px.parallel_coordinates(
        df,
        color="Best Fitness",
        dimensions=["Best Fitness", "Average SINR (dB)", "Average Throughput (Mbps)", 
                "Fairness Index", "Load Variance", "Execution Time (s)"],
        color_continuous_scale=px.colors.sequential.Viridis,
        title="Multi-Metric Relationship"
    )
    fig.write_html("comparison/multi_metric_parallel.html")

if __name__ == "__main__":
    # Create output directory
    os.makedirs("comparison", exist_ok=True)
    
    # Load all studies
    studies = load_studies()
    
    if not studies:
        print("No studies found in database!")
        exit(1)
        
    # Generate comparison report
    print("\n=== Generating Comparative Analysis ===")
    results_df = generate_comparison_report(studies)
    
    # Print summary table
    print("\n=== Algorithm Performance Summary ===")
    print(results_df.to_string(index=False))
    
    