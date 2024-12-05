import matplotlib.pyplot as plt

def plot_cost_vs_optimization(y, start, output_file="data.png"):
    """
    Plots the cost vs optimization graph with the minimum cost point.
    
    Parameters:
    - x: List of optimization values (x-axis)
    - y: List of cost values (y-axis)
    - output_file: Name of the file to save the plot (default is "data.png")
    """
    x = list(range(start,start+len(y)))

    # Create the plot
    plt.figure(figsize=(10, 6))  # Optional: Adjust the figure size for better visibility
    plt.plot(x, y, 'ro-', markersize=1, label='Cost vs Optimization')

    # Label the axes
    plt.xlabel('Cost Optimization ↑ / Volume Optimization ↓', fontsize=12)
    plt.ylabel('Cost', fontsize=12)

    # Add a title (optional)
    plt.title('Cost Optimization vs Volume Optimization', fontsize=14)

    # Identify the minimum cost and its corresponding optimization value
    min_cost = min(y)
    min_index = y.index(min_cost)
    min_opt = x[min_index]

    # Plot the minimum point
    plt.plot(min_opt, min_cost, 'bs', markersize=8, label='Minimum Cost')

    # Annotate the minimum point
    plt.annotate(
        f'Minimum Cost: {min_cost}',
        xy=(min_opt, min_cost),
        xytext=(min_opt, min_cost - 50),  # Adjust the text position
        horizontalalignment='center',
        fontweight='bold'
    )

    # Add grid for better readability (optional)
    plt.grid(True, linestyle='--', alpha=0.2)

    # Add a legend
    plt.legend()

    # Display the plot
    plt.tight_layout()  # Adjust the padding

    # Save the plot to a file
    plt.savefig(output_file)
    plt.show()
