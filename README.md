
# FedEx 3D Packing Optimization

## Overview
This project addresses the complex 3D packing problem for loading packages into Unit Load Devices (ULDs) for flights. The solution prioritizes efficient utilization of ULDs, adherence to weight and volume constraints, and optimization of cost. The algorithm ensures priority packages are always packed while maintaining high packing efficiency and minimizing costs.

Key outputs of the program include:
1. **Packing Decisions**: Specifies which package is packed in which ULD, along with orientation and coordinates.
2. **Performance Metrics**: Reports costs, packing efficiency, and other relevant statistics.
3. **Visualization**: Generates a graph illustrating the relationship between cost optimization and volume optimization.

## Features
- **Greedy Algorithm with Heuristics**: Uses sorting, permutation, and multi-threading to achieve optimal packing and cost minimization.
- **Priority Handling**: Ensures all priority packages are packed, leaving economy packages optional.
- **Cost Optimization**: Balances cost and volume considerations to achieve minimum total cost.
- **Parallel Processing**: Utilizes multi-core processing for improved performance.
- **Detailed Outputs**:
  - `answer.txt`: Packing assignments and details.
  - `stats.txt`: Packing efficiency and cost metrics.
  - `data.png`: Cost vs. optimization graph.

## Installation
1. Clone this repository.
2. Ensure you have Python 3.8 or higher installed.
3. Install required dependencies:
   ```bash
   pip install matplotlib rectpack
   ```

## Usage
1. Prepare the input file (`input.txt`) containing ULD and package details. The format should be as follows:
   ```
   <cost of splitting priority packages>
   <number of ULDs>
   <ULD_ID>,<length>,<width>,<height>,<weight_limit>
   ...
   <number of packages>
   <package_ID>,<length>,<width>,<height>,<weight>,<type>,<cost>
   ...
   ```
2. Run the script:
   ```bash
   python FedEx.py
   ```
3. Outputs will be generated:
   - `answer.txt`: Packing details.
   - `stats.txt`: Packing statistics.
   - `data.png`: Graph of cost optimization vs. volume optimization.

## Outputs
### 1. Packing Assignments (`answer.txt`)
Lists the ULD each package is assigned to, with orientation and coordinates.

Example:
```
ULD ULD1 : (50, 40, 30)
Packed Priority P001 : (0, 0, 0) - (20, 15, 10)
Packed Economy E001 : (0, 0, 10) - (10, 10, 20)
...

ULD ULD2 : (60, 50, 40)
...
```

### 2. Packing Statistics (`stats.txt`)
Includes metrics such as:
- **Minimum Cost**: Total minimized cost.
- **Packages Packed**: Number of packages successfully packed.
- **Volume Packed**: Total packed volume compared to available volume.
- **Packing Efficiency**: Percentage of volume utilization.

Example:
```
Minimum Cost    : 1500
Packages Packed : 120
Volume Packed   : 72000/90000
Packing Eff.    : 80.0%
```

### 3. Cost vs. Optimization Graph (`data.png`)
Shows how cost changes with optimization parameters. The minimum cost is highlighted.

## How It Works
1. **Input Parsing**: Reads and parses input data for ULDs and packages.
2. **Package Sorting**:
   - Priority and economy packages sorted by dimensions and cost/volume ratio.
3. **Greedy Packing**:
   - Iterates through ULD permutations.
   - Packs priority packages first, followed by economy packages.
   - Fixes weight constraints and optimizes voids.
4. **Multi-threading**:
   - Divides the workload across available CPU threads for faster execution.
5. **Output Generation**:
   - Produces text reports and graphical visualization.

## Visualization Example
![Cost Optimization vs Volume Optimization](data.png)

## Acknowledgements
This project utilizes:
- **[RectPack](https://github.com/secnot/rectpack)** for 2D rectangle packing.
- **Matplotlib** for visualizing results.

## Contact
For any queries or contributions, feel free to reach out or submit an issue on GitHub.
