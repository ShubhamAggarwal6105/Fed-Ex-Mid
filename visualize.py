import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import random

ulds = []
with open("input.txt", "r") as file:
    k = file.readline()
    n_uld = file.readline()
    uld_details = [file.readline().split(",") for i in range(int(n_uld))]
    
    n_packs = file.readline()
    packs_details = [file.readline().split(",") for i in range(int(n_packs))]

# Define ULDs
for item in uld_details:
    ulds.append((item[0], int(item[1]), int(item[2]), int(item[3]), int(item[4])))   

# Define Packages
packages = []
with open("output.txt", "r") as file:
    r = file.readline()
    v = file.readlines()

def check_priority(val):
    return val=="Priority"

for i, item in enumerate(v):
    vals = item.split(',')
    packages.append((vals[0], vals[1], int(vals[2]), int(vals[3]), int(vals[4]), int(vals[5]), int(vals[6]), int(vals[7]), check_priority(packs_details[i][5])))
print(packages[0:5])

def random_color():
    return "#" + ''.join(random.choices('0123456789ABCDEF', k=6))

def draw_uld_with_packages(uld_id, length, width, height, packages_in_uld):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    # Set plot limits and labels
    ax.set_xlim([0, length])
    ax.set_ylim([0, width])
    ax.set_zlim([0, height])
    ax.set_xlabel('Length (cm)')
    ax.set_ylabel('Width (cm)')
    ax.set_zlabel('Height (cm)')
    ax.set_title(f'ULD {uld_id} with Packages')

    # Draw ULD
    uld_vertices = [
        (0, 0, 0), (length, 0, 0), (length, width, 0), (0, width, 0),
        (0, 0, height), (length, 0, height), (length, width, height), (0, width, height)
    ]
    edges = [
        (0, 1, 5, 4), (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7),  # Vertical edges
        (0, 1, 2, 3), (4, 5, 6, 7)  # Top and bottom faces
    ]
    for edge in edges:
        ax.add_collection3d(Poly3DCollection([[uld_vertices[i] for i in edge]], alpha=0.1, color='blue'))
    # Sort packages by bottom face (z1)
    packages_in_uld.sort(key=lambda p: p[4])
    # Draw Packages one by one
    for package in packages_in_uld:
        _, _, x1, y1, z1, x2, y2, z2, pri = package
        package_vertices = [
            (x1, y1, z1), (x2, y1, z1), (x2, y2, z1), (x1, y2, z1),
            (x1, y1, z2), (x2, y1, z2), (x2, y2, z2), (x1, y2, z2)
        ]
        color = random_color()
        for edge in edges:
            if pri == True:
                color = "black"
            ax.add_collection3d(Poly3DCollection([[package_vertices[i] for i in edge]], alpha=0.5, color=color))

        # Update plot to show current package
        plt.draw()
        plt.pause(0.1)  # Pause for 1 second to create animation effect
    plt.show()

# Group packages by ULD
uld_packages_map = {uld[0]: [] for uld in ulds}
for package in packages:
    if package[1] != "NONE":
        uld_packages_map[package[1]].append(package)

# Visualize each ULD with its packages
for uld in ulds:
    uld_id, length, width, height, _ = uld
    packages_in_uld = uld_packages_map[uld_id]
    draw_uld_with_packages(uld_id, length, width, height, packages_in_uld)
