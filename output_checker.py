## CHECKS IF THE OUTPUT FILE IS CORRECT
## Requirements: output.txt file with correct output format.

import numpy as np

INF = 10**10

print("READING INPUT...")
with open("input.txt", "r") as f:
    k = int(f.readline().strip())
    num_uld = int(f.readline().strip())

    uld_containers = {}
    weights = {}
    contains_priority = {}
    for _ in range(num_uld):
        uld_id, length, width, height, weight = f.readline().strip().split(",")
        uld_containers[uld_id] = np.zeros((int(length), int(width), int(height)), dtype=int)
        weights[uld_id] = int(weight)
        contains_priority[uld_id] = False

    num_packages = int(f.readline().strip())
    packages = {}
    for _ in range(num_packages):
        (
            package_id,
            length,
            width,
            height,
            weight,
            package_type,
            cost,
        ) = f.readline().strip().split(",")
        packages[package_id] = [
            int(length),
            int(width),
            int(height),
            int(weight),
            package_type,
            INF if package_type == "Priority" else int(cost),
        ]

print("READING OUTPUT...")
with open("output.txt") as f:
    data = f.readlines()

output_cost, output_packed, output_split = map(int, data[0].split(","))


def verify():
    cost = 0
    packed = 0

    for line in data[1:]:
        pid, uid, a, b, c, x, y, z = line.split(",")
        a, b, c, x, y, z = map(int, [a, b, c, x, y, z])

        if pid not in packages:
            print(f"ERROR: {pid} doesn't exist / packed twice.")
            return

        if uid == "NONE":
            if packages[pid][4] == "Priority":
                print(f"ERROR: Priority package {pid} is not packed.")
                return
            cost += packages[pid][5]
            del packages[pid]
            continue

        packed += 1

        if uid not in uld_containers:
            print(f"ERROR: ULD {uid} doesn't exist")
            return

        if packages[pid][4] == "Priority":
            contains_priority[uid] = True

        pkg = packages[pid]

        if sorted([x - a, y - b, z - c]) != sorted(pkg[:3]):
            print(f"ERROR: Dimensions of {pid} don't match packing")
            return

        # Check overlap using slicing
        container_slice = uld_containers[uid][a:x, b:y, c:z]
        if np.any(container_slice):
            print(f"ERROR: Package {pid} overlaps.")
            return

        # Mark the space as filled
        uld_containers[uid][a:x, b:y, c:z] = 1

        weights[uid] -= packages[pid][3]
        del packages[pid]

    split = sum(contains_priority.values())

    cost += split*k

    if cost != output_cost:
        print(f"ERROR: Costs don't match. Actual cost of given packing is {cost}")
        return

    for uld in weights:
        if weights[uld] < 0:
            print(f"ERROR: Weight in ULD {uld} exceeds by {-weights[uld]}")
            return

    if packed != output_packed:
        print(f"ERROR: Packed count doesn't match. Actual packed count of given packing is {packed}")
        return

    if split != output_split:
        print(f"ERROR: Splits don't match. Actual priority split of given packing is {split}")
        return

    print("SUCCESSFUL: Output is correct")


print("VERIFYING...")
verify()