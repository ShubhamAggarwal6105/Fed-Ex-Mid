from rectpack import newPacker
from itertools import permutations
import os
from multiprocessing import Pool, freeze_support

class ULDContainer:
    def __init__(self, uld_id, height, width, length, weight):
        self.uld_id = uld_id
        self.height, self.width, self.length = height, width, length
        self.weight = weight

    @property
    def volume(self):
        return self.height * self.width * self.length

class Package:
    def __init__(self, package_id, length, width, height, weight, package_type, cost):
        self.package_id = package_id
        self.height, self.width, self.length = height, width, length
        self.weight = weight
        self.type = package_type
        self.cost = cost

    @property
    def volume(self):
        return self.height * self.width * self.length

def similar_height_sorting(packages):
    n = len(packages)
    max_height = max(max(p.height, p.length, p.width) for p in packages)
    index_list = [set() for i in range(max_height+1)]
    for idx in range(n):
        index_list[packages[idx].height].add(idx)
        index_list[packages[idx].length].add(idx)
        index_list[packages[idx].width].add(idx)
    index_list = list(enumerate(index_list))

    sorted_packages = []
    index_list.sort(key = lambda a: len(a[1]), reverse=True)
    while len(sorted_packages) < n:
        height = index_list[0][0]
        to_delete = []
        for idx in index_list[0][1]:
            p = packages[idx]
            dimensions = [p.height, p.length, p.width]
            dimensions.remove(height)
            sorted_packages.append(Package(p.package_id, dimensions[0], dimensions[1], height, p.weight, p.type, p.cost))
            to_delete.append(idx)
        index_list.sort()
        for idx in to_delete:
            p = packages[idx]
            dimensions = set([p.height, p.length, p.width])
            for height in dimensions:
                index_list[height][1].remove(idx)
        index_list.sort(key = lambda a: len(a[1]), reverse=True)
    return sorted(sorted_packages, key=lambda p: (p.height, p.volume))

def least_height_sorting(packages):
    n = len(packages)
    sorted_packages = []
    for p in packages:
        a,b,c = sorted([p.length, p.width, p.height])
        sorted_packages.append(Package(p.package_id, c, b, a, p.weight, p.type, p.cost))
    return sorted(sorted_packages, key=lambda p: (p.height, p.volume))

class PackageLoader:
    def __init__(self, uld_containers, priority_packages, economy_packages):
        # self.uld_containers = sorted(uld_containers, key=lambda u: u.volume, reverse=True)
        self.uld_containers = uld_containers
        self.priority_packages = sorted(priority_packages, key=lambda p: (p.height, p.volume))
        self.economy_packages = sorted(economy_packages, key=lambda p: (p.height, p.volume))
        self.reached_heights = [0] * len(self.uld_containers)
        self.heights = [[] for i in range(len(self.uld_containers))]
        self.packed = [[] for i in range(len(self.uld_containers))]
        self.coordinates = [[] for i in range(len(self.uld_containers))]

    def _load_single_run(self, packages):
        curr_uld = 0
        reached_height = self.reached_heights[curr_uld]
        packed_volume = 0
        packed_count = 0
        uld_changed = False

        while curr_uld < len(self.uld_containers) and packages:
            if uld_changed:
                self.reached_heights[curr_uld-1] = reached_height
                reached_height = self.reached_heights[curr_uld]
                uld_changed = False

            packer = newPacker()
            packer.add_bin(self.uld_containers[curr_uld].length, self.uld_containers[curr_uld].width)
            packed = []
            package_tried = 0

            curr = 0
            while curr < len(packages) and package_tried == len(packed):
                curr_height = packages[curr].height

                if reached_height + curr_height > self.uld_containers[curr_uld].height:
                    curr_uld += 1
                    uld_changed = True
                    break
                
                while curr < len(packages):
                    package = packages[curr]
                    if package.height == curr_height:
                        package_tried += 1
                        packer.add_rect(package.width, package.length, curr)
                        curr += 1
                    else:
                        break

                packer.pack()
                packed = packer.rect_list()

            for _, x, y, w, h, idx in packed:
                self.coordinates[curr_uld-int(uld_changed)].append([packages[idx], ((x,y,reached_height), (x+w,y+h,reached_height+packages[idx].height))])

            self.heights[curr_uld-int(uld_changed)].append(max([0]+[packages[idx].height for _,x,y,w,h,idx in packed]))
            reached_height += max([0]+[packages[idx].height for _,x,y,w,h,idx in packed])

            ## 2 Pointer Optimization for deleting packed packages.
            n = len(packages)

            len_to_delete = len(packed)
            to_delete = [-1]+sorted([idx for _,x,y,w,h,idx in packed])+[n]

            curr = 0
            for i in range(1,len_to_delete+2):
                for idx in range(to_delete[i-1]+1, to_delete[i]):
                    packages[curr] = packages[idx]
                    curr+=1
                if i==len_to_delete+1: break
                self.packed[curr_uld-int(uld_changed)].append(packages[to_delete[i]])
                packed_count += 1
                packed_volume += packages[to_delete[i]].height * packages[to_delete[i]].width * packages[to_delete[i]].length

            for i in range(len_to_delete): packages.pop()
        
        self.reached_heights[curr_uld-int(uld_changed)] = reached_height
        return packed_volume

    def load_priority_packages(self):
        return self._load_single_run(self.priority_packages)

    def load_economy_packages(self):
        return self._load_single_run(self.economy_packages)

    def fix_weight_constraint(self):
        removed = []
        for curr in range(len(self.uld_containers)):
            packages = self.packed[curr]
            max_weight = self.uld_containers[curr].weight
            curr_weight = sum(pkg.weight for pkg in packages)
            if curr_weight <= max_weight: continue
            packages.sort(key = lambda p: (int(p.type=='Economy'), p.weight, -p.cost))
            while curr_weight > max_weight:
                p = packages.pop()
                removed.append(p)
                curr_weight -= p.weight
            self.economy_packages = packages
            self.load_economy_packages()
        return removed
            
    def push_packages(self):
        for curr in range(len(self.uld_containers)):
            ## Find correct orientation of original ULD
            coordinates = self.coordinates[curr]
            curr_uld = self.uld_containers[curr]
            for og_uld in uld_containers_file_input:
                if og_uld.uld_id == curr_uld.uld_id:
                    break
            
            if og_uld.length == curr_uld.length:
                pass
            elif og_uld.length == curr_uld.width:
                curr_uld.length, curr_uld.width = curr_uld.width, curr_uld.length
                coordinates = [(pkg, ((c[0][1],c[0][0],c[0][2]),(c[1][1],c[1][0],c[1][2]))) for pkg,c in coordinates]
            else:
                curr_uld.length, curr_uld.height = curr_uld.height, curr_uld.length
                coordinates = [(pkg, ((c[0][2],c[0][1],c[0][0]),(c[1][2],c[1][1],c[1][0]))) for pkg,c in coordinates]

            if og_uld.width == curr_uld.width:
                pass
            else:
                curr_uld.width, curr_uld.height = curr_uld.height, curr_uld.width
                coordinates = [(pkg, ((c[0][0],c[0][2],c[0][1]),(c[1][0],c[1][2],c[1][1]))) for pkg,c in coordinates]

            self.coordinates[curr] = coordinates
            self.uld_containers[curr] = curr_uld
            new_coordinates = []
            coordinates.sort(key = lambda a: min(a[1][0][2], a[1][1][2]))
            grid = [[[[0] for i in range(self.uld_containers[curr].height)] for j in range(self.uld_containers[curr].width)] for k in range(self.uld_containers[curr].length)]
            for pkg, coordinate in coordinates:
                l1,b1,h1 = coordinate[0]
                l2,b2,h2 = coordinate[1]
                l1,l2 = sorted([l1,l2])
                b1,b2 = sorted([b1,b2])
                h1,h2 = sorted([h1,h2])
                while h1 > 0:
                    flag = True
                    for l in range(l1,l2):
                        for b in range(b1,b2):
                            if grid[l][b][h1-1]==1:
                                flag = False
                                break
                    if flag:
                        h1 -= 1
                        h2 -= 1
                    else:
                        break
                for l in range(l1,l2):
                    for b in range(b1,b2):
                        for h in range(h1,h2):
                            grid[l][b][h] = 1
                new_coordinates.append([pkg, ((l1,b1,h1), (l2,b2,h2))])
            self.coordinates[curr] = new_coordinates

    def calculate_voids(self, packed_volume):
        total_uld_volume = sum(uld.volume for uld in self.uld_containers)
        return total_uld_volume - packed_volume

def read_input(file_path):
    with open(file_path, "r") as f:
        k = int(f.readline().strip())
        num_uld = int(f.readline().strip())

        uld_containers = []
        for _ in range(num_uld):
            uld_id, length, width, height, weight = f.readline().strip().split(",")
            uld_containers.append(ULDContainer(uld_id, int(length), int(width), int(height), int(weight)))

        num_packages = int(f.readline().strip())
        priority_packages, economy_packages = [], []

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
            package = Package(
                package_id, int(length), int(width), int(height), int(weight), package_type, INF if package_type=='Priority' else int(cost)
            )
            if package.type == "Economy":
                economy_packages.append(package)
            else:
                priority_packages.append(package)

        return k, uld_containers, priority_packages, economy_packages
    
def greedy_packing(l, r):
    min_cost = INF
    best_loaders = None
    costs = [0]*(r-l)
    for check in range(l,r):
        print("Running task", check)

        economy_packages_input = sorted(economy_packages_file_input, key=lambda p: p.cost / p.volume, reverse=True)[:check]

        priority_packages_input = similar_height_sorting(priority_packages_file_input)
        economy_packages_input = similar_height_sorting(economy_packages_input)

        total_trying = len(economy_packages_input) + len(priority_packages_input)

        m_cost = INF
        for ulds in uld_permutations:
            priority_packages = priority_packages_input
            economy_packages = economy_packages_input
            total_priority_split = 0
            total_priority_volume = 0
            total_economy_volume = 0
            total_cost = 0
            loaders = [None]*len(ulds)
            for i in range(len(ulds)):
                loader1 = PackageLoader([ULDContainer(ulds[i].uld_id, ulds[i].length, ulds[i].width, ulds[i].height, ulds[i].weight)], priority_packages, economy_packages)
                priority_volume1 = loader1.load_priority_packages()
                economy_volume1 = loader1.load_economy_packages()
                packed_count_1 = total_trying - len(loader1.priority_packages) - len(loader1.economy_packages)

                loader2 = PackageLoader([ULDContainer(ulds[i].uld_id, ulds[i].height, ulds[i].width, ulds[i].length, ulds[i].weight)], priority_packages, economy_packages)
                priority_volume2 = loader2.load_priority_packages()
                economy_volume2 = loader2.load_economy_packages()
                packed_count_2 = total_trying - len(loader2.priority_packages) - len(loader2.economy_packages)

                loader3 = PackageLoader([ULDContainer(ulds[i].uld_id, ulds[i].width, ulds[i].length, ulds[i].height, ulds[i].weight)], priority_packages, economy_packages)
                priority_volume3 = loader3.load_priority_packages()
                economy_volume3 = loader3.load_economy_packages()
                packed_count_3 = total_trying - len(loader3.priority_packages) - len(loader3.economy_packages)

                if packed_count_1 >= max(packed_count_2, packed_count_3):
                    loaders[i] = loader1
                    priority_volume = priority_volume1
                    economy_volume = economy_volume1

                elif packed_count_2 >= max(packed_count_1, packed_count_3):
                    loaders[i] = loader2
                    priority_volume = priority_volume2
                    economy_volume = economy_volume2
                
                elif packed_count_3 >= max(packed_count_2, packed_count_1):
                    loaders[i] = loader3
                    priority_volume = priority_volume3
                    economy_volume = economy_volume3

                priority_packages = loaders[i].priority_packages
                economy_packages = loaders[i].economy_packages

                if priority_volume != 0: total_priority_split += 1
                total_priority_volume += priority_volume
                total_economy_volume += economy_volume


            economy_packages += economy_packages_file_input[check:]
            
            # Optimization 1
            for i in range(len(loaders)):
                loaders[i].economy_packages = least_height_sorting(economy_packages)
                loaders[i].load_economy_packages()
                economy_packages = loaders[i].economy_packages

            # Optimization 2
            economy_packages = least_height_sorting(economy_packages)
            economy_packages.sort(key = lambda p: (-p.height, -p.volume))

            for i in range(len(loaders)):
                while len(economy_packages) > 0:
                    loaders[i].economy_packages = []
                    packages = loaders[i].packed[0].copy()
                    packages.append(economy_packages[-1])
                    packages = similar_height_sorting(packages)
                    packages = sorted(packages, key=lambda p: (p.height, p.volume))
                    loader = PackageLoader(loaders[i].uld_containers, packages, [])
                    loader.load_priority_packages()
                    if len(loader.priority_packages) == 0:
                        loaders[i] = loader
                        total_economy_volume += economy_packages[-1].height*economy_packages[-1].width*economy_packages[-1].length
                        economy_packages.pop()
                    else:
                        break

            ## Fix Weight constraint and push packages
            for i in range(len(loaders)):
                removed = loaders[i].fix_weight_constraint()
                for remove in removed:
                    economy_packages.append(remove)

            for package in economy_packages:
                total_cost += package.cost
            total_cost += total_priority_split*k

            if total_cost < m_cost:
                best_loaders = loaders
                m_cost = total_cost

        costs[check-l] = m_cost
        min_cost = min(min_cost, m_cost)
    return min_cost, best_loaders, costs

INF = 10**10

file_path = "input.txt"
k, uld_containers_file_input, priority_packages_file_input, economy_packages_file_input = read_input(file_path)

uld_permutations = list(permutations(uld_containers_file_input))

## Deleting similar permutations of ULD
l = len(uld_permutations)
i = 0
while i < l:
    b1 = uld_permutations[i]
    a1 = [(b.length, b.width, b.height, b.weight) for b in b1]
    j = i+1
    while j < l:
        b2 = uld_permutations[j]
        a2 = [(b.length, b.width, b.height, b.weight) for b in b2]
        if a2==a1:
            del uld_permutations[j]
            l -= 1
        else:
            j += 1
        i+=1

## THREADING
if __name__ == "__main__":
    freeze_support()

    print("Initializing Algorithm")
    cost, loader, _ = greedy_packing(len(economy_packages_file_input), len(economy_packages_file_input)+1)
    print(_)

    end = sum(len(l.coordinates[0]) for l in loader) - len(priority_packages_file_input)
    start = max(end-50, 1)

    threads = end - start

    cpu_threads = os.cpu_count()-2
    print("Running on",cpu_threads,"threads")
    print(f"Checking from {start} to {end}")

    threads_per_worker = (threads + cpu_threads - 1) // cpu_threads
    divisions = [(i*threads_per_worker+start, (i+1)*threads_per_worker+start if i+1<cpu_threads else threads+start) for i in range(cpu_threads)]
    with Pool(cpu_threads) as pool:
        results = pool.starmap(greedy_packing, divisions)

    min_cost, best_loaders, _ = min(results, key=lambda x: x[0])
    costs = []
    for _, __, cost in results:
        costs += cost
    ## ----------

    best_loaders.sort(key = lambda l: l.uld_containers[0].uld_id)
    for loader in best_loaders:
        loader.push_packages()
        loader.coordinates[0].sort(key = lambda p: int(p[0].package_id[2:]))
        
    answer = ""
    total_packed_volume = 0
    max_packed_count = 0
    for loader in best_loaders:
        curr_uld = loader.uld_containers[0]
        answer += f"ULD {curr_uld.uld_id} : ({curr_uld.length}, {curr_uld.width}, {curr_uld.height})\n"
        for package, coordinates in loader.coordinates[0]:
            total_packed_volume += package.volume
            max_packed_count += 1
            answer += f"Packed {package.type} {package.package_id} : {coordinates[0]} - {coordinates[1]}\n"
        answer += "\n"

    total_volume = sum(uld.height*uld.length*uld.width for uld in uld_containers_file_input)

    stats = ""
    stats += f"Minimum Cost    : {min_cost}\n"
    stats += f"Packages packed : {max_packed_count}\n"
    stats += f"Volume Packed   : {total_packed_volume}/{total_volume}\n"
    stats += f"Packing Eff.    : {int(10000*total_packed_volume/total_volume)/100}%\n"
    stats += "\n"

    with open("answer.txt",'w') as f:
        f.write(answer)

    with open("stats.txt",'w') as f:
        f.write(stats)

    cost_string = ""
    for i in range(threads):
        cost_string += f"{i+start}:{costs[i]}\n"
    with open("costs.txt",'w') as f:
        f.write(cost_string)

    print(stats + answer)

    from plotter import plot_cost_vs_optimization
    plot_cost_vs_optimization(costs, start)