import numpy as np
import os
import random
from operator import itemgetter
from itertools import combinations
from place_db_proto import get_node_info
from place_db_proto import get_net_info
import sys
import pickle
sys.path.append('ariane')
from ariane.read_info import get_netlist_info_dict
# Macro dict (macro id -> name, x, y)
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def read_node_file(fopen, benchmark):
    node_info = {}
    node_info_raw_id_name ={}
    node_cnt = 0
    for line in fopen.readlines():
        if not line.startswith("\t"):
            continue
        line = line.strip().split()
        if line[-1] != "terminal":
            continue
        node_name = line[0]
        x = int(line[1])
        y = int(line[2])
        node_info[node_name] = {"id": node_cnt, "x": x , "y": y }
        node_info_raw_id_name[node_cnt] = node_name
        node_cnt += 1
    print("len node_info", len(node_info))
    return node_info, node_info_raw_id_name


def read_net_file(fopen, node_info):
    net_info = {}
    net_name = None
    net_cnt = 0
    for line in fopen.readlines():
        if not line.startswith("\t") and not line.startswith("NetDegree"):
            continue
        line = line.strip().split()
        if line[0] == "NetDegree":
            net_name = line[-1]
        else:
            node_name = line[0]
            if node_name in node_info:
                if not net_name in net_info:
                    net_info[net_name] = {}
                    net_info[net_name]["nodes"] = {}
                    net_info[net_name]["ports"] = {}
                if not node_name in net_info[net_name]["nodes"]:
                    x_offset = float(line[-2])
                    y_offset = float(line[-1])
                    net_info[net_name]["nodes"][node_name] = {}
                    net_info[net_name]["nodes"][node_name] = {"x_offset": x_offset, "y_offset": y_offset}
    for net_name in list(net_info.keys()):
        if len(net_info[net_name]["nodes"]) <= 1:
            net_info.pop(net_name)
    for net_name in net_info:
        net_info[net_name]['id'] = net_cnt
        net_cnt += 1
    print("adjust net size = {}".format(len(net_info)))
    return net_info


def get_comp_hpwl_dict(node_info, net_info):
    comp_hpwl_dict = {}
    for net_name in net_info:
        max_idx = 0
        for node_name in net_info[net_name]["nodes"]:
            max_idx = max(max_idx, node_info[node_name]["id"])
        if not max_idx in comp_hpwl_dict:
            comp_hpwl_dict[max_idx] = []
        comp_hpwl_dict[max_idx].append(net_name)
    return comp_hpwl_dict


def get_node_to_net_dict(node_info, net_info):
    node_to_net_dict = {}
    for node_name in node_info:
        node_to_net_dict[node_name] = set()
    for net_name in net_info:
        for node_name in net_info[net_name]["nodes"]:
            node_to_net_dict[node_name].add(net_name)
    return node_to_net_dict


def get_port_to_net_dict(port_info, net_info):
    port_to_net_dict = {}
    for port_name in port_info:
        port_to_net_dict[port_name] = set()
    for net_name in net_info:
        for port_name in net_info[net_name]["ports"]:
            port_to_net_dict[port_name].add(net_name)
    return port_to_net_dict

def read_pl_file(fopen, node_info):
    max_height = 0
    max_width = 0
    for line in fopen.readlines():
        if not line.startswith('o'):
            continue
        line = line.strip().split()
        node_name = line[0]
        if not node_name in node_info:
            continue
        place_x = int(line[1])
        place_y = int(line[2])
        max_height = max(max_height, node_info[node_name]["x"] + place_x)
        max_width = max(max_width, node_info[node_name]["y"] + place_y)
        node_info[node_name]["raw_x"] = place_x
        node_info[node_name]["raw_y"] = place_y
    return max(max_height, max_width), max(max_height, max_width)


def get_node_id_to_name(node_info, node_to_net_dict):
    node_name_and_num = []
    for node_name in node_info:
        node_name_and_num.append((node_name, len(node_to_net_dict[node_name])))
    node_name_and_num = sorted(node_name_and_num, key=itemgetter(1), reverse = True)
    print("node_name_and_num", node_name_and_num)
    node_id_to_name = [node_name for node_name, _ in node_name_and_num]
    for i, node_name in enumerate(node_id_to_name):
        node_info[node_name]["id"] = i
    return node_id_to_name


def get_node_id_to_name_topology(node_info, node_to_net_dict, net_info, benchmark):
    node_id_to_name = []
    adjacency = {}
    for net_name in net_info:
        for node_name_1, node_name_2 in list(combinations(net_info[net_name]['nodes'],2)):
            if node_name_1 not in adjacency:
                adjacency[node_name_1] = set()
            if node_name_2 not in adjacency:
                adjacency[node_name_2] = set()
            adjacency[node_name_1].add(node_name_2)
            adjacency[node_name_2].add(node_name_1)

    visited_node = set()

    node_net_num = {}
    for node_name in node_info:
        node_net_num[node_name] = len(node_to_net_dict[node_name])
    
    node_net_num_fea= {}
    node_net_num_max = max(node_net_num.values())
    print("node_net_num_max", node_net_num_max)
    for node_name in node_info:
        node_net_num_fea[node_name] = node_net_num[node_name]/node_net_num_max
    
    node_area_fea = {}
    node_area_max_node = max(node_info, key = lambda x : node_info[x]['x'] * node_info[x]['y'])
    node_area_max = node_info[node_area_max_node]['x'] * node_info[node_area_max_node]['y']
    print("node_area_max = {}".format(node_area_max))
    for node_name in node_info:
        node_area_fea[node_name] = node_info[node_name]['x'] * node_info[node_name]['y'] / node_area_max
    
    if "V" in node_info:
        add_node = "V"
        visited_node.add(add_node)
        node_id_to_name.append((add_node, node_net_num[add_node]))
        node_net_num.pop(add_node)
    
    add_node = max(node_net_num, key = lambda v: node_net_num[v])
    visited_node.add(add_node)
    node_id_to_name.append((add_node, node_net_num[add_node]))
    node_net_num.pop(add_node)

    while len(node_id_to_name) < len(node_info):
        candidates = {}
        for node_name in visited_node:
            if node_name not in adjacency:
                continue
            for node_name_2 in adjacency[node_name]:
                if node_name_2 in visited_node:
                    continue
                if node_name_2 not in candidates:
                    candidates[node_name_2] = 0
                candidates[node_name_2] += 1
        for node_name in node_info:
            if node_name not in candidates and node_name not in visited_node:
                candidates[node_name] = 0
        if len(candidates) > 0:
            if benchmark != 'ariane':
                if benchmark == "bigblue3":
                    add_node = max(candidates, key = lambda v: candidates[v]*1 + node_net_num[v]*100000 +\
                        node_info[v]['x']*node_info[v]['y'] * 1 +int(hash(v)%10000)*1e-6)
                else:
                    add_node = max(candidates, key = lambda v: candidates[v]*1 + node_net_num[v]*1000 +\
                        node_info[v]['x']*node_info[v]['y'] * 1 +int(hash(v)%10000)*1e-6)
            else:
                add_node = max(candidates, key = lambda v: candidates[v]*30000 + node_net_num[v]*1000 +\
                    node_info[v]['x']*node_info[v]['y']*1 +int(hash(v)%10000)*1e-6)
        else:
            if benchmark != 'ariane':
                if benchmark == "bigblue3":
                    add_node = max(node_net_num, key = lambda v: node_net_num[v]*100000 + node_info[v]['x']*node_info[v]['y']*1)
                else:
                    add_node = max(node_net_num, key = lambda v: node_net_num[v]*1000 + node_info[v]['x']*node_info[v]['y']*1)
            else:
                add_node = max(node_net_num, key = lambda v: node_net_num[v]*1000 + node_info[v]['x']*node_info[v]['y']*1)

        visited_node.add(add_node)
        node_id_to_name.append((add_node, node_net_num[add_node])) 
        node_net_num.pop(add_node)
    for i, (node_name, _) in enumerate(node_id_to_name):
        node_info[node_name]["id"] = i
    # print("node_id_to_name")
    # print(node_id_to_name)
    node_id_to_name_res = [x for x, _ in node_id_to_name]
    return node_id_to_name_res


class PlaceDB():

    def __init__(self, benchmark = "adaptec1"):
        self.benchmark = benchmark
        if benchmark == "ariane" or benchmark == "sample_clustered":
            path = benchmark + '/netlist.pb.txt'
            pbtxt = get_netlist_info_dict(path)
            self.node_info, self.node_info_raw_id_name = get_node_info(pbtxt)
            self.node_cnt = len(self.node_info)
            self.net_info, self.port_info = get_net_info(pbtxt)
            self.net_cnt = len(self.net_info)
            self.max_height, self.max_width = 357, 357
            self.port_to_net_dict = get_port_to_net_dict(self.port_info, self.net_info)
        else:
            assert os.path.exists(benchmark)
            node_file = open(os.path.join(benchmark, benchmark+".nodes"), "r")
            self.node_info, self.node_info_raw_id_name = read_node_file(node_file, benchmark)
            pl_file = open(os.path.join(benchmark, benchmark+".pl"), "r")
            self.port_info = {}
            self.node_cnt = len(self.node_info)
            node_file.close()
            net_file = open(os.path.join(benchmark, benchmark+".nets"), "r")
            self.net_info = read_net_file(net_file, self.node_info)
            self.net_cnt = len(self.net_info)
            net_file.close()
            pl_file = open(os.path.join(benchmark, benchmark+".pl"), "r")
            self.max_height, self.max_width = read_pl_file(pl_file, self.node_info)
            pl_file.close()
            self.port_to_net_dict = {}
        self.node_to_net_dict = get_node_to_net_dict(self.node_info, self.net_info)
        self.node_id_to_name = get_node_id_to_name_topology(self.node_info, self.node_to_net_dict, self.net_info, self.benchmark)

    def debug_str(self):
        print("node_cnt = {}".format(len(self.node_info)))
        print("net_cnt = {}".format(len(self.net_info)))
        print("max_height = {}".format(self.max_height))
        print("max_width = {}".format(self.max_width))

    def save_fig(self, file_path):
        fig1 = plt.figure()
        ax1 = fig1.add_subplot(111, aspect='equal')
        ax1.axes.xaxis.set_visible(False)
        ax1.axes.yaxis.set_visible(False)
        max_x = 0
        max_y = 0
        for node_name in self.node_info:
            x = self.node_info[node_name]['x']
            y = self.node_info[node_name]['y']
            size_x = self.node_info[node_name]['raw_x']
            size_y = self.node_info[node_name]['raw_y']
            max_x = max(max_x, x + size_x)
            max_y = max(max_y, y + size_y)
            ax1.add_patch(
                patches.Rectangle(
                    (x, y),   # (x,y)
                    size_x,          # width
                    size_y, linewidth=1, edgecolor='k',
                )
            )
            ax1.text(x, y, node_name, ha='center', va='center', fontsize=3, color='red')

        ax1.set_xlim(-100, max_x + 100)
        ax1.set_ylim(-100, max_y + 100)
        fig1.savefig(file_path, dpi=300, bbox_inches='tight')
        plt.close()


if __name__ == "__main__":
    placedb = PlaceDB("ariane")
    placedb.debug_str()

