from place_db import PlaceDB
from prim import prim_real

def comp_res(placedb, node_pos, ratio):

    hpwl = 0.0
    cost = 0.0
    for net_name in placedb.net_info:
        max_x = 0.0
        min_x = placedb.max_height * 1.1  # (why 1.1?)
        max_y = 0.0
        min_y = placedb.max_height * 1.1  # (why 1.1?)
        for node_name in placedb.net_info[net_name]["nodes"]:
            if node_name not in node_pos:
                continue
            h = placedb.node_info[node_name]['x']
            w = placedb.node_info[node_name]['y']
            pin_x = node_pos[node_name][0] * ratio + h / 2.0 + placedb.net_info[net_name]["nodes"][node_name]["x_offset"]
            pin_y = node_pos[node_name][1] * ratio + w / 2.0 + placedb.net_info[net_name]["nodes"][node_name]["y_offset"]
            max_x = max(pin_x, max_x)
            min_x = min(pin_x, min_x)
            max_y = max(pin_y, max_y)
            min_y = min(pin_y, min_y)
        for port_name in placedb.net_info[net_name]["ports"]:
            h = placedb.port_info[port_name]['x']
            w = placedb.port_info[port_name]['y']
            pin_x = h
            pin_y = w
            max_x = max(pin_x, max_x)
            min_x = min(pin_x, min_x)
            max_y = max(pin_y, max_y)
            min_y = min(pin_y, min_y)
        if min_x <= placedb.max_height:
            hpwl_tmp = (max_x - min_x) + (max_y - min_y)
        else:
            hpwl_tmp = 0
        if "weight" in placedb.net_info[net_name]:
            hpwl_tmp *= placedb.net_info[net_name]["weight"]
        hpwl += hpwl_tmp
        net_node_set = set.union(set(placedb.net_info[net_name]["nodes"]),
                            set(placedb.net_info[net_name]["ports"]))
        for net_node in list(net_node_set):
            if net_node not in node_pos and net_node not in placedb.port_info:
                net_node_set.discard(net_node)
        prim_cost = prim_real(net_node_set, node_pos, placedb.net_info[net_name]["nodes"], ratio, placedb.node_info, placedb.port_info)
        if "weight" in placedb.net_info[net_name]:
            prim_cost *= placedb.net_info[net_name]["weight"]
        assert hpwl_tmp <= prim_cost +1e-5
        cost += prim_cost
    return hpwl, cost
