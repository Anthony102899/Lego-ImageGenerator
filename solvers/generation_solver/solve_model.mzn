int : nodes_num;
int : nums_edge_collision;
int : nums_edge_connect;
set of int: NODE = 1..nodes_num;
set of int: EDGE_COLL = 1..nums_edge_collision;
set of int: EDGE_CON = 1..nums_edge_connect;

array[NODE] of float : node_volume;
array[NODE] of var 0..1 : flag;
array[EDGE_COLL] of NODE : from_collision;
array[EDGE_COLL] of NODE : to_collision;
array[EDGE_CON] of NODE : from_connect;
array[EDGE_CON] of NODE : to_connect;

array[NODE] of var bool : node;
array[EDGE_CON] of var bool : selected_edges;

%%%%%%%%%%%%%%%%%% constraints
% constraint no collisions
constraint
forall(e in EDGE_COLL) (
        node[from_collision[e]] + node[to_collision[e]] <= 1
);

% constraint all connected
include "connected.mzn";
constraint connected(from_connect, to_connect, node, selected_edges);

%constraint node[1] = 1;
%constraint node[2] = 1;
%constraint node[3] = 1;

%%%%%%%%%%%%%%%%%% objectives
% calculate total brick volume
var float : node_volume_obj = sum(n in NODE) (node[n] * node_volume[n] * flag[n]);

var float : obj;
obj = node_volume_obj;
solve maximize obj;
