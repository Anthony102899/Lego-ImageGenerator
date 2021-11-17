int : base_count;
int : nodes_num;
int : nums_edge_collision;
int : nums_edge_connect;

set of int: NODE = 1..nodes_num;
set of int: EDGE_COLL = 1..nums_edge_collision;
set of int: EDGE_CON = 1..nums_edge_connect;

array[NODE] of float : node_sd;
array[NODE] of float : node_area;
array[NODE] of float : node_weight;
array[EDGE_COLL] of NODE : from_collision;
array[EDGE_COLL] of NODE : to_collision;
array[EDGE_CON] of NODE : from_connect;
array[EDGE_CON] of NODE : to_connect;
array[NODE] of var bool : node;
array[EDGE_CON] of var bool : selected_edges;
array[NODE] of float : square_root_weight;

%%%%%%%%%%%%%%%%%% constraints
% keep the base bricks 
constraint forall(i in 1..base_count) (node[i] = 1);

% discard all bricks surpassing the graph boundary
constraint forall(i in base_count+1..nodes_num) ((node[i] == 0 /\ node_sd[i] < 0 ) \/ (node_sd[i] >= 0));

% get square root of weight
% constraint forall(n in NODE) (node_weight[n] == square_root_weight[n] * square_root_weight[n] /\ square_root_weight[n] >= 0);

% constraint no collisions
constraint forall(e in EDGE_COLL) (node[from_collision[e]] + node[to_collision[e]] <= 1);

% Discarded because the connection is useless
% constraint all connected
% include "connected.mzn";
% constraint connected(from_connect, to_connect, node, selected_edges);

%%%%%%%%%%%%%%%%%% objectives
% calculate total brick sd
float : sum_area = sum(n in NODE) ( node_weight[n] * node_area[n]);
var float : sum_selected_area = sum(n in NODE) ( node_weight[n] * node_area[n] * node[n]);
float : sum_sd = sum(n in NODE) ((node_weight[n]) * node_sd[n]);
var float : sum_selected_sd = sum(n in NODE) (node_sd[n] * (node_weight[n]) * node[n]);
var float : La = 1 - 1.0 * (sum_selected_area / sum_area);
var float : Ls = 1 + 0.005 * (sum_selected_sd / sum_sd);
var float : obj = La * Ls;
% var float : node_sd_obj = sum(n in NODE) (node[n] * node_weight[n] * (150 * node_sd[n] + 1 * node_area[n]));

% var float : obj;
% obj = node_sd_obj;
% solve maximize obj;
solve minimize obj;