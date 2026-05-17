### Why to use an A* for the global planning?

I have chosen the _A*_ because it is simple yet effective for out purposes.
It is faster than the Dijkstra's algorithm because it also has a heuristic function which allows us to
use the most promising node locations

#### Heuristic function `h(n)`
The heuristic function `h(n)` provides an estimated cost from the current node to the goal node,
acting as the algorithm's "informed guess" about the remaining path. 

Mathematically, for any given node n, the heuristic estimate must satisfy the condition `h(n)≤h*(n)`,
where `h*(n)` is the actual cost to the goal, making it admissible by never overestimating the true cost.

#### Total estimated cost `f(n)`

The total estimated cost `f(n)` is the cornerstone of _A*_ algorithm's decision-making process, combining both the actual
path cost and the heuristic estimate to evaluate each node's potential.

For any node n, this cost is calculated as: `f(n) = g(n) + h(n)`
Where:

`g(n)` represents the actual cost from the start to the current node,
`h(n)` represents the estimated cost from the current node to the goal. 
The algorithm uses this combined value to strategically choose which node to explore next,
always selecting the node with the lowest `f(n)` value from the open list, thus ensuring an optimal balance between known costs and estimated remaining distances.