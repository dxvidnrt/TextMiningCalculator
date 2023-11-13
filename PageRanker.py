from typing import List, Tuple, Dict, Any, Set


def _get_all_nodes(web_graph: List[Tuple[Any, Any]]) -> Set[Any]:
    """Returns a list of nodes given a web graph.

    Params:
        web_graph: List of edges.

    Returns:
        Set of nodes.
    """
    nodes = set()
    for (from_node, to_node) in web_graph:
        nodes.add(from_node)
        nodes.add(to_node)

    return nodes


def _get_outlinks_num(web_graph: List[Tuple[Any, Any]]) -> Dict[Any, int]:
    """Computes the number of outgoing links for each node in a web graph.

    Param:
        web_graph: List of edges.

    Returns:
        Dict with nodes as keys and the number of outgoing nodes as values.
    """
    outlinks = {node: 0 for node in _get_all_nodes(web_graph)}
    for (from_node, to_node) in web_graph:
        outlinks[from_node] += 1
    return outlinks


def _pr_to_str(pr: Dict) -> str:
    output = "\n"
    for key, value in sorted(pr.items()):
        output += f"{key}: {value} \n"
    return output


def pagerank(web_graph: List[Tuple[Any, Any]], q: float = 0.15, iterations: int = 3):
    """Computes PageRank for all nodes in a web graph.

    Params:
        web_graph: List of edges.
        q: Random jump probability.
        iterations: Number of iterations.

    Returns:
        Dict with node names as keys and PageRank scores as values.    
    """
    nodes = _get_all_nodes(web_graph)
    # Calculate the number of outgoing links of each page.
    outlinks_num = _get_outlinks_num(web_graph)
    # Collect all inlinks of a page for more efficient PageRank computation.
    inlinks = {node: [] for node in nodes}
    for (from_node, to_node) in web_graph:
        inlinks[to_node].append(from_node)

    # Identify and deal with rank sinks.
    for node, lnum in outlinks_num.items():
        if lnum == 0:
            print('Node {} is a rank sink!'.format(node))
            # Add links to all nodes (including the node itself).
            for to_node in nodes:
                inlinks[to_node].append(node)
            # Update outlinks count.
            outlinks_num[node] = len(nodes)

    # Initialize pagerank values.
    pr = {node: 1 / len(nodes) for node in nodes}

    print(f"PageRank iteration 0: {_pr_to_str(pr)}")

    # Calculate pagerank scores iteratively.
    for i in range(iterations):
        pr_old = pr.copy()
        for node in pr.keys():
            pr[node] = q / len(nodes)
            # Iterating over all pages p_i that link to node. 
            for from_node in inlinks[node]:
                pr[node] += (1 - q) * pr_old[from_node] / outlinks_num[from_node]
        print(f"PageRank iteration {i+1}: {_pr_to_str(pr)}")
