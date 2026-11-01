#!/usr/bin/env python
"""
Run a Meshroom graph that has no CameraInit, as meshroom_batch would.

meshroom_batch insists on feeding images to a CameraInit node, so a graph that starts
from an existing SfMData file (see benchmark_meshroom.py --from_sfm) runs here instead.
Overrides use meshroom_batch's syntax and semantics. The cache goes next to the project
file, as meshroom_batch does with -s and no --cache.

Run with $MESHROOM_ROOT on PYTHONPATH:
    python baseline/meshroom_run_graph.py <graph.mg> <output_dir> <project.mg> \
        [--paramOverrides Node:attr=value ...]
"""
import argparse
import os
import re

from meshroom import setupEnvironment

setupEnvironment()

import meshroom.core  # noqa: E402
import meshroom.core.graph  # noqa: E402

# meshroom_batch's override syntax: NodeType:attr=value for every node of a type,
# NodeName.attr=value for one node; attr may be dotted into a group.
OVERRIDE = re.compile(r"(\w+)([:.])(\w[\w.]*)=(.*)")


def apply_overrides(graph, overrides):
    for p in overrides:
        match = OVERRIDE.match(p)
        if not match:
            raise ValueError(f"Invalid param override: {p}")
        node, sep, param, value = match.groups()
        if sep == ":":
            nodes = graph.nodesOfType(node)
            if not nodes:
                raise ValueError(f'No node with the type "{node}" in the scene.')
        else:
            nodes = [graph.findNode(node)]
        for n in nodes:
            print(f"Overrides {n.name}.{param}={value}")
            n.attribute(param).value = value


def main():
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument("graph", help="Graph template (.mg) to run")
    parser.add_argument("output", help="Folder the Publish nodes copy results to")
    parser.add_argument("project", help="Where to save the project; the cache goes beside it")
    parser.add_argument("--paramOverrides", nargs="*", default=[], help="As in meshroom_batch")
    args = parser.parse_args()

    meshroom.core.initPlugins()
    meshroom.core.initNodes()

    graph = meshroom.core.graph.Graph(name=os.path.splitext(os.path.basename(args.project))[0])
    with meshroom.core.graph.GraphModification(graph):
        graph.initFromTemplate(args.graph, publishOutputs=True)
        publish = graph.nodesOfType("Publish")
        if not publish:
            raise RuntimeError(f"{args.graph} has no Publish node")
        for node in publish:
            node.output.value = os.path.abspath(args.output)
        apply_overrides(graph, args.paramOverrides)

    if not graph.canComputeLeaves:
        raise RuntimeError("Graph cannot be computed. Check for compatibility issues.")

    # setupProjectFile puts MeshroomCache next to the project, where prune_cache looks.
    graph.save(os.path.abspath(args.project), setupProjectFile=True)
    print(f'File successfully saved: "{args.project}" (cache: {graph.cacheDir})')

    meshroom.core.graph.executeGraph(graph)


if __name__ == "__main__":
    main()
