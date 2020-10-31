package exprgraph is an experimental rewrite of the basic graph data structure in Gorgonia. It is likely to fail and is not usable anywhere.

The goal is to rewrite the graph and nodes such that it creates less allocation. An idea is to separate the data from the structure.

# Exprgraph

## About

Exprgraph is a package holding a graph structure as a representation of a mathematical formulae.

### Graph

The top level structure of the package is a `*Graph` that is a direct, acyclic weighted graph.

#### Implementation

### Node

### Interfaces

#### Lifter