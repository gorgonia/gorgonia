This package provides a backend compatible with the `backend.ComputationGraph` interface.

The runtime is based on [Gorgonia](gorgonia.org/gorgonia)

## How to add new operators

An operator is basically an object that must fulfills the interface:

[embedmd]:# (operator.go /type operator/ /}/)
```go
type operator interface {
	// apply analyse the graph to find the children of the node
	// then extract its gorgonia.Node references
	// and assign the result of the operation to the node n
	apply(g *Graph, n *Node) error
	// init the operator with name and attributes as carried by the onnx.Operator
	init(o onnx.Operation) error
}
```

The operator must be registered to be usable, this is typically done within an init function:

[embedmd]:# (apigen_operators.go /type .* struct/ /}/)
```go
type hadamardProd struct{}
```

[embedmd]:# (apigen_operators.go /func init/ /^}/)
```go
func init() {
	register("Mul", &hadamardProd{})
}
```

### Tests

All the registered operators are tested against the official onnx tests if they exists.
simply run `go test -v` to check it out

### ApiGen

For common arithmetic operators, a `genapi` command can be found in a subdirectory of this package
