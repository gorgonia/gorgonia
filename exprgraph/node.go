package exprgraph

import (
	"github.com/chewxy/hm"
)

// flag is a flag to identify what kind of node it is
// The default flag is:
//	is an expression (isStmt = 0)
//	is not a constant value (isConst = 0)
//	is mutable (isImmutable = 0)
// 	is not a argument (isArg = 0)
//	is not a root node (isRoot = 0)
// 	is deterministic (isRandom = 0)
//
// Despite having 256 possible combinations, there are only very few valid states. See the isValid() method
type flag byte

const (
	isStmt      flag = 1 << iota // does this node represent a statement?
	isConst                      // does this node represent a constant value?
	isImmutable                  // does this node's Op represent an immutable (in-place) value?
	isArg                        // is this node an argument node (i.e. Op == nil)
	isRoot                       // is this node the root node?
	isRandom                     // does this node represent a non-determinism?
)

// there are very few valid states despite the combinatorial combination
func (f flag) isValid() bool {
	isStmt := (f&isStmt == 1)
	isConst := (f&isConst == 1)
	isImmutable := (f&isImmutable == 1)
	isArg := (f&isArg == 1)
	isRoot := (f&isRoot == 1)
	isRandom := (f&isRandom == 1)
	switch {
	case isStmt:
		// statements cannot be anything else other than statements.
		if isConst || isImmutable || isArg || isRandom || isRoot {
			return false
		}
		return true
	case isConst:
		if isStmt || isArg || isRandom {
			return false
		}
		return true
	case isImmutable:
		if isStmt || isConst || isArg || isRandom {
			return false
		}
		return true
	case isArg:
		if isStmt || isConst || isRandom {
			return false
		}
		return true
	case isRandom:
		if isStmt || isConst || isArg {
			return false
		}
		return true
	default:
		return false
	}
}

// NodeID represents a node's ID. It also implements gonum.org/v1/gonum/graph.Node
type NodeID int64

func (n NodeID) ID() int64 { return int64(n) }

// Node is a payload
type Node struct {
	NodeID
	t       hm.Type
	op      Op        //TODO
	boundTo dualValue //TODO
	dataOn  device    //TODO
	flag    flag
}

// Node implements gorgonia.Result

func (n Node) Node() Node   { return n }
func (n Node) Nodes() Nodes { return Nodes{n} }
func (n Node) Err() error   { return nil }

// GraphNode is a tuple of a graph object and a node. This allows for querying the payload of the Node.
//
// This is the object that should be used for any kind of query (topsort, etc)
type GraphNode struct {
	*Graph
	Node
}

//go:notinheap
type gn struct {
	*Graph
	Node
}

// node is a node for internal use. Its graph is defined by the links (i.e. pointers).
// if the ID is negative, it means that the node is in-progress
type node struct {
	Node
	children []*node
}

type dualValue interface {
	Value() interface{} //tmp
	Deriv() interface{}
}

type Op interface{}

type device int
