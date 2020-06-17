package exprgraph

import (
	"fmt"
	"log"
	"strconv"

	"gorgonia.org/gorgonia"
	"gorgonia.org/gorgonia/values"
	"gorgonia.org/tensor"
)

// T2T tries to find a `tensor.Tensor` from a gorgonia.Tensor
func T2T(a gorgonia.Tensor) tensor.Tensor {
	switch t := a.(type) {
	case Node:
		return T2T(t.Value.(gorgonia.Tensor))
	case tensor.Tensor:
		return t
	default:
		panic("XXX")
	}
}

func tonode(t values.Value) node {
	switch a := t.(type) {
	case Node:
		return node{Node: a}
	case *Symbolic:
		return node{
			Node: Node{
				Value:  a,
				NodeID: -1,
			},
		}
	case tensor.Tensor:
		return node{
			Node: Node{
				Value:  a,
				NodeID: -1,
			},
		}
	case *node:
		return *a
	case *Node:
		return node{Node: *a}
	default:
		log.Printf("tonode %T not handleed", t)
	}
	panic("Unreachable")
}

func consFmtStr(a fmt.State, c rune) string {
	retVal := "%"
	acceptable := []rune{'+', '-', ' ', '#', '0'}
	for _, f := range acceptable {
		if a.Flag(int(f)) {
			retVal = retVal + string(f)
		}
	}
	width, wok := a.Width()
	prec, pok := a.Precision()
	if wok {
		retVal = retVal + strconv.Itoa(width)
	}
	if pok {
		retVal = retVal + "." + strconv.Itoa(prec)
	}
	retVal = retVal + string(c)
	return retVal
}
