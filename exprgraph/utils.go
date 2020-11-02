package exprgraph

import (
	"fmt"
	"strconv"

	"gorgonia.org/gorgonia"
	"gorgonia.org/gorgonia/values/dual"
	"gorgonia.org/tensor"
)

// T2T tries to find a `tensor.Tensor` from a gorgonia.Tensor
// it returns nil if no tensor is found
func T2T(a gorgonia.Tensor) tensor.Tensor {
	switch t := a.(type) {
	case *Node:
		if t.Tensor == nil {
			return nil
		}
		return T2T(t.Tensor.(gorgonia.Tensor))
	case *dual.Dual:
		return t
	case tensor.Tensor:
		return t
	default:
		return nil
	}
}

/*
func tonode(t gorgonia.Tensor) node {
	switch a := t.(type) {
	case Node:
		return node{Node: a}
	case *Symbolic:
		return node{
			Node: Node{
				Tensor: a,
				id:     -1,
			},
		}
	case *dual.Dual:
		return node{
			Node: Node{
				Tensor: a,
				id:     -1,
			},
		}
	case tensor.Tensor:
		return node{
			Node: Node{
				Tensor: a,
				id:     -1,
			},
		}
	case *node:
		return *a
	case *Node:
		return node{Node: *a}
	default:
		log.Printf("tonode %T not handled", t)
	}
	panic("Unreachable")
}
*/

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
