package gorgonia

import (
	"bytes"
	"fmt"
	"sort"
	"unsafe"

	"github.com/chewxy/gorgonia/tensor/types"
	"github.com/xtgo/set"
)

// Nodes is a slice of nodes, but it also acts as a set of nodes by implementing the Sort interface
type Nodes []*Node

// implement sort.Interface

func (ns Nodes) Len() int { return len(ns) }
func (ns Nodes) Less(i, j int) bool {
	return uintptr(unsafe.Pointer(ns[i])) < uintptr(unsafe.Pointer(ns[j]))
	// return ns[i].Hashcode() < ns[j].Hashcode()
}
func (ns Nodes) Swap(i, j int) { ns[i], ns[j] = ns[j], ns[i] }

// uses xtgo/set stuff

func (ns Nodes) Set() Nodes {
	sort.Sort(ns)
	size := set.Uniq(ns)
	ns = ns[:size]
	return ns
}

// Add adds to set
func (ns Nodes) Add(n *Node) Nodes {
	for _, node := range ns {
		if node == n {
			return ns
		}
	}
	ns = append(ns, n)
	return ns
}

func (ns Nodes) Contains(want *Node) bool {
	for _, n := range ns {
		if n == want {
			return true
		}
	}
	return false
}

// Format implements fmt.Formatter, which allows Nodes to be differently formatted depending on the verbs
func (ns Nodes) Format(s fmt.State, c rune) {
	delimiter := ","
	if s.Flag(' ') {
		delimiter = " "
	}
	if s.Flag('+') {
		delimiter = ",\n"
	}
	switch c {
	case 'd':
		s.Write([]byte("["))
		for i, n := range ns {
			fmt.Fprintf(s, "%x", n.Hashcode())
			if i < len(ns)-1 {
				fmt.Fprintf(s, "%s ", delimiter)
			}
		}
		s.Write([]byte("]"))
	case 'v', 's':
		s.Write([]byte("["))
		for i, n := range ns {
			if s.Flag('#') {
				fmt.Fprintf(s, "%s :: %v", n.Name(), n.t)
			} else {
				fmt.Fprintf(s, "%s", n.Name())
			}
			if i < len(ns)-1 {
				fmt.Fprintf(s, "%s ", delimiter)
			}
		}
		s.Write([]byte("]"))
	case 'Y':
		if s.Flag('#') {
			s.Write([]byte("["))
			for i, n := range ns {
				fmt.Fprintf(s, "%v", n.t)
				if i < len(ns)-1 {
					fmt.Fprintf(s, "%s ", delimiter)
				}
			}
			s.Write([]byte("]"))
		}
	case 'P':
		s.Write([]byte("["))
		for i, n := range ns {
			fmt.Fprintf(s, "%p", n)
			if i < len(ns)-1 {
				fmt.Fprintf(s, "%s ", delimiter)
			}
		}
		s.Write([]byte("]"))
	}
}

// Difference is ns - other. Bear in mind it is NOT commutative
func (ns Nodes) Difference(other Nodes) Nodes {
	sort.Sort(ns)
	sort.Sort(other)
	s := append(ns, other...)
	count := set.Diff(s, len(ns))
	return s[:count]
}

func (ns Nodes) Intersect(other Nodes) Nodes {
	sort.Sort(ns)
	sort.Sort(other)
	s := append(ns, other...)
	count := set.Inter(s, len(ns))
	return s[:count]
}

func (ns Nodes) AllSameGraph() bool {
	if len(ns) == 0 {
		return false
	}

	var g *ExprGraph
	for _, n := range ns {
		if !n.isConstant() {
			g = n.g
			break
		}
	}

	for _, n := range ns {
		if n.g != g && !n.isConstant() {
			return false
		}
	}
	return true
}

func (ns Nodes) mapSet() NodeSet { return NewNodeSet(ns...) }

func (ns Nodes) index(n *Node) int {
	for i, node := range ns {
		if node == n {
			return i
		}
	}
	return -1
}

func (ns Nodes) reverse() {
	l := len(ns)
	for i := l/2 - 1; i >= 0; i-- {
		o := l - 1 - i
		ns[i], ns[o] = ns[o], ns[i]
	}
}

func (ns Nodes) replace(what, with *Node) Nodes {
	for i, n := range ns {
		if n == what {
			ns[i] = with
		}
	}
	return ns
}

func (ns Nodes) remove(what *Node) Nodes {
	for i := ns.index(what); i != -1; i = ns.index(what) {
		copy(ns[i:], ns[i+1:])
		ns[len(ns)-1] = nil // to prevent any unwanted references so things can be GC'd away
		ns = ns[:len(ns)-1]
	}
	return ns
}

func (ns Nodes) shapes() []types.Shape {
	retVal := make([]types.Shape, len(ns))
	for i, n := range ns {
		retVal[i] = n.shape
	}
	return retVal
}

func (ns Nodes) dimSizers() []DimSizer {
	retVal := make([]DimSizer, len(ns))
	for i, n := range ns {
		retVal[i] = n.shape
	}
	return retVal
}

/* TYPES */

type Types []Type

func (ts Types) String() string {
	var buf bytes.Buffer
	buf.WriteString("[")
	for i, t := range ts {
		buf.WriteString(t.String())
		if i < len(ts)-1 {
			buf.WriteString(", ")
		}
	}
	buf.WriteString("]")
	return buf.String()
}
