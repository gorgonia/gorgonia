package compiler

import (
	"bytes"
	"fmt"
)

type Op interface {
	// indicates if the Op will return a pointer (allowing possible inplace edits) or by value
	// if it's false, the return value of the Op will be a copy of its input
	ReturnsPtr() bool

	// Does this op potentially call external (cgo or cuda) functions (thereby requiring extra overhead for Go's trampolining thing)
	CallsExtern() bool

	// overwriteInput() is a method which states which input the output will be overwriting.
	// This allows for some efficiency gains as the underlying arrays wouldn't have to be re-allocated.
	// The method returns an int instead of a bool because potentially different operations may be allowed
	// to overwrite certain inputs. For example, consider an operation to increment a value:
	// the IncrementOp would be a unary operator, and assuming we would like to overwrite the input,
	// the retVal of overwriteInput() will be 0 (inputs[0]).
	// -1 is returned if overwriting of input is disallowed
	OverwritesInput() int
}

type Program struct {
	instructions fragment
	args         int
	cpulocs      int
	gpulocs      int
	cpumem       int64
	gpumem       []int64
	g            *exprgraph.ExprGraph // original dag
	df           *dataflow            // dataflow analysis
	m            map[*Node]fragment   // store which nodes create which instructions
	sorted       Nodes
}

func (p *Program) String() string {
	var buf bytes.Buffer
	fmt.Fprintf(&buf, "Instructions:\n%s\nArgs: %d | CPU Memories: %d | GPU Memories: %d\nCPU Mem: %v | GPU Mem %v\n\nNode:instructions map:\n", p.instructions, p.args, p.cpulocs, p.gpulocs, p.cpumem, p.gpumem)

	for i, n := range p.sorted {
		fmt.Fprintf(&buf, "\t%d\t%x:", i, n.ID())
		frag := p.m[n]
		for j, instr := range frag {
			if j == 0 {
				fmt.Fprintf(&buf, "\t%v\n", instr)
			} else {
				fmt.Fprintf(&buf, "\t\t%v\n", instr)
			}
		}

	}

	return buf.String()
}

// Graph enables the end user to inspect the graph (typically useful for debugging)
func (p *program) Graph() *ExprGraph { return p.g }

func (p *program) CPUMemReq() int64 { return p.cpumem }

func (p *program) GPUMemReq() []int64 {
	retVal := make([]int64, len(p.gpumem))
	copy(retVal, p.gpumem)
	return retVal
}
