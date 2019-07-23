package compiler

import (
	"bytes"
	"fmt"

	"gorgonia.org/gorgonia/exprgraph"
)

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
