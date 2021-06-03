package xvm

import (
	"context"
	"fmt"
	"time"

	"gorgonia.org/gorgonia"
)

func Examplenode_Compute() {
	forty := gorgonia.F32(40.0)
	two := gorgonia.F32(2.0)
	n := &node{
		op:          &sumF32{},
		inputValues: make([]gorgonia.Value, 2),
		outputC:     make(chan gorgonia.Value, 0),
		inputC:      make(chan ioValue, 0),
	}
	ctx, cancel := context.WithTimeout(context.Background(), 100*time.Millisecond)
	// releases resources if ComputeForward completes before timeout elapses
	defer cancel()

	go n.Compute(ctx)
	n.inputC <- struct {
		pos int
		v   gorgonia.Value
	}{
		0,
		&forty,
	}
	n.inputC <- struct {
		pos int
		v   gorgonia.Value
	}{
		1,
		&two,
	}
	fmt.Println(<-n.outputC)
	// output: 42
}
