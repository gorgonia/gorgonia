package xvm

import (
	"context"
	"errors"

	"gorgonia.org/gorgonia"
)

type node struct {
	op             gorgonia.Op
	output         gorgonia.Value
	outputC        chan gorgonia.Value
	receivedValues int
	err            error
	inputValues    []gorgonia.Value
	inputC         chan struct {
		pos int
		v   gorgonia.Value
	}
}

type stateFn func(context.Context, *node) stateFn

func defaultState(_ context.Context, n *node) stateFn {
	if n.op == nil {
		return nil
	}
	return receiveInput
}

func receiveInput(ctx context.Context, n *node) stateFn {
	select {
	case <-ctx.Done():
		n.err = ctx.Err()
		return nil
	case input := <-n.inputC:
		if input.pos >= len(n.inputValues) {
			n.err = errors.New("bad arity")
			return nil
		}
		n.receivedValues++
		n.inputValues[input.pos] = input.v
		if n.receivedValues < len(n.inputValues) {
			return receiveInput
		}
	}
	return computeFwd
}

func computeFwd(_ context.Context, n *node) stateFn {
	v, err := n.op.Do(n.inputValues...)
	if err != nil {
		n.err = err
		return nil
	}
	n.output = v
	return emitOutput
}

func emitOutput(_ context.Context, _ *node) stateFn {
	return nil
}

func computeBackward(_ context.Context, _ *node) stateFn {
	return nil
}

func (n *node) ComputeForward(ctx context.Context) error {
	for state := defaultState; state != nil; {
		state = state(ctx, n)
	}
	return n.err
}
