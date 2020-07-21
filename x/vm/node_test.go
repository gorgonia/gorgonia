package xvm

import (
	"context"
	"errors"
	"reflect"
	"testing"

	"gorgonia.org/gorgonia"
)

func Test_receiveInput(t *testing.T) {
	cancelCtx, cancel := context.WithCancel(context.Background())
	inputC := make(chan ioValue, 0)
	type args struct {
		ctx context.Context
		o   *node
		fn  func()
	}
	tests := []struct {
		name string
		args args
		want stateFn
	}{
		{
			"context cancelation",
			args{
				cancelCtx,
				&node{
					inputC: make(chan ioValue, 0),
				},
				nil,
			},
			nil,
		},
		{
			"bad input value position",
			args{
				context.Background(),
				&node{
					inputC:      inputC,
					inputValues: make([]gorgonia.Value, 1),
				},
				func() {
					inputC <- struct {
						pos int
						v   gorgonia.Value
					}{
						pos: 1,
						v:   nil,
					}
				},
			},
			nil,
		},
		{
			"more value to receive",
			args{
				context.Background(),
				&node{
					inputC:      inputC,
					inputValues: make([]gorgonia.Value, 2),
				},
				func() {
					inputC <- struct {
						pos int
						v   gorgonia.Value
					}{
						pos: 0,
						v:   nil,
					}
				},
			},
			receiveInput,
		},
		{
			"no input chan go to conpute",
			args{
				context.Background(),
				&node{
					inputValues: make([]gorgonia.Value, 1),
				},
				nil,
			},
			computeFwd,
		},
		{
			"all done go to compute",
			args{
				context.Background(),
				&node{
					inputC:      inputC,
					inputValues: make([]gorgonia.Value, 1),
				},
				func() {
					inputC <- struct {
						pos int
						v   gorgonia.Value
					}{
						pos: 0,
						v:   nil,
					}
				},
			},
			computeFwd,
		},
		// TODO: Add test cases.
	}
	cancel()
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if tt.args.fn != nil {
				go tt.args.fn()
			}
			got := receiveInput(tt.args.ctx, tt.args.o)
			gotPrt := reflect.ValueOf(got).Pointer()
			wantPtr := reflect.ValueOf(tt.want).Pointer()
			if gotPrt != wantPtr {
				t.Errorf("receiveInput() = %v, want %v", got, tt.want)
			}
		})
	}
}

func Test_computeFwd(t *testing.T) {
	type args struct {
		in0 context.Context
		n   *node
	}
	tests := []struct {
		name string
		args args
		want stateFn
	}{
		{
			"simple no error",
			args{
				nil,
				&node{
					op:          &noOpTest{},
					inputValues: []gorgonia.Value{nil},
				},
			},
			emitOutput,
		},
		{
			"simple with error",
			args{
				nil,
				&node{
					op:          &noOpTest{err: errors.New("")},
					inputValues: []gorgonia.Value{nil},
				},
			},
			nil,
		},
		// TODO: Add test cases.
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := computeFwd(tt.args.in0, tt.args.n)
			gotPrt := reflect.ValueOf(got).Pointer()
			wantPtr := reflect.ValueOf(tt.want).Pointer()
			if gotPrt != wantPtr {
				t.Errorf("computeFwd() = %v, want %v", got, tt.want)
			}
		})
	}
}

func Test_node_ComputeForward(t *testing.T) {
	type fields struct {
		op             gorgonia.Op
		output         gorgonia.Value
		outputC        chan gorgonia.Value
		receivedValues int
		err            error
		inputValues    []gorgonia.Value
		inputC         chan ioValue
	}
	type args struct {
		ctx context.Context
	}
	tests := []struct {
		name    string
		fields  fields
		args    args
		wantErr bool
	}{
		{
			"simple",
			fields{
				op: nil,
			},
			args{
				nil,
			},
			false,
		},
		// TODO: Add test cases.
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			n := &node{
				op:             tt.fields.op,
				output:         tt.fields.output,
				outputC:        tt.fields.outputC,
				receivedValues: tt.fields.receivedValues,
				err:            tt.fields.err,
				inputValues:    tt.fields.inputValues,
				inputC:         tt.fields.inputC,
			}
			if err := n.Compute(tt.args.ctx); (err != nil) != tt.wantErr {
				t.Errorf("node.ComputeForward() error = %v, wantErr %v", err, tt.wantErr)
			}
		})
	}
}

type sumF32 struct{}

func (*sumF32) Do(v ...gorgonia.Value) (gorgonia.Value, error) {
	val := v[0].Data().(float32) + v[1].Data().(float32)
	value := gorgonia.F32(val)
	return &value, nil
}

func Test_emitOutput(t *testing.T) {
	cancelCtx, cancel := context.WithCancel(context.Background())
	outputC1 := make(chan gorgonia.Value, 0)
	outputC2 := make(chan gorgonia.Value, 1)
	type args struct {
		ctx context.Context
		n   *node
	}
	tests := []struct {
		name string
		args args
		want stateFn
	}{
		{
			"nil node",
			args{nil, nil},
			nil,
		},
		{
			"context cancelation",
			args{
				cancelCtx,
				&node{
					outputC: outputC1,
				},
			},
			nil,
		},
		{
			"emit output",
			args{
				context.Background(),
				&node{
					outputC: outputC2,
				},
			},
			nil,
		},
	}
	cancel()
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := emitOutput(tt.args.ctx, tt.args.n)
			gotPrt := reflect.ValueOf(got).Pointer()
			wantPtr := reflect.ValueOf(tt.want).Pointer()
			if gotPrt != wantPtr {
				t.Errorf("emitOutput() = %v, want %v", got, tt.want)
			}
		})
	}
}

func Test_computeBackward(t *testing.T) {
	type args struct {
		in0 context.Context
		in1 *node
	}
	tests := []struct {
		name string
		args args
		want stateFn
	}{
		{
			"simple",
			args{
				nil,
				nil,
			},
			nil,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := computeBackward(tt.args.in0, tt.args.in1); !reflect.DeepEqual(got, tt.want) {
				t.Errorf("computeBackward() = %v, want %v", got, tt.want)
			}
		})
	}
}

func Test_newOp(t *testing.T) {
	g := gorgonia.NewGraph()
	fortyTwo := gorgonia.F32(42.0)
	n1 := gorgonia.NodeFromAny(g, fortyTwo)
	n2 := gorgonia.NodeFromAny(g, fortyTwo)
	addOp, err := gorgonia.Add(n1, n2)
	if err != nil {
		t.Fatal(err)
	}
	type args struct {
		n             *gorgonia.Node
		hasOutputChan bool
	}
	tests := []struct {
		name string
		args args
		want *node
	}{
		{
			"no op",
			args{nil, false},
			nil,
		},
		{
			"add with outputChan",
			args{addOp, true},
			&node{
				id:          addOp.ID(),
				op:          addOp.Op(),
				inputC:      make(chan ioValue, 0),
				outputC:     make(chan gorgonia.Value, 0),
				inputValues: make([]gorgonia.Value, 2),
			},
		},
		{
			"add without outputChan",
			args{addOp, false},
			&node{
				id:          addOp.ID(),
				op:          addOp.Op(),
				inputC:      make(chan ioValue, 0),
				outputC:     nil,
				inputValues: make([]gorgonia.Value, 2),
			},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := newOp(tt.args.n, tt.args.hasOutputChan)
			if got == tt.want {
				return
			}
			if got.id != tt.want.id {
				t.Errorf("newOp() = \n%#v, want \n%#v", got, tt.want)
			}
			if !reflect.DeepEqual(got.op, tt.want.op) {
				t.Errorf("newOp() = \n%#v, want \n%#v", got, tt.want)
			}
			if !reflect.DeepEqual(got.inputValues, tt.want.inputValues) {
				t.Errorf("newOp() = \n%#v, want \n%#v", got, tt.want)
			}
			if got.receivedValues != tt.want.receivedValues {
				t.Errorf("newOp() = \n%#v, want \n%#v", got, tt.want)
			}
			if got.err != tt.want.err {
				t.Errorf("newOp() = \n%#v, want \n%#v", got, tt.want)
			}
			if (got.inputC == nil && tt.want.inputC != nil) ||
				(got.inputC != nil && tt.want.inputC == nil) {
				t.Errorf("newOp() = \n%#v, want \n%#v", got, tt.want)
			}
			if (got.outputC == nil && tt.want.outputC != nil) ||
				(got.outputC != nil && tt.want.outputC == nil) {
				t.Errorf("newOp() = \n%#v, want \n%#v", got, tt.want)
			}
			if cap(got.outputC) != cap(tt.want.outputC) {
				t.Errorf("newOp() = \n%#v, want \n%#v", got, tt.want)
			}
			if len(got.outputC) != len(tt.want.outputC) {
				t.Errorf("newOp() = \n%#v, want \n%#v", got, tt.want)
			}
			if cap(got.inputC) != cap(tt.want.inputC) {
				t.Errorf("newOp() = \n%#v, want \n%#v", got, tt.want)
			}
			if len(got.inputC) != len(tt.want.inputC) {
				t.Errorf("newOp() = \n%#v, want \n%#v", got, tt.want)
			}

		})
	}
}

func Test_newInput(t *testing.T) {
	g := gorgonia.NewGraph()
	fortyTwo := gorgonia.F32(42.0)
	n1 := gorgonia.NodeFromAny(g, &fortyTwo)
	type args struct {
		n *gorgonia.Node
	}
	tests := []struct {
		name string
		args args
		want *node
	}{
		{
			"nil",
			args{nil},
			nil,
		},
		{
			"simple",
			args{n1},
			&node{
				outputC: make(chan gorgonia.Value, 0),
				output:  &fortyTwo,
			},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := newInput(tt.args.n)
			if got == tt.want {
				return
			}
			compareNodes(t, got, tt.want)
		})
	}
}

func compareNodes(t *testing.T, got, want *node) {
	if got.id != want.id {
		t.Errorf("nodes ID are different = \n%#v, want \n%#v", got.id, want.id)
	}
	if !reflect.DeepEqual(got.op, want.op) {
		t.Errorf("nodes OP are different = \n%#v, want \n%#v", got.op, want.op)
	}
	if !reflect.DeepEqual(got.inputValues, want.inputValues) {
		t.Errorf("nodes inputValues are different = \n%#v, want \n%#v", got.inputValues, want.inputValues)
	}
	if got.receivedValues != want.receivedValues {
		t.Errorf("nodes receivedValues are different = \n%#v, want \n%#v", got.receivedValues, want.receivedValues)
	}
	if got.err != want.err {
		t.Errorf("nodes errors are different = \n%#v, want \n%#v", got.err, want.err)
	}
	if (got.inputC == nil && want.inputC != nil) ||
		(got.inputC != nil && want.inputC == nil) {
		t.Errorf("newInput() = \n%#v, want \n%#v", got, want)
	}
	if (got.outputC == nil && want.outputC != nil) ||
		(got.outputC != nil && want.outputC == nil) {
		t.Errorf("newInput() = \n%#v, want \n%#v", got, want)
	}
	if cap(got.outputC) != cap(want.outputC) {
		t.Errorf("newInput() = \n%#v, want \n%#v", got, want)
	}
	if len(got.outputC) != len(want.outputC) {
		t.Errorf("newInput() = \n%#v, want \n%#v", got, want)
	}
	if cap(got.inputC) != cap(want.inputC) {
		t.Errorf("newInput() = \n%#v, want \n%#v", got, want)
	}
	if len(got.inputC) != len(want.inputC) {
		t.Errorf("newInput() = \n%#v, want \n%#v", got, want)
	}

}
