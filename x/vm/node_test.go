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
	inputC := make(chan struct {
		pos int
		v   gorgonia.Value
	}, 0)
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
				&node{},
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
					op:          &dummyTestOp{},
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
					op:          &dummyTestOp{err: errors.New("")},
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
		inputC         chan struct {
			pos int
			v   gorgonia.Value
		}
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
			if err := n.ComputeForward(tt.args.ctx); (err != nil) != tt.wantErr {
				t.Errorf("node.ComputeForward() error = %v, wantErr %v", err, tt.wantErr)
			}
		})
	}
}
