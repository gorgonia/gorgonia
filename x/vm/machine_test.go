package xvm

import (
	"context"
	"reflect"
	"testing"

	"gorgonia.org/gorgonia"
)

func TestMachine_runAllNodes(t *testing.T) {
	inputC1 := make(chan ioValue, 0)
	outputC1 := make(chan gorgonia.Value, 1)
	inputC2 := make(chan ioValue, 0)
	outputC2 := make(chan gorgonia.Value, 1)

	n1 := &node{
		op:          &sumF32{},
		inputValues: make([]gorgonia.Value, 2),
		outputC:     outputC1,
		inputC:      inputC1,
	}
	n2 := &node{
		op:          &sumF32{},
		inputValues: make([]gorgonia.Value, 2),
		outputC:     outputC2,
		inputC:      inputC2,
	}
	type fields struct {
		nodes   []*node
		pubsubs []*pubsub
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
				nodes: []*node{n1, n2},
			},
			args{
				context.Background(),
			},
			false,
		},
		// TODO: Add test cases.
	}
	for _, tt := range tests {
		forty := gorgonia.F32(40.0)
		fortyTwo := gorgonia.F32(42.0)
		two := gorgonia.F32(2.0)
		t.Run(tt.name, func(t *testing.T) {
			m := &Machine{
				nodes:   tt.fields.nodes,
				pubsubs: tt.fields.pubsubs,
			}
			go func() {
				inputC1 <- struct {
					pos int
					v   gorgonia.Value
				}{
					0,
					&forty,
				}
				inputC1 <- struct {
					pos int
					v   gorgonia.Value
				}{
					1,
					&two,
				}
				inputC2 <- struct {
					pos int
					v   gorgonia.Value
				}{
					0,
					&forty,
				}
				inputC2 <- struct {
					pos int
					v   gorgonia.Value
				}{
					1,
					&two,
				}
			}()
			if err := m.runAllNodes(tt.args.ctx); (err != nil) != tt.wantErr {
				t.Errorf("Machine.runAllNodes() error = %v, wantErr %v", err, tt.wantErr)
			}
			out1 := <-outputC1
			out2 := <-outputC2
			if !reflect.DeepEqual(out1.Data(), fortyTwo.Data()) {
				t.Errorf("out1: bad result, expected %v, got %v", fortyTwo, out1)
			}
			if !reflect.DeepEqual(out2.Data(), fortyTwo.Data()) {
				t.Errorf("out2: bad result, expected %v, got %v", fortyTwo, out2)
			}
		})
	}
}
