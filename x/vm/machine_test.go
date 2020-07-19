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

func TestNewMachine(t *testing.T) {
	simpleGraph := gorgonia.NewGraph()
	forty := gorgonia.F32(40.0)
	//fortyTwo := gorgonia.F32(42.0)
	two := gorgonia.F32(2.0)
	n1 := gorgonia.NodeFromAny(simpleGraph, forty)
	n2 := gorgonia.NodeFromAny(simpleGraph, two)
	added, _ := gorgonia.Add(n1, n2)
	type args struct {
		g *gorgonia.ExprGraph
	}
	tests := []struct {
		name string
		args args
		want *Machine
	}{
		{
			"nil graph",
			args{nil},
			nil,
		},
		{
			"simple graph WIP",
			args{
				simpleGraph,
			},
			&Machine{
				nodes: []*node{
					{
						id: added.ID(),
						op: added.Op(),
					},
				},
			},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := NewMachine(tt.args.g); !reflect.DeepEqual(got, tt.want) {
				t.Errorf("NewMachine() = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestMachine_runAllPubSub(t *testing.T) {
	type fields struct {
		nodes   []*node
		pubsubs []*pubsub
	}
	type args struct {
		ctx context.Context
	}
	tests := []struct {
		name   string
		fields fields
		args   args
	}{
		{
			"no subscribers",
			fields{},
			args{
				context.Background(),
			},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			m := &Machine{
				nodes:   tt.fields.nodes,
				pubsubs: tt.fields.pubsubs,
			}
			m.runAllPubSub(tt.args.ctx)
		})
	}
	t.Run("functionnal test in goroutine", func(t *testing.T) {
		publishers := make([]chan gorgonia.Value, 4)
		for i := range publishers {
			publishers[i] = make(chan gorgonia.Value, 0)
		}
		subscribers := make([]chan ioValue, 2)
		for i := range subscribers {
			subscribers[i] = make(chan ioValue, 0)
		}
		p := &pubsub{
			publishers:  publishers,
			subscribers: subscribers,
		}
		machine := &Machine{
			pubsubs: []*pubsub{
				p,
			},
		}
		machine.runAllPubSub(context.Background())
		fortyTwo := gorgonia.F32(42.0)
		fortyThree := gorgonia.F32(43.0)
		publishers[0] <- &fortyTwo
		publishers[2] <- &fortyThree
		sub0_0 := <-subscribers[0]
		sub1_0 := <-subscribers[1]
		sub0_1 := <-subscribers[0]
		sub1_1 := <-subscribers[1]
		if !reflect.DeepEqual(sub0_0, ioValue{pos: 0, v: &fortyTwo}) {
			t.Errorf("sub0_0 - expected %v, got %v", ioValue{pos: 0, v: &fortyTwo}, sub0_0)
		}
		if !reflect.DeepEqual(sub1_0, ioValue{pos: 0, v: &fortyTwo}) {
			t.Errorf("sub1_0 - expected %v, got %v", ioValue{pos: 0, v: &fortyTwo}, sub1_0)
		}
		if !reflect.DeepEqual(sub0_1, ioValue{pos: 2, v: &fortyThree}) {
			t.Errorf("sub0_1 - expected %v, got %v", ioValue{pos: 2, v: &fortyThree}, sub0_1)
		}
		if !reflect.DeepEqual(sub1_1, ioValue{pos: 2, v: &fortyThree}) {
			t.Errorf("sub1_1 - expected %v, got %v", ioValue{pos: 2, v: &fortyThree}, sub1_1)
		}
	})
}
