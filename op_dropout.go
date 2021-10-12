package gorgonia

import (
	"fmt"
	"hash"
	"hash/fnv"
	"log"

	"github.com/chewxy/hm"
	"github.com/pkg/errors"
	"gorgonia.org/tensor"
)

type randomGenF func() float64

type dropoutOp struct {
	probability float64
	isTraining  bool
	rndGen      randomGenF
}

func newDropoutOp(probability float64, rndGen randomGenF) *dropoutOp {
	dropoutop := &dropoutOp{
		probability: probability,
		isTraining:  true,
		rndGen:      rndGen,
	}

	return dropoutop
}

func (op *dropoutOp) SetTraining(isTraining bool) error { op.isTraining = isTraining; return nil }

func (op *dropoutOp) Arity() int { return 1 }

func (op *dropoutOp) ReturnsPtr() bool { return false }

func (op *dropoutOp) CallsExtern() bool { return false }

func (op *dropoutOp) WriteHash(h hash.Hash) { fmt.Fprintf(h, op.String()) }

func (op *dropoutOp) Hashcode() uint32 { return simpleHash(op) }

func (op *dropoutOp) String() string { return fmt.Sprintf("Dropout{training=%v}()", op.isTraining) }

func (op *dropoutOp) InferShape(inputs ...DimSizer) (tensor.Shape, error) {
	return inputs[0].(tensor.Shape), nil
}

func (op *dropoutOp) Type() hm.Type {
	a := hm.TypeVariable('a')
	return hm.NewFnType(a, a) // f(float64) float64
}

func (op *dropoutOp) OverwritesInput() int { return -1 }

func (op *dropoutOp) checkInput(inputs ...Value) (tensor.Tensor, error) {
	if err := checkArity(op, len(inputs)); err != nil {
		return nil, err
	}

	var (
		in tensor.Tensor
		ok bool
	)

	if in, ok = inputs[0].(tensor.Tensor); !ok {
		return nil, errors.Errorf("Expected input to be a tensor")
	}

	return in, nil
}

func (op *dropoutOp) Do(inputs ...Value) (retVal Value, err error) {
	inputTensor, err := op.checkInput(inputs...)
	if err != nil {
		return nil, fmt.Errorf("Can't check Dropout input: %w", err)
	}

	if op.probability == 0.0 || !op.isTraining {
		return inputTensor, nil
	}

	ret := tensor.New(tensor.WithShape(inputTensor.Shape().Clone()...), tensor.Of(inputTensor.Dtype()))

	op.do(inputTensor.Data(), ret.Data())

	return ret, nil
}

func (op *dropoutOp) UsePreallocDo(prealloc Value, inputs ...Value) (Value, error) {
	inputTensor, err := op.checkInput(inputs...)
	if err != nil {
		return nil, fmt.Errorf("Can't check Dropout input: %w", err)
	}

	if op.probability == 0.0 || !op.isTraining {
		return inputTensor, nil
	}

	op.do(inputTensor.Data, prealloc.Data())

	return prealloc, nil
}

func (op *dropoutOp) do(input, output interface{}) {
	keepProb := 1.0 - op.probability

	switch v := input.(type) {
	case []float32:
		outputV := output.([]float32)
		for i, d := range v {
			r := float32(op.rndGen())
			if r < float32(keepProb) {
				outputV[i] = d / float32(keepProb)
			}
		}
	case []float64:
		outputV := output.([]float64)
		for i, d := range v {
			r := op.rndGen()
			if r < keepProb {
				outputV[i] = d / keepProb
			}
		}
	default:
		log.Panicf("unknown dtype: %T", output)
	}
}

// DoDiff calculates the diff and sets its value to the output node. Implementation for ADOp interface.
func (op *dropoutOp) DoDiff(ctx ExecutionContext, inputs Nodes, output *Node) error {
	if len(inputs) != 1 {
		return fmt.Errorf("dropout diff requires 1 arguments")
	}

	diff := &dropoutDiffOp{op}
	xdv, ydv := getDV(inputs[0], output)

	_, err := diff.UsePreallocDo(xdv.d, xdv.Value, output.Value(), ydv.d)

	return err
}

// SymDiff applies the diff op. Implementation for SDOp interface.
func (op *dropoutOp) SymDiff(inputs Nodes, output, grad *Node) (Nodes, error) {
	err := checkArity(op, len(inputs))
	if err != nil {
		return nil, err
	}

	input := inputs[0]
	diff := &dropoutDiffOp{op}

	ret, err := ApplyOp(diff, input, output, grad)
	if err != nil {
		return nil, err
	}

	return Nodes{ret}, nil
}

// DiffWRT is an implementation for the SDOp interface
func (op *dropoutOp) DiffWRT(inputs int) []bool {
	if inputs != 1 {
		panic(fmt.Sprintf("dropout operator only supports one input, got %d instead", inputs))
	}

	return []bool{true}
}

type dropoutDiffOp struct {
	*dropoutOp
}

// Arity returns 1
func (op dropoutDiffOp) Arity() int { return 3 }

// Type returns a → a
func (op dropoutDiffOp) Type() hm.Type {
	t := hm.TypeVariable('a')
	return hm.NewFnType(t, t, t, t)
}

// InferShape returns the output shape as a function of the inputs
func (op dropoutDiffOp) InferShape(ds ...DimSizer) (tensor.Shape, error) {
	return ds[0].(tensor.Shape), nil
}

// Do executes the op
func (op *dropoutDiffOp) Do(values ...Value) (Value, error) {
	input := values[0].(*tensor.Dense)
	output := values[1].(*tensor.Dense)
	grad := values[2].(*tensor.Dense)

	dy, err := CloneValue(input)
	if err != nil {
		return nil, err
	}

	v, err := op.UsePreallocDo(dy, input, output, grad)

	return v, err
}

func (op *dropoutDiffOp) UsePreallocDo(prealloc Value, inputs ...Value) (retVal Value, err error) {
	input := inputs[0].(*tensor.Dense)
	result := prealloc.(*tensor.Dense)
	output := inputs[1].(*tensor.Dense)
	outGrad := inputs[2].(*tensor.Dense)

	probability := op.probability

	switch input.Dtype() {
	case Float64:
		dy := outGrad.Float64s()
		dx := result.Float64s()
		outputA := output.Float64s()

		for i := 0; i < len(dy); i++ {
			if probability != 0 && outputA[i] != 0 {
				dx[i] = dy[i] / probability
			} else {
				dx[i] = 0.0
			}
		}
	case Float32:
		dy := outGrad.Float32s()
		dx := result.Float32s()
		outputA := output.Float32s()

		for i := 0; i < len(dy); i++ {
			if probability != 0 && outputA[i] != 0 {
				dx[i] = dy[i] / float32(probability)
			} else {
				dx[i] = 0.0
			}
		}
	}

	return prealloc, nil
}

// ReturnsPtr indicates if the Op will return a pointer (allowing possible inplace edits) or by value
// if it's false, the return value of the Op will be a copy of its input
func (op dropoutDiffOp) ReturnsPtr() bool { return true }

// CallsExtern returns false.
func (op dropoutDiffOp) CallsExtern() bool { return false }

// OverwritesInput is a method which states which input the output will be overwriting.
// This allows for some efficiency gains as the underlying arrays wouldn't have to be re-allocated.
// The method returns an int instead of a bool because potentially different operations may be allowed
// to overwrite certain inputs. For example, consider an operation to increment a value:
// the IncrementOp would be a unary operator, and assuming we would like to overwrite the input,
// the retVal of overwriteInput() will be 0 (inputs[0]).
// -1 is returned if overwriting of input is disallowed
func (op dropoutDiffOp) OverwritesInput() int { return -1 }

/* Other methods */

func (op dropoutDiffOp) WriteHash(h hash.Hash) { h.Write([]byte(op.String())) }

func (op dropoutDiffOp) Hashcode() uint32 {
	h := fnv.New32a()
	op.WriteHash(h)
	return h.Sum32()
}

func (op dropoutDiffOp) String() string { return fmt.Sprintf("dropoutDiffOp(%f)", op.probability) }

// ensure it complies with the Op interface
var (
	_ Op          = &dropoutOp{}
	_ ADOp        = &dropoutOp{}
	_ SDOp        = &dropoutOp{}
	_ TrainModeOp = &dropoutOp{}
)
