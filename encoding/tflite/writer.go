// Package tflite exports trained gorgonia expression graphs as TensorFlow
// Lite models. Only float32 tensors and a small set of operators are
// supported.
package tflite

import (
	"encoding/binary"
	"fmt"
	"math"

	flatbuffers "github.com/google/flatbuffers/go"
)

// Activation is a fused activation function of an op
// (ActivationFunctionType in the TFLite schema). TANH and SIGN_BIT are
// omitted: most kernels do not implement them as fused activations.
type Activation int

const (
	ActNone      Activation = 0
	ActRelu      Activation = 1
	ActReluN1To1 Activation = 2
	ActRelu6     Activation = 3
)

// TFLite schema constants (tensorflow/lite/schema/schema.fbs).
const (
	opAdd            = 0
	opFullyConnected = 9
	opLogistic       = 14
	opSoftmax        = 25
	opTanh           = 28

	optionsFullyConnected = 8 // BuiltinOptions union member index
	optionsSoftmax        = 9
	optionsAdd            = 11
)

// Tensor is a handle to a tensor of the model being built.
type Tensor int

type tensorInfo struct {
	name   string
	shape  []int32
	buffer int32
}

type opInfo struct {
	opcode      int32
	inputs      []int32
	outputs     []int32
	optionsType byte
	options     func(b *flatbuffers.Builder) flatbuffers.UOffsetT
}

// Model builds a TFLite model with a single subgraph.
type Model struct {
	tensors []tensorInfo
	buffers [][]byte
	ops     []opInfo
	inputs  []int32
	outputs []int32
	err     error
}

func NewModel() *Model {
	return &Model{
		buffers: [][]byte{nil}, // buffer 0 is the empty sentinel
	}
}

func (m *Model) fail(format string, args ...interface{}) Tensor {
	if m.err == nil {
		m.err = fmt.Errorf(format, args...)
	}
	return Tensor(-1)
}

func (m *Model) tensor(name string, shape []int, buffer int32) Tensor {
	s := make([]int32, len(shape))
	for i, v := range shape {
		s[i] = int32(v)
	}
	m.tensors = append(m.tensors, tensorInfo{name: name, shape: s, buffer: buffer})
	return Tensor(len(m.tensors) - 1)
}

func (m *Model) shape(t Tensor) []int {
	shape := make([]int, len(m.tensors[t].shape))
	for i, v := range m.tensors[t].shape {
		shape[i] = int(v)
	}
	return shape
}

func numElements(shape []int) int {
	n := 1
	for _, v := range shape {
		n *= v
	}
	return n
}

func floatBytes(v []float32) []byte {
	b := make([]byte, 4*len(v))
	for i, x := range v {
		binary.LittleEndian.PutUint32(b[i*4:], math.Float32bits(x))
	}
	return b
}

// Input adds a model input of the given shape.
func (m *Model) Input(name string, shape []int) Tensor {
	t := m.tensor(name, shape, 0)
	m.inputs = append(m.inputs, int32(t))
	return t
}

// Constant adds a tensor with fixed contents, such as trained weights.
func (m *Model) Constant(name string, shape []int, data []float32) Tensor {
	if len(data) != numElements(shape) {
		return m.fail("constant %s: %d values do not fill shape %v", name, len(data), shape)
	}
	m.buffers = append(m.buffers, floatBytes(data))
	return m.tensor(name, shape, int32(len(m.buffers)-1))
}

// Output marks a tensor as a model output.
func (m *Model) Output(t Tensor) {
	m.outputs = append(m.outputs, int32(t))
}

func (m *Model) addOp(op opInfo, name string, shape []int) Tensor {
	out := m.tensor(name, shape, 0)
	op.outputs = []int32{int32(out)}
	m.ops = append(m.ops, op)
	return out
}

// FullyConnected computes x*filterᵀ + bias with an optional fused
// activation. filter must have shape [out, in]; bias must have shape [out],
// or be negative for no bias.
func (m *Model) FullyConnected(x, filter, bias Tensor, act Activation) Tensor {
	fs := m.shape(filter)
	if len(fs) != 2 {
		return m.fail("fully connected filter must be [out, in], got %v", fs)
	}
	xs := m.shape(x)
	if len(xs) == 0 || xs[len(xs)-1] != fs[1] {
		return m.fail("fully connected input %v does not match filter %v", xs, fs)
	}
	shape := append(append([]int{}, xs[:len(xs)-1]...), fs[0])
	inputs := []int32{int32(x), int32(filter), int32(bias)}
	return m.addOp(opInfo{
		opcode:      opFullyConnected,
		inputs:      inputs,
		optionsType: optionsFullyConnected,
		options: func(b *flatbuffers.Builder) flatbuffers.UOffsetT {
			b.StartObject(4)
			b.PrependByteSlot(0, byte(act), 0)
			return b.EndObject()
		},
	}, m.tensors[filter].name+"/out", shape)
}

// Add computes a + b with an optional fused activation. The shapes must
// either match or be broadcastable by the TFLite ADD kernel.
func (m *Model) Add(a, b Tensor, act Activation) Tensor {
	return m.addOp(opInfo{
		opcode:      opAdd,
		inputs:      []int32{int32(a), int32(b)},
		optionsType: optionsAdd,
		options: func(fb *flatbuffers.Builder) flatbuffers.UOffsetT {
			fb.StartObject(2)
			fb.PrependByteSlot(0, byte(act), 0)
			return fb.EndObject()
		},
	}, "add/out", m.shape(a))
}

// Tanh applies the elementwise hyperbolic tangent.
func (m *Model) Tanh(x Tensor) Tensor {
	return m.addOp(opInfo{opcode: opTanh, inputs: []int32{int32(x)}}, "tanh/out", m.shape(x))
}

// Logistic applies the elementwise sigmoid.
func (m *Model) Logistic(x Tensor) Tensor {
	return m.addOp(opInfo{opcode: opLogistic, inputs: []int32{int32(x)}}, "logistic/out", m.shape(x))
}

// Softmax applies softmax with the given beta over the last dimension.
func (m *Model) Softmax(x Tensor, beta float32) Tensor {
	return m.addOp(opInfo{
		opcode:      opSoftmax,
		inputs:      []int32{int32(x)},
		optionsType: optionsSoftmax,
		options: func(b *flatbuffers.Builder) flatbuffers.UOffsetT {
			b.StartObject(1)
			b.PrependFloat32Slot(0, beta, 0.0)
			return b.EndObject()
		},
	}, "softmax/out", m.shape(x))
}

func fbIntVector(b *flatbuffers.Builder, vals []int32) flatbuffers.UOffsetT {
	b.StartVector(4, len(vals), 4)
	for i := len(vals) - 1; i >= 0; i-- {
		b.PrependInt32(vals[i])
	}
	return b.EndVector(len(vals))
}

func fbOffsetVector(b *flatbuffers.Builder, offs []flatbuffers.UOffsetT) flatbuffers.UOffsetT {
	b.StartVector(4, len(offs), 4)
	for i := len(offs) - 1; i >= 0; i-- {
		b.PrependUOffsetT(offs[i])
	}
	return b.EndVector(len(offs))
}

// Bytes serializes the model as a TFLite flatbuffer.
func (m *Model) Bytes() ([]byte, error) {
	if m.err != nil {
		return nil, m.err
	}
	if len(m.outputs) == 0 {
		return nil, fmt.Errorf("model has no outputs")
	}

	b := flatbuffers.NewBuilder(16 * 1024)

	buffers := make([]flatbuffers.UOffsetT, len(m.buffers))
	for i, data := range m.buffers {
		var off flatbuffers.UOffsetT
		if len(data) > 0 {
			b.Prep(16, len(data)) // TFLite wants tensor data 16-byte aligned
			off = b.CreateByteVector(data)
		}
		b.StartObject(1)
		if len(data) > 0 {
			b.PrependUOffsetTSlot(0, off, 0)
		}
		buffers[i] = b.EndObject()
	}

	tensors := make([]flatbuffers.UOffsetT, len(m.tensors))
	for i, t := range m.tensors {
		nameOff := b.CreateString(t.name)
		shapeOff := fbIntVector(b, t.shape)
		b.StartObject(4)
		b.PrependUOffsetTSlot(0, shapeOff, 0)
		// slot 1 (type) omitted: FLOAT32 = 0 is the default
		b.PrependUint32Slot(2, uint32(t.buffer), 0)
		b.PrependUOffsetTSlot(3, nameOff, 0)
		tensors[i] = b.EndObject()
	}

	codes := []int32{}
	codeIndex := map[int32]uint32{}
	for _, op := range m.ops {
		if _, ok := codeIndex[op.opcode]; !ok {
			codeIndex[op.opcode] = uint32(len(codes))
			codes = append(codes, op.opcode)
		}
	}
	opcodes := make([]flatbuffers.UOffsetT, len(codes))
	for i, code := range codes {
		b.StartObject(4)
		if code <= 127 {
			b.PrependInt8Slot(0, int8(code), 0) // deprecated_builtin_code
		} else {
			b.PrependInt8Slot(0, 127, 0)
		}
		b.PrependInt32Slot(3, code, 0)
		opcodes[i] = b.EndObject()
	}

	operators := make([]flatbuffers.UOffsetT, len(m.ops))
	for i, op := range m.ops {
		var opts flatbuffers.UOffsetT
		if op.options != nil {
			opts = op.options(b)
		}
		inOff := fbIntVector(b, op.inputs)
		outOff := fbIntVector(b, op.outputs)
		b.StartObject(5)
		b.PrependUint32Slot(0, codeIndex[op.opcode], 0)
		b.PrependUOffsetTSlot(1, inOff, 0)
		b.PrependUOffsetTSlot(2, outOff, 0)
		if op.optionsType != 0 {
			b.PrependByteSlot(3, op.optionsType, 0)
			b.PrependUOffsetTSlot(4, opts, 0)
		}
		operators[i] = b.EndObject()
	}

	subgraphName := b.CreateString("main")
	tensorsOff := fbOffsetVector(b, tensors)
	inputsOff := fbIntVector(b, m.inputs)
	outputsOff := fbIntVector(b, m.outputs)
	operatorsOff := fbOffsetVector(b, operators)
	b.StartObject(5)
	b.PrependUOffsetTSlot(0, tensorsOff, 0)
	b.PrependUOffsetTSlot(1, inputsOff, 0)
	b.PrependUOffsetTSlot(2, outputsOff, 0)
	b.PrependUOffsetTSlot(3, operatorsOff, 0)
	b.PrependUOffsetTSlot(4, subgraphName, 0)
	subgraph := b.EndObject()

	description := b.CreateString("Exported by github.com/mattn/go-tflite/export.")
	opcodesOff := fbOffsetVector(b, opcodes)
	subgraphsOff := fbOffsetVector(b, []flatbuffers.UOffsetT{subgraph})
	buffersOff := fbOffsetVector(b, buffers)
	b.StartObject(5)
	b.PrependUint32Slot(0, 3, 0) // schema version
	b.PrependUOffsetTSlot(1, opcodesOff, 0)
	b.PrependUOffsetTSlot(2, subgraphsOff, 0)
	b.PrependUOffsetTSlot(3, description, 0)
	b.PrependUOffsetTSlot(4, buffersOff, 0)
	model := b.EndObject()

	b.FinishWithFileIdentifier(model, []byte("TFL3"))
	return b.FinishedBytes(), nil
}
