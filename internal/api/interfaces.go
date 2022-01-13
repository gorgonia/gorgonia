package api

// Queueer is anything that can enqueue an op, inputs and outputs.
type Queueer interface {
	Q(op ops.Op, inputs []datatype.Tensor, output data.Tensor) error
}
