package gorgonia

// Batched interface describes any object that can process batch work
type Batched interface {
	WorkAvailable() <-chan struct{}
	DoWork()
}

// BatchedBLAS interface describes any object that can process BLAS work in batch
type BatchedBLAS interface {
	Batched
	BLAS
}

// BatchedDevice is the superset of BatchedBLAS and the batched CUDA workflow.
type BatchedDevice interface {
	Batched
	Retval() interface{}
	Errors() error
}
