package gorgonia

// Batched interface describes ny object that can process batch work
type Batched interface {
	WorkAvailable() <-chan struct{}
	DoWork()
}

// BatchedBLAS interface describes ny object that can process BLAS work in batch
type BatchedBLAS interface {
	Batched
	BLAS
}

// BatchedDevice ...
type BatchedDevice interface {
	Batched
	Retval() interface{}
	Errors() error
}
