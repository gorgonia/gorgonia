package gorgonia

type Batched interface {
	WorkAvailable() <-chan struct{}
	DoWork()
}

type BatchedBLAS interface {
	Batched
	BLAS
}

type BatchedDevice interface {
	Batched
	Retval() interface{}
	Errors() error
}
