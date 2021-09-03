// +build cuda

package execution

import "gorgonia.org/tensor"

type ExternMetadata struct {
	tensor.Engine

	sync.Mutex

	// operational stuff
	u cu.Device // device currently in use

	engines       []cuda.Engine
	workAvailable chan bool
	syncChan      chan struct{}
	initialized   bool
}
