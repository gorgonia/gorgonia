// +build cuda

package nnops

import (
	"gorgonia.org/cu/dnn"
)

type dropout struct {
	*cudnn.Dropout
}

// dropoutState is a dummy op. It's supposed to be like UniformRandomOp but doesn't actually do anything.
type dropoutState struct {
}
