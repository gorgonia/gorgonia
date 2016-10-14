// +build native

package main

import (
	. "github.com/chewxy/gorgonia"
	"github.com/gonum/blas/native"
)

func init() {
	Use(native.Implementation{})
}
