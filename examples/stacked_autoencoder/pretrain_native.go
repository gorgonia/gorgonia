// +build native

package main

import (
	"gonum.org/v1/gonum/blas/gonum"
	. "gorgonia.org/gorgonia"
)

func init() {
	Use(gonum.Implementation{})
}
