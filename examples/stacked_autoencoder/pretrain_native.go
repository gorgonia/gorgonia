// +build native

package main

import (
	"gonum.org/v1/gonum/blas/gonum"
	. "gorgonia"
)

func init() {
	Use(gonum.Implementation{})
}
