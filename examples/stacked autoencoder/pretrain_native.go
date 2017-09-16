// +build native

package main

import (
	. "github.com/chewxy/gorgonia"
	"gonum.org/v1/gonum/blas/gonum"
)

func init() {
	Use(gonum.Implementation{})
}
