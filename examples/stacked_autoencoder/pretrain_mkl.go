// +build !native

package main

import (
	. "gorgonia.org/gorgonia"
	"gorgonia.org/gorgonia/blase"
)

func init() {
	Use(blase.Implementation())
}
