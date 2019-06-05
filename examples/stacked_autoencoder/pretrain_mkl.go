// +build !native

package main

import (
	. "gorgonia"
	"gorgonia.org/gorgonia/blase"
)

func init() {
	Use(blase.Implementation())
}
