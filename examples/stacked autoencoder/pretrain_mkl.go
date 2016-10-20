// +build !native

package main

import (
	. "github.com/chewxy/gorgonia"
	"github.com/chewxy/gorgonia/blase"
)

func init() {
	Use(blase.Implementation())
}
