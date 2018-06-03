package nnops

import (
	"io/ioutil"
	"log"
	"testing"

	"gorgonia.org/gorgonia"
)

func TestDropout(t *testing.T) {
	g := gorgonia.NewGraph()
	x := gorgonia.NewMatrix(g, gorgonia.Float64, gorgonia.WithShape(2, 3), gorgonia.WithName("x"))
	do, _ := Dropout(x, 0.5)
	log.Printf("%v", do)
	ioutil.WriteFile("foo.dot", []byte(g.ToDot()), 0644)

}
