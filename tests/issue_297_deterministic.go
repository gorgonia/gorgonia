package tests

import (
	"fmt"
	"log"

	"gorgonia.org/gorgonia"
	"gorgonia.org/tensor"
)

func main() {
	g := gorgonia.NewGraph()

	mat1Val := tensor.New(tensor.WithShape(2, 2), tensor.WithBacking([]float32{1, 2, 3, 4}))
	mat2Val := tensor.New(tensor.WithShape(2, 2), tensor.WithBacking([]float32{5, 6, 7, 8}))
	mat3Val := tensor.New(tensor.WithShape(2, 2), tensor.WithBacking([]float32{9, 10, 11, 12}))

	mat1 := gorgonia.NewMatrix(g, tensor.Float32, gorgonia.WithShape(2, 2), gorgonia.WithName("mat1"), gorgonia.WithValue(mat1Val))
	mat2 := gorgonia.NewMatrix(g, tensor.Float32, gorgonia.WithShape(2, 2), gorgonia.WithName("mat2"), gorgonia.WithValue(mat2Val))
	mat3 := gorgonia.NewMatrix(g, tensor.Float32, gorgonia.WithShape(2, 2), gorgonia.WithName("mat3"), gorgonia.WithValue(mat3Val))

	z, err := gorgonia.Mul(mat1, mat2)
	if err != nil {
		log.Fatal(err)
	}

	var zOutput gorgonia.Value
	gorgonia.Read(z, &zOutput)

	z2, err := gorgonia.Mul(z, mat3)
	if err != nil {
		log.Fatal(err)
	}

	var z2Output gorgonia.Value
	gorgonia.Read(z2, &z2Output)

	machine := gorgonia.NewTapeMachine(g)
	if err := machine.RunAll(); err != nil {
		log.Fatal(err)
	}

	// Verify the results
	expectedZ2 := tensor.New(tensor.WithShape(2, 2), tensor.WithBacking([]float32{413, 454, 937, 1030}))

	actualData := z2Output.Data().([]float32)
	expectedData := expectedZ2.Data().([]float32)

	if len(actualData) != len(expectedData) {
		log.Fatalf("Deterministic test failed! Length mismatch. Expected: %d, Got: %d", len(expectedData), len(actualData))
	}

	for i := range actualData {
		if actualData[i] != expectedData[i] {
			log.Fatalf("Deterministic test failed! Mismatch at index %d. Expected: %f, Got: %f\nExpected:\n%v\nGot:\n%v", i, expectedData[i], actualData[i], expectedZ2, z2Output)
		}
	}
	fmt.Println("Deterministic test passed!")
	fmt.Printf("Result:\n%v\n", z2Output)
}