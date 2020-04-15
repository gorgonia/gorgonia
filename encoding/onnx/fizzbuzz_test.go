package gorgonnx

import (
	"fmt"
	"log"

	"github.com/owulveryck/onnx-go"
	"gorgonia.org/tensor"
)

func bin(n int, numDigits int) []float32 {
	f := make([]float32, numDigits)
	for i := uint(0); i < uint(numDigits); i++ {
		f[i] = float32((n >> i) & 1)
	}
	return f[:]
}

func dec(b []float32) int {
	for i := 0; i < len(b); i++ {
		if b[i] > 0.4 {
			return i
		}
	}
	panic("Sorry, I'm wrong")
}

func display(v []float32, i int) {
	switch dec(v) {
	case 0:
		fmt.Println(i)
	case 1:
		fmt.Println("Fizz")
	case 2:
		fmt.Println("Buzz")
	case 3:
		fmt.Println("FizzBuzz")
	}
}

func Example_fizzBuzz() {
	backend := NewGraph()

	m := onnx.NewModel(backend)

	err := m.UnmarshalBinary(fizzBuzzOnnx)

	if err != nil {
		log.Fatal(err)
	}

	input := tensor.New(tensor.WithShape(7), tensor.Of(tensor.Float32))
	for i := 1; i <= 100; i++ {
		for j, v := range bin(i, 7) {
			input.SetAt(v, j)
		}
		m.SetInput(0, input)
		err = backend.Run()
		if err != nil {
			log.Fatal(err)
		}
		output, err := m.GetOutputTensors()
		if err != nil {
			log.Fatal(err)
		}
		display(output[0].Data().([]float32), i)
	}
	// Output:
	// 1
	// 2
	// Fizz
	// 4
	// Buzz
	// Fizz
	// 7
	// 8
	// Fizz
	// Buzz
	// 11
	// Fizz
	// 13
	// 14
	// FizzBuzz
	// 16
	// 17
	// Fizz
	// 19
	// Buzz
	// Fizz
	// 22
	// 23
	// Fizz
	// Buzz
	// 26
	// Fizz
	// 28
	// 29
	// FizzBuzz
	// 31
	// 32
	// Fizz
	// 34
	// Buzz
	// Fizz
	// 37
	// 38
	// Fizz
	// Buzz
	// 41
	// Fizz
	// 43
	// 44
	// FizzBuzz
	// 46
	// 47
	// Fizz
	// 49
	// Buzz
	// Fizz
	// 52
	// 53
	// Fizz
	// Buzz
	// 56
	// Fizz
	// 58
	// 59
	// FizzBuzz
	// 61
	// 62
	// Fizz
	// 64
	// Buzz
	// Fizz
	// 67
	// 68
	// Fizz
	// Buzz
	// 71
	// Fizz
	// 73
	// 74
	// FizzBuzz
	// 76
	// 77
	// Fizz
	// 79
	// Buzz
	// Fizz
	// 82
	// 83
	// Fizz
	// Buzz
	// 86
	// Fizz
	// 88
	// 89
	// FizzBuzz
	// 91
	// 92
	// Fizz
	// 94
	// Buzz
	// Fizz
	// 97
	// 98
	// Fizz
	// Buzz
}
