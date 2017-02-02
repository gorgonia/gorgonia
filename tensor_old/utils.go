package tensor

import (
	"log"
	"math"
	"math/rand"
	"sort"

	tf32 "github.com/chewxy/gorgonia/tensor/f32"
	tf64 "github.com/chewxy/gorgonia/tensor/f64"
)

func MinInt(a, b int) int {
	if a < b {
		return a
	}
	return b
}

// similar to numpy argsort
func SortIndex(in interface{}) (out []int) {
	switch list := in.(type) {
	case []int:
		orig := make([]int, len(list))
		out = make([]int, len(list))
		copy(orig, list)
		sort.Ints(list)
		for i, s := range list {
			for j, o := range orig {
				if o == s {
					out[i] = j
					break
				}
			}
		}
	case []float64:
		orig := make([]float64, len(list))
		out = make([]int, len(list))
		copy(orig, list)
		sort.Float64s(list)

		for i, s := range list {
			for j, o := range orig {
				if o == s {
					out[i] = j
					break
				}
			}
		}
	case sort.Interface:
		sort.Sort(list)

		log.Printf("TODO: SortIndex for sort.Interface not yet done.")
	}

	return
}

func SampleIndex(in interface{}) int {
	var l int
	switch list := in.(type) {
	case []int:
		var sum, i int
		l = len(list)
		r := rand.Int()
		for {
			sum += list[i]
			if sum > r && i > 0 {
				return i
			}
			i++
		}
	case []float64:
		var sum float64
		var i int
		l = len(list)
		r := rand.Float64()
		for {
			sum += list[i]
			if sum > r && i > 0 {
				return i
			}
			i++
		}
	case *tf64.Tensor:
		var sum float64
		var i int

		r := rand.Float64()
		data := list.Data().([]float64)
		l = len(data)
		for {
			datum := data[i]
			if math.IsNaN(datum) || math.IsInf(datum, 0) {
				return i
			}

			sum += datum
			if sum > r && i > 0 {
				return i
			}
			i++
		}
	case *tf32.Tensor:
		var sum float32
		var i int

		r := float32(rand.Float64())
		data := list.Data().([]float32)
		l = len(data)
		for {
			datum := data[i]
			if math.IsNaN(float64(datum)) || math.IsInf(float64(datum), 0) {
				return i
			}

			sum += datum
			if sum > r && i > 0 {
				return i
			}
			i++
		}
	default:
		panic("Not yet implemented")
	}
	return l - 1
}
