package tensor

import (
	"log"
	"math"
	"math/rand"
	"reflect"
	"sort"

	"github.com/chewxy/math32"
)

// SortIndex is similar to numpy's argsort
// TODO: tidy this up
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

// SampleIndex samples a slice or a Tensor.
// TODO: tidy this up.
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
	case *Dense:
		var i int
		switch list.t.Kind() {
		case reflect.Float64:
			var sum float64
			r := rand.Float64()
			data := list.Float64s()
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
		case reflect.Float32:
			var sum float32
			r := rand.Float32()
			data := list.Float32s()
			l = len(data)
			for {
				datum := data[i]
				if math32.IsNaN(datum) || math32.IsInf(datum, 0) {
					return i
				}

				sum += datum
				if sum > r && i > 0 {
					return i
				}
				i++
			}
		default:
			panic("not yet implemented")
		}
	default:
		panic("Not yet implemented")
	}
	return l - 1
}
