package tensor

import (
	"reflect"
	"testing"
)

// junkyard tests the miscelleneous things

func TestRandom(t *testing.T) {
	const size = 50
	for _, typ := range numberTypes.set {
		r := Random(typ, size)

		typR := reflect.TypeOf(r).Elem()
		valR := reflect.ValueOf(r)

		if typR != typ.Type {
			t.Errorf("Expected typR to be %v. Got %v instead", typ, typR)
		}
		if valR.Len() != size {
			t.Errorf("Expected length to be %v. Got %v instead", size, valR.Len())
		}

	}
}
