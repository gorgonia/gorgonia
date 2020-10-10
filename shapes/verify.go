package shapes

import (
	"fmt"
	"reflect"
	"strconv"
	"strings"
)

// Verify checks that the values of a data structure conforms to the expected shape, given in struct tags.
//
// Example - consider the following struct:
// 	type Foo struct {
//		A tensor.Tensor `shape:"(a, b)"`
//		B tensor.Tensor `shape:"(b, c)"`
//	}
//
// At runtime, the fields A and B would be populated with a Tensor of arbitrary shape.
// Verify can verify that A and B have the expected patterns of shapes.
//
// So, if A has a shape of (2, 3), B's shape cannot be (4, 5). It can be (3, 5).
func Verify(any interface{}) error {
	t := reflect.TypeOf(any)
	if t.Kind() != reflect.Struct {
		// no op - or maybe panic
		return nil
	}

	m := make(map[string]Abstract)
	fields := t.NumField()
	for i := 0; i < fields; i++ {
		f := t.Field(i)
		shapeStr := f.Tag.Get("shape")
		if shapeStr == "" { // TODO: nested types?
			continue
		}
		abs := parseAbs(shapeStr)
		m[f.PkgPath+"."+f.Name] = abs
	}
	m2 := make(map[string]Shape)
	v := reflect.ValueOf(any)
	for i := 0; i < fields; i++ {
		f := v.Field(i)
		iface := f.Interface()
		tf := t.Field(i)
		switch ff := iface.(type) {
		case Shape:
			m2[tf.PkgPath+"."+tf.Name] = ff
		case Shaper:
			m2[tf.PkgPath+"."+tf.Name] = ff.Shape()
		}
	}

	// make constraints
	var cs constraints
	for k, s := range m {
		v := m2[k]
		cs = append(cs, exprConstraint{s, v})
	}
	_, err := solve(cs, nil)

	return err
}

func verifyConstraints() {}

func parseAbs(a string) Abstract {
	a = strings.Trim(a, "()")
	a = strings.Replace(a, " ", "", -1)
	s := strings.Split(a, ",")
	retVal := make(Abstract, 0, len(s))
	for i := range s {
		if sz, err := strconv.Atoi(s[i]); err == nil {
			retVal = append(retVal, Size(sz))
			continue
		}
		if len(s[i]) > 1 {
			panic(fmt.Sprintf("Unsupported shape variable type: %q. Variables need to be a single char", s[i]))
		}
		retVal = append(retVal, Var(s[i][0]))
	}
	return retVal
}
