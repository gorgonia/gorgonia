package stdops

import (
	"gorgonia.org/gorgonia/internal/errors"
	"gorgonia.org/shapes"
	"gorgonia.org/tensor"
)

// // funcName returns the name of the function.
// func funcName(f interface{}) string {
// 	val := reflect.ValueOf(f)
// 	if val.Kind() != reflect.Func {
// 		return "Not a function"
// 	}

// 	//TODO(go1.18) - pc := uintptr(val.UnsafePointer())
// 	pc := val.Pointer()
// 	return runtime.FuncForPC(pc).Name()
// }

func handleFuncOpts[DT any, T tensor.Basic[DT]](e tensor.Engine, t T, expShape shapes.Shape, opts ...tensor.FuncOpt) (retVal T, fo tensor.Option, err error) {
	switch e := e.(type) {
	// case tensor.SpecializedFuncOptHandler[DT, T]:
	// 	return e.HandleFuncOptsSpecialized(t, expShape, opts...)
	case tensor.FuncOptHandler[DT]:
		var ret tensor.Basic[DT]
		ret, fo, err = e.HandleFuncOpts(t, expShape, opts...)
		if err != nil {
			return retVal, fo, errors.Wrapf(err, errors.FailedFuncOpt, errors.ThisFn())
		}
		var ok bool
		if retVal, ok = ret.(T); !ok {
			return retVal, fo, errors.Errorf("Expected retVal type to be %T", retVal)
		}
		return
	case tensor.DescFuncOptHandler[DT]:
		var ret tensor.DescWithStorage
		ret, fo, err = e.HandleFuncOptsDesc(t, expShape, opts...)
		if err != nil {
			return retVal, fo, err
		}
		var ok bool
		if retVal, ok = ret.(T); !ok {
			return retVal, fo, errors.Errorf("Expected retVal type to be %T", retVal)
		}
		return
	}
	return retVal, fo, errors.Errorf(errors.EngineSupport, e, e, errors.ThisFn())
}

func getLargestShape(ss ...shapes.Shape) shapes.Shape {
	var max shapes.Shape
	var maxSize int
	for _, s := range ss {
		sz := s.TotalSize()
		if sz > maxSize {
			max = s
			maxSize = sz
		}
	}
	return max
}

func checkCompatibleShape(expected shapes.Shape, others ...shapes.Shape) error {
	expLen := expected.TotalSize()
	for _, s := range others {
		if s.TotalSize() != expLen {
			return errors.Errorf(errors.ShapeMismatch, expected, s)
		}
	}
	return nil

}

func elimInnermostOutermost(a, b shapes.Shape) shapes.Shape {
	a2 := a.Clone()
	return append(a2[:len(a)-1], b[1:]...)
}
