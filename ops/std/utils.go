package stdops

import (
	"reflect"
	"runtime"

	"gorgonia.org/gorgonia/internal/errors"
	"gorgonia.org/shapes"
	"gorgonia.org/tensor"
)

// funcName returns the name of the function.
func funcName(f interface{}) string {
	val := reflect.ValueOf(f)
	if val.Kind() != reflect.Func {
		return "Not a function"
	}

	//TODO(go1.18) - pc := uintptr(val.UnsafePointer())
	pc := val.Pointer()
	return runtime.FuncForPC(pc).Name()
}

func getEngine(ts ...tensor.Engineer) tensor.Engine {
	// TODO: get highest capability engine
	for _, t := range ts {
		if e := t.Engine(); e != nil {
			return e
		}
	}
	return nil
}

func handleFuncOpts[DT any, T tensor.Tensor[DT, T]](e Engine, t T, expShape shapes.Shape, opts ...FuncOpt) (retVal T, fo Option, err error) {
	switch e := e.(type) {
	case tensor.SpecializedFuncOptHandler[DT, T]:
		return e.HandleFuncOptsSpecialized(t, expShape, opts...)
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
