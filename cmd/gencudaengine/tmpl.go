package main

import "text/template"

const binopRaw = `// {{.Method}} implements tensor.{{.Method}}er. It does not support safe or increment operation options and will return an error if those options are passed in.
func (e *Engine[DT,T]) {{.Method}}(ctx context.Context, a, b, retVal T, opts ...tensor.FuncOpt) (err error) {
	name := constructBinName2(a, b, "{{.ScalarMethod | lower}}")

	if err = binaryCheck[DT](a, b); err != nil {
		return errors.Wrap(err, "Basic checks failed for {{.Method}}")
	}
        mem, memB, size := e.opMem(a, b, retVal)

	debug.Logf("CUDADO %q, Mem: %v MemB: %v size %v", name, mem, memB, size)
	debug.Logf("LaunchKernel Params. mem: %v. Size %v", mem, size)
	if err = e.Call(name, int(size), unsafe.Pointer(&mem), unsafe.Pointer(&memB), unsafe.Pointer(&size)); err !=nil{
		err = errors.Wrap(err, "Unable to perform engine.{{.Method}} - CUDA LaunchAndSync failed.")
	}
	return
}

// {{.ScalarMethod}}Scalar implements tensor.{{.Method}}er. It does not support safe or increment operation options and will return an error if those options are passed in.
func (e *Engine[DT,T]) {{.ScalarMethod}}Scalar(ctx context.Context, a T, b DT, retVal T,  leftTensor, toIncr bool) (err error) {
return errors.NYI()
/*
	name := constructBinName1(a, leftTensor, "{{.ScalarMethod | lower}}")

	var bMem tensor.Memory
	var ok bool
	if bMem, ok = b.(tensor.Memory); !ok {
		return errors.Errorf("b has to be a tensor.Memory. Got %T instead", b)
	}

	if err = unaryCheck[DT](a); err != nil {
		return errors.Wrap(err, "Basic checks failed for {{.ScalarMethod}}Scalar")
	}


	var mem, memB cu.DevicePtr
	var size int64
	if mem, size, retVal, err = e.opMem(a, opts...); err != nil{
		return errors.Wrap(err, "Unable to perform {{.Method}}")
	}
	memB = cu.DevicePtr(bMem.Uintptr())
	if !leftTensor {
		mem, memB = memB, mem
	}

	debug.Logf("CUDADO %q, Mem: %v size %v, args %v", name, mem, size)
	debug.Logf("LaunchKernel Params. mem: %v. Size %v", mem, size)
	if err = e.Call(name, int(size), unsafe.Pointer(&mem), unsafe.Pointer(&memB), unsafe.Pointer(&size)); err != nil{
		err = errors.Wrap(err, "Unable to perform engine.{{.ScalarMethod}} - CUDA LaunchAndSync failed.")
	}
	return
*/
}

`

const unopRaw = `// {{.Method}} implements tensor.{{.Method}}er. It does not support safe or increment options and will return an error if those options are passed in.
func (e *Engine[DT,T]) {{.Method}}(ctx context.Context, a T, retVal T) (err error) {
	name := constructUnOpName(a, "{{.KernelName}}")
	mem, _, size := e.opMem(a, retVal)

	debug.Logf("CUDADO %q, Mem: %v size %v, args %v", name, mem, size)
	debug.Logf("LaunchKernel Params. mem: %v. Size %v", mem, size)
	if err = e.Call(name, int(size), unsafe.Pointer(&mem), unsafe.Pointer(&size)); err != nil {
		err = errors.Wrap(err, "Unable to perform engine.Neg - CUDA LaunchAndSync failed")
	}
	return
}

`

var (
	binopTmpl *template.Template
	unopTmpl  *template.Template
)

func init() {
	binopTmpl = template.Must(template.New("binop").Funcs(funcmap).Parse(binopRaw))
	unopTmpl = template.Must(template.New("unop").Funcs(funcmap).Parse(unopRaw))
}
