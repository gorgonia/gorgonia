package main

import "text/template"

const binopRaw = `// {{.Method}} implements tensor.{{.Method}}er. It does not support safe or increment operation options and will return an error if those options are passed in
func (e *Engine) {{.Method}}(a tensor.Tensor, b tensor.Tensor, opts ...tensor.FuncOpt) (retVal tensor.Tensor, err error) {
	name := constructBinName2(a, b, "{{.ScalarMethod | lower}}")

	if err = binaryCheck(a, b); err != nil {
		return nil, errors.Wrap(err, "Basic checks failed for {{.Method}}")
	}

	var mem, memB cu.DevicePtr
	var size int64
	if mem, size, retVal, err = e.binopMem(a, opts...); err != nil{
		return nil, errors.Wrap(err, "Unable to perform {{.Method}}")
	}
	memB = cu.DevicePtr(b.Uintptr())

	debug.Logf("CUDADO %q, Mem: %v MemB: %v size %v", name, mem, memB, size)
	debug.Logf("LaunchKernel Params. mem: %v. Size %v", mem, size)
	if err = e.Call(name, int(size), unsafe.Pointer(&mem), unsafe.Pointer(&memB), unsafe.Pointer(&size)); err !=nil{
		err = errors.Wrap(err, "Unable to perform engine.{{.Method}} - CUDA LaunchAndSync failed.")
	}
	return
}

// {{.ScalarMethod}}Scalar implements tensor.{{.Method}}er. It does not support safe or increment operation options and will return an error if those options are passed in
func (e *Engine) {{.ScalarMethod}}Scalar(a tensor.Tensor, b interface{}, leftTensor bool, opts ...tensor.FuncOpt) (retVal tensor.Tensor, err error) {
	name := constructBinName1(a, leftTensor, "{{.ScalarMethod | lower}}")

	var bMem tensor.Memory
	var ok bool
	if bMem, ok = b.(tensor.Memory); !ok {
		return nil, errors.Errorf("b has to be a tensor.Memory. Got %T instead", b)
	}

	if err = unaryCheck(a); err != nil {
		return nil, errors.Wrap(err, "Basic checks failed for {{.ScalarMethod}}Scalar")
	}


	var mem, memB cu.DevicePtr
	var size int64
	if mem, size, retVal, err = e.binopMem(a, opts...); err != nil{
		return nil, errors.Wrap(err, "Unable to perform {{.Method}}")
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
}
`

var (
	binopTmpl *template.Template
)

func init() {
	binopTmpl = template.Must(template.New("binop").Funcs(funcmap).Parse(binopRaw))
}
