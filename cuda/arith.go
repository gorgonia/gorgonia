package cuda

import (
	"fmt"
	"unsafe"

	"github.com/pkg/errors"
	"gorgonia.org/cu"
	"gorgonia.org/tensor"
)

func (e *Engine) Add(a tensor.Tensor, b tensor.Tensor, opts ...tensor.FuncOpt) (retVal tensor.Tensor, err error) {
	dt := a.Dtype()
	as := a.Shape()
	bs := b.Shape()
	var name string
	switch {
	case as.IsScalar() && bs.IsScalar():
		name = fmt.Sprintf("%v.add_ss_f%d", elemBinOpMod, int(dt.Size()*8))
	case as.IsScalar() && !bs.IsScalar():
		name = fmt.Sprintf("%v.add_sv_f%d", elemBinOpMod, int(dt.Size()*8))
	case !as.IsScalar() && bs.IsScalar():
		name = fmt.Sprintf("%v.add_vs_f%d", elemBinOpMod, int(dt.Size()*8))
	default:
		name = fmt.Sprintf("%v.add_vv_f%d", elemBinOpMod, int(dt.Size()*8))
	}

	if !e.HasFunc(name) {
		return nil, errors.Errorf("Unable to perform Add(). The tensor engine does not have the function %q", name)
	}

	if err = binaryCheck(a, b); err != nil {
		return nil, errors.Wrap(err, "Basic checks failed for Add")
	}

	var reuse tensor.DenseTensor
	var safe, toReuse bool
	if reuse, safe, toReuse, _, _, err = handleFuncOpts(a.Shape(), a.Dtype(), a.DataOrder(), true, opts...); err != nil {
		return nil, errors.Wrap(err, "Unable to handle funcOpts")
	}

	var mem, memB cu.DevicePtr
	var size int64

	switch {
	case toReuse:
		mem = cu.DevicePtr(reuse.Uintptr())
		memA := cu.DevicePtr(a.Uintptr())
		memSize := int64(a.MemSize())
		e.Memcpy(mem, memA)

		size = int64(logicalSize(reuse.Shape()))
		retVal = reuse
	case !safe:
		mem = cu.DevicePtr(a.Uintptr())
		retVal = reuse
		size = int64(logicalSize(a.Shape()))
	default:
		// error
	}

	fn := e.f[name]
	gridDimX, gridDimY, gridDimZ, blockDimX, blockDimY, blockDimZ := e.ElemGridSize(int(size))
	args := []unsafe.Pointer{
		unsafe.Pointer(&mem),
		unsafe.Pointer(&memB),
		unsafe.Pointer(&size),
	}
	ctx.LaunchAndSync(fn, gridDimX, gridDimY, gridDimZ, blockDimX, blockDimY, blockDimZ, 0, cu.NoStream, args)
	return
}

func (e *Engine) AddScalar(a tensor.Tensor, b interface{}, leftTensor bool, opts ...tensor.FuncOpt) (tensor.Tensor, error) {
	dt := a.Dtype()
	var name string
	if leftTensor {
		name = fmt.Sprintf("%v.add_vs_f%d", elemBinOpMod, int(dt.Size()*8))
	} else {
		name = fmt.Sprintf("%v.add_ss_f%d", elemBinOpMod, int(dt.Size()*8))
	}

	if err = unaryCheck(a); err != nil {
		return nil, errors.Wrap(err, "Basic checks failed for AddScalar")
	}

	var reuse tensor.DenseTensor
	var safe, toReuse bool
	if reuse, safe, toReuse, _, _, err = handleFuncOpts(a.Shape(), a.Dtype(), a.DataOrder(), true, opts...); err != nil {
		return nil, errors.Wrap(err, "Unable to handle funcOpts")
	}

	var mem, memB cu.DevicePtr
	var size int64

	switch {
	case toReuse:
		mem = cu.DevicePtr(reuse.Uintptr())
		memA := cu.DevicePtr(a.Uintptr())
		memSize := int64(a.MemSize())
		e.Memcpy(mem, memA)

		size = int64(logicalSize(reuse.Shape()))
		retVal = reuse
	case !safe:
		mem = cu.DevicePtr(a.Uintptr())
		retVal = reuse
		size = int64(logicalSize(a.Shape()))
	default:
		// error
	}

	fn := e.f[name]
	gridDimX, gridDimY, gridDimZ, blockDimX, blockDimY, blockDimZ := e.ElemGridSize(int(size))
	args := []unsafe.Pointer{
		unsafe.Pointer(&mem),
		unsafe.Pointer(&memB),
		unsafe.Pointer(&size),
	}
	ctx.LaunchAndSync(fn, gridDimX, gridDimY, gridDimZ, blockDimX, blockDimY, blockDimZ, 0, cu.NoStream, args)
	return
}

func (e *Engine) Sub(a tensor.Tensor, b tensor.Tensor, opts ...tensor.FuncOpt) (retVal tensor.Tensor, err error) {
	dt := a.Dtype()
	as := a.Shape()
	bs := b.Shape()
	var name string
	switch {
	case as.IsScalar() && bs.IsScalar():
		name = fmt.Sprintf("%v.sub_ss_f%d", elemBinOpMod, int(dt.Size()*8))
	case as.IsScalar() && !bs.IsScalar():
		name = fmt.Sprintf("%v.sub_sv_f%d", elemBinOpMod, int(dt.Size()*8))
	case !as.IsScalar() && bs.IsScalar():
		name = fmt.Sprintf("%v.sub_vs_f%d", elemBinOpMod, int(dt.Size()*8))
	default:
		name = fmt.Sprintf("%v.sub_vv_f%d", elemBinOpMod, int(dt.Size()*8))
	}

	if !e.HasFunc(name) {
		return nil, errors.Errorf("Unable to perform Sub(). The tensor engine does not have the function %q", name)
	}

	if err = binaryCheck(a, b); err != nil {
		return nil, errors.Wrap(err, "Basic checks failed for Sub")
	}

	var reuse tensor.DenseTensor
	var safe, toReuse bool
	if reuse, safe, toReuse, _, _, err = handleFuncOpts(a.Shape(), a.Dtype(), a.DataOrder(), true, opts...); err != nil {
		return nil, errors.Wrap(err, "Unable to handle funcOpts")
	}

	var mem, memB cu.DevicePtr
	var size int64

	switch {
	case toReuse:
		mem = cu.DevicePtr(reuse.Uintptr())
		memA := cu.DevicePtr(a.Uintptr())
		memSize := int64(a.MemSize())
		e.Memcpy(mem, memA)

		size = int64(logicalSize(reuse.Shape()))
		retVal = reuse
	case !safe:
		mem = cu.DevicePtr(a.Uintptr())
		retVal = reuse
		size = int64(logicalSize(a.Shape()))
	default:
		// error
	}

	fn := e.f[name]
	gridDimX, gridDimY, gridDimZ, blockDimX, blockDimY, blockDimZ := e.ElemGridSize(int(size))
	args := []unsafe.Pointer{
		unsafe.Pointer(&mem),
		unsafe.Pointer(&memB),
		unsafe.Pointer(&size),
	}
	ctx.LaunchAndSync(fn, gridDimX, gridDimY, gridDimZ, blockDimX, blockDimY, blockDimZ, 0, cu.NoStream, args)
	return
}

func (e *Engine) SubScalar(a tensor.Tensor, b interface{}, leftTensor bool, opts ...tensor.FuncOpt) (tensor.Tensor, error) {
	dt := a.Dtype()
	var name string
	if leftTensor {
		name = fmt.Sprintf("%v.sub_vs_f%d", elemBinOpMod, int(dt.Size()*8))
	} else {
		name = fmt.Sprintf("%v.sub_ss_f%d", elemBinOpMod, int(dt.Size()*8))
	}

	if err = unaryCheck(a); err != nil {
		return nil, errors.Wrap(err, "Basic checks failed for SubScalar")
	}

	var reuse tensor.DenseTensor
	var safe, toReuse bool
	if reuse, safe, toReuse, _, _, err = handleFuncOpts(a.Shape(), a.Dtype(), a.DataOrder(), true, opts...); err != nil {
		return nil, errors.Wrap(err, "Unable to handle funcOpts")
	}

	var mem, memB cu.DevicePtr
	var size int64

	switch {
	case toReuse:
		mem = cu.DevicePtr(reuse.Uintptr())
		memA := cu.DevicePtr(a.Uintptr())
		memSize := int64(a.MemSize())
		e.Memcpy(mem, memA)

		size = int64(logicalSize(reuse.Shape()))
		retVal = reuse
	case !safe:
		mem = cu.DevicePtr(a.Uintptr())
		retVal = reuse
		size = int64(logicalSize(a.Shape()))
	default:
		// error
	}

	fn := e.f[name]
	gridDimX, gridDimY, gridDimZ, blockDimX, blockDimY, blockDimZ := e.ElemGridSize(int(size))
	args := []unsafe.Pointer{
		unsafe.Pointer(&mem),
		unsafe.Pointer(&memB),
		unsafe.Pointer(&size),
	}
	ctx.LaunchAndSync(fn, gridDimX, gridDimY, gridDimZ, blockDimX, blockDimY, blockDimZ, 0, cu.NoStream, args)
	return
}

func (e *Engine) Mul(a tensor.Tensor, b tensor.Tensor, opts ...tensor.FuncOpt) (retVal tensor.Tensor, err error) {
	dt := a.Dtype()
	as := a.Shape()
	bs := b.Shape()
	var name string
	switch {
	case as.IsScalar() && bs.IsScalar():
		name = fmt.Sprintf("%v.mul_ss_f%d", elemBinOpMod, int(dt.Size()*8))
	case as.IsScalar() && !bs.IsScalar():
		name = fmt.Sprintf("%v.mul_sv_f%d", elemBinOpMod, int(dt.Size()*8))
	case !as.IsScalar() && bs.IsScalar():
		name = fmt.Sprintf("%v.mul_vs_f%d", elemBinOpMod, int(dt.Size()*8))
	default:
		name = fmt.Sprintf("%v.mul_vv_f%d", elemBinOpMod, int(dt.Size()*8))
	}

	if !e.HasFunc(name) {
		return nil, errors.Errorf("Unable to perform Mul(). The tensor engine does not have the function %q", name)
	}

	if err = binaryCheck(a, b); err != nil {
		return nil, errors.Wrap(err, "Basic checks failed for Mul")
	}

	var reuse tensor.DenseTensor
	var safe, toReuse bool
	if reuse, safe, toReuse, _, _, err = handleFuncOpts(a.Shape(), a.Dtype(), a.DataOrder(), true, opts...); err != nil {
		return nil, errors.Wrap(err, "Unable to handle funcOpts")
	}

	var mem, memB cu.DevicePtr
	var size int64

	switch {
	case toReuse:
		mem = cu.DevicePtr(reuse.Uintptr())
		memA := cu.DevicePtr(a.Uintptr())
		memSize := int64(a.MemSize())
		e.Memcpy(mem, memA)

		size = int64(logicalSize(reuse.Shape()))
		retVal = reuse
	case !safe:
		mem = cu.DevicePtr(a.Uintptr())
		retVal = reuse
		size = int64(logicalSize(a.Shape()))
	default:
		// error
	}

	fn := e.f[name]
	gridDimX, gridDimY, gridDimZ, blockDimX, blockDimY, blockDimZ := e.ElemGridSize(int(size))
	args := []unsafe.Pointer{
		unsafe.Pointer(&mem),
		unsafe.Pointer(&memB),
		unsafe.Pointer(&size),
	}
	ctx.LaunchAndSync(fn, gridDimX, gridDimY, gridDimZ, blockDimX, blockDimY, blockDimZ, 0, cu.NoStream, args)
	return
}

func (e *Engine) MulScalar(a tensor.Tensor, b interface{}, leftTensor bool, opts ...tensor.FuncOpt) (tensor.Tensor, error) {
	dt := a.Dtype()
	var name string
	if leftTensor {
		name = fmt.Sprintf("%v.mul_vs_f%d", elemBinOpMod, int(dt.Size()*8))
	} else {
		name = fmt.Sprintf("%v.mul_ss_f%d", elemBinOpMod, int(dt.Size()*8))
	}

	if err = unaryCheck(a); err != nil {
		return nil, errors.Wrap(err, "Basic checks failed for MulScalar")
	}

	var reuse tensor.DenseTensor
	var safe, toReuse bool
	if reuse, safe, toReuse, _, _, err = handleFuncOpts(a.Shape(), a.Dtype(), a.DataOrder(), true, opts...); err != nil {
		return nil, errors.Wrap(err, "Unable to handle funcOpts")
	}

	var mem, memB cu.DevicePtr
	var size int64

	switch {
	case toReuse:
		mem = cu.DevicePtr(reuse.Uintptr())
		memA := cu.DevicePtr(a.Uintptr())
		memSize := int64(a.MemSize())
		e.Memcpy(mem, memA)

		size = int64(logicalSize(reuse.Shape()))
		retVal = reuse
	case !safe:
		mem = cu.DevicePtr(a.Uintptr())
		retVal = reuse
		size = int64(logicalSize(a.Shape()))
	default:
		// error
	}

	fn := e.f[name]
	gridDimX, gridDimY, gridDimZ, blockDimX, blockDimY, blockDimZ := e.ElemGridSize(int(size))
	args := []unsafe.Pointer{
		unsafe.Pointer(&mem),
		unsafe.Pointer(&memB),
		unsafe.Pointer(&size),
	}
	ctx.LaunchAndSync(fn, gridDimX, gridDimY, gridDimZ, blockDimX, blockDimY, blockDimZ, 0, cu.NoStream, args)
	return
}

func (e *Engine) Div(a tensor.Tensor, b tensor.Tensor, opts ...tensor.FuncOpt) (retVal tensor.Tensor, err error) {
	dt := a.Dtype()
	as := a.Shape()
	bs := b.Shape()
	var name string
	switch {
	case as.IsScalar() && bs.IsScalar():
		name = fmt.Sprintf("%v.div_ss_f%d", elemBinOpMod, int(dt.Size()*8))
	case as.IsScalar() && !bs.IsScalar():
		name = fmt.Sprintf("%v.div_sv_f%d", elemBinOpMod, int(dt.Size()*8))
	case !as.IsScalar() && bs.IsScalar():
		name = fmt.Sprintf("%v.div_vs_f%d", elemBinOpMod, int(dt.Size()*8))
	default:
		name = fmt.Sprintf("%v.div_vv_f%d", elemBinOpMod, int(dt.Size()*8))
	}

	if !e.HasFunc(name) {
		return nil, errors.Errorf("Unable to perform Div(). The tensor engine does not have the function %q", name)
	}

	if err = binaryCheck(a, b); err != nil {
		return nil, errors.Wrap(err, "Basic checks failed for Div")
	}

	var reuse tensor.DenseTensor
	var safe, toReuse bool
	if reuse, safe, toReuse, _, _, err = handleFuncOpts(a.Shape(), a.Dtype(), a.DataOrder(), true, opts...); err != nil {
		return nil, errors.Wrap(err, "Unable to handle funcOpts")
	}

	var mem, memB cu.DevicePtr
	var size int64

	switch {
	case toReuse:
		mem = cu.DevicePtr(reuse.Uintptr())
		memA := cu.DevicePtr(a.Uintptr())
		memSize := int64(a.MemSize())
		e.Memcpy(mem, memA)

		size = int64(logicalSize(reuse.Shape()))
		retVal = reuse
	case !safe:
		mem = cu.DevicePtr(a.Uintptr())
		retVal = reuse
		size = int64(logicalSize(a.Shape()))
	default:
		// error
	}

	fn := e.f[name]
	gridDimX, gridDimY, gridDimZ, blockDimX, blockDimY, blockDimZ := e.ElemGridSize(int(size))
	args := []unsafe.Pointer{
		unsafe.Pointer(&mem),
		unsafe.Pointer(&memB),
		unsafe.Pointer(&size),
	}
	ctx.LaunchAndSync(fn, gridDimX, gridDimY, gridDimZ, blockDimX, blockDimY, blockDimZ, 0, cu.NoStream, args)
	return
}

func (e *Engine) DivScalar(a tensor.Tensor, b interface{}, leftTensor bool, opts ...tensor.FuncOpt) (tensor.Tensor, error) {
	dt := a.Dtype()
	var name string
	if leftTensor {
		name = fmt.Sprintf("%v.div_vs_f%d", elemBinOpMod, int(dt.Size()*8))
	} else {
		name = fmt.Sprintf("%v.div_ss_f%d", elemBinOpMod, int(dt.Size()*8))
	}

	if err = unaryCheck(a); err != nil {
		return nil, errors.Wrap(err, "Basic checks failed for DivScalar")
	}

	var reuse tensor.DenseTensor
	var safe, toReuse bool
	if reuse, safe, toReuse, _, _, err = handleFuncOpts(a.Shape(), a.Dtype(), a.DataOrder(), true, opts...); err != nil {
		return nil, errors.Wrap(err, "Unable to handle funcOpts")
	}

	var mem, memB cu.DevicePtr
	var size int64

	switch {
	case toReuse:
		mem = cu.DevicePtr(reuse.Uintptr())
		memA := cu.DevicePtr(a.Uintptr())
		memSize := int64(a.MemSize())
		e.Memcpy(mem, memA)

		size = int64(logicalSize(reuse.Shape()))
		retVal = reuse
	case !safe:
		mem = cu.DevicePtr(a.Uintptr())
		retVal = reuse
		size = int64(logicalSize(a.Shape()))
	default:
		// error
	}

	fn := e.f[name]
	gridDimX, gridDimY, gridDimZ, blockDimX, blockDimY, blockDimZ := e.ElemGridSize(int(size))
	args := []unsafe.Pointer{
		unsafe.Pointer(&mem),
		unsafe.Pointer(&memB),
		unsafe.Pointer(&size),
	}
	ctx.LaunchAndSync(fn, gridDimX, gridDimY, gridDimZ, blockDimX, blockDimY, blockDimZ, 0, cu.NoStream, args)
	return
}

func (e *Engine) Pow(a tensor.Tensor, b tensor.Tensor, opts ...tensor.FuncOpt) (retVal tensor.Tensor, err error) {
	dt := a.Dtype()
	as := a.Shape()
	bs := b.Shape()
	var name string
	switch {
	case as.IsScalar() && bs.IsScalar():
		name = fmt.Sprintf("%v.pow_ss_f%d", elemBinOpMod, int(dt.Size()*8))
	case as.IsScalar() && !bs.IsScalar():
		name = fmt.Sprintf("%v.pow_sv_f%d", elemBinOpMod, int(dt.Size()*8))
	case !as.IsScalar() && bs.IsScalar():
		name = fmt.Sprintf("%v.pow_vs_f%d", elemBinOpMod, int(dt.Size()*8))
	default:
		name = fmt.Sprintf("%v.pow_vv_f%d", elemBinOpMod, int(dt.Size()*8))
	}

	if !e.HasFunc(name) {
		return nil, errors.Errorf("Unable to perform Pow(). The tensor engine does not have the function %q", name)
	}

	if err = binaryCheck(a, b); err != nil {
		return nil, errors.Wrap(err, "Basic checks failed for Pow")
	}

	var reuse tensor.DenseTensor
	var safe, toReuse bool
	if reuse, safe, toReuse, _, _, err = handleFuncOpts(a.Shape(), a.Dtype(), a.DataOrder(), true, opts...); err != nil {
		return nil, errors.Wrap(err, "Unable to handle funcOpts")
	}

	var mem, memB cu.DevicePtr
	var size int64

	switch {
	case toReuse:
		mem = cu.DevicePtr(reuse.Uintptr())
		memA := cu.DevicePtr(a.Uintptr())
		memSize := int64(a.MemSize())
		e.Memcpy(mem, memA)

		size = int64(logicalSize(reuse.Shape()))
		retVal = reuse
	case !safe:
		mem = cu.DevicePtr(a.Uintptr())
		retVal = reuse
		size = int64(logicalSize(a.Shape()))
	default:
		// error
	}

	fn := e.f[name]
	gridDimX, gridDimY, gridDimZ, blockDimX, blockDimY, blockDimZ := e.ElemGridSize(int(size))
	args := []unsafe.Pointer{
		unsafe.Pointer(&mem),
		unsafe.Pointer(&memB),
		unsafe.Pointer(&size),
	}
	ctx.LaunchAndSync(fn, gridDimX, gridDimY, gridDimZ, blockDimX, blockDimY, blockDimZ, 0, cu.NoStream, args)
	return
}

func (e *Engine) PowScalar(a tensor.Tensor, b interface{}, leftTensor bool, opts ...tensor.FuncOpt) (tensor.Tensor, error) {
	dt := a.Dtype()
	var name string
	if leftTensor {
		name = fmt.Sprintf("%v.pow_vs_f%d", elemBinOpMod, int(dt.Size()*8))
	} else {
		name = fmt.Sprintf("%v.pow_ss_f%d", elemBinOpMod, int(dt.Size()*8))
	}

	if err = unaryCheck(a); err != nil {
		return nil, errors.Wrap(err, "Basic checks failed for PowScalar")
	}

	var reuse tensor.DenseTensor
	var safe, toReuse bool
	if reuse, safe, toReuse, _, _, err = handleFuncOpts(a.Shape(), a.Dtype(), a.DataOrder(), true, opts...); err != nil {
		return nil, errors.Wrap(err, "Unable to handle funcOpts")
	}

	var mem, memB cu.DevicePtr
	var size int64

	switch {
	case toReuse:
		mem = cu.DevicePtr(reuse.Uintptr())
		memA := cu.DevicePtr(a.Uintptr())
		memSize := int64(a.MemSize())
		e.Memcpy(mem, memA)

		size = int64(logicalSize(reuse.Shape()))
		retVal = reuse
	case !safe:
		mem = cu.DevicePtr(a.Uintptr())
		retVal = reuse
		size = int64(logicalSize(a.Shape()))
	default:
		// error
	}

	fn := e.f[name]
	gridDimX, gridDimY, gridDimZ, blockDimX, blockDimY, blockDimZ := e.ElemGridSize(int(size))
	args := []unsafe.Pointer{
		unsafe.Pointer(&mem),
		unsafe.Pointer(&memB),
		unsafe.Pointer(&size),
	}
	ctx.LaunchAndSync(fn, gridDimX, gridDimY, gridDimZ, blockDimX, blockDimY, blockDimZ, 0, cu.NoStream, args)
	return
}

func (e *Engine) Mod(a tensor.Tensor, b tensor.Tensor, opts ...tensor.FuncOpt) (retVal tensor.Tensor, err error) {
	dt := a.Dtype()
	as := a.Shape()
	bs := b.Shape()
	var name string
	switch {
	case as.IsScalar() && bs.IsScalar():
		name = fmt.Sprintf("%v.mod_ss_f%d", elemBinOpMod, int(dt.Size()*8))
	case as.IsScalar() && !bs.IsScalar():
		name = fmt.Sprintf("%v.mod_sv_f%d", elemBinOpMod, int(dt.Size()*8))
	case !as.IsScalar() && bs.IsScalar():
		name = fmt.Sprintf("%v.mod_vs_f%d", elemBinOpMod, int(dt.Size()*8))
	default:
		name = fmt.Sprintf("%v.mod_vv_f%d", elemBinOpMod, int(dt.Size()*8))
	}

	if !e.HasFunc(name) {
		return nil, errors.Errorf("Unable to perform Mod(). The tensor engine does not have the function %q", name)
	}

	if err = binaryCheck(a, b); err != nil {
		return nil, errors.Wrap(err, "Basic checks failed for Mod")
	}

	var reuse tensor.DenseTensor
	var safe, toReuse bool
	if reuse, safe, toReuse, _, _, err = handleFuncOpts(a.Shape(), a.Dtype(), a.DataOrder(), true, opts...); err != nil {
		return nil, errors.Wrap(err, "Unable to handle funcOpts")
	}

	var mem, memB cu.DevicePtr
	var size int64

	switch {
	case toReuse:
		mem = cu.DevicePtr(reuse.Uintptr())
		memA := cu.DevicePtr(a.Uintptr())
		memSize := int64(a.MemSize())
		e.Memcpy(mem, memA)

		size = int64(logicalSize(reuse.Shape()))
		retVal = reuse
	case !safe:
		mem = cu.DevicePtr(a.Uintptr())
		retVal = reuse
		size = int64(logicalSize(a.Shape()))
	default:
		// error
	}

	fn := e.f[name]
	gridDimX, gridDimY, gridDimZ, blockDimX, blockDimY, blockDimZ := e.ElemGridSize(int(size))
	args := []unsafe.Pointer{
		unsafe.Pointer(&mem),
		unsafe.Pointer(&memB),
		unsafe.Pointer(&size),
	}
	ctx.LaunchAndSync(fn, gridDimX, gridDimY, gridDimZ, blockDimX, blockDimY, blockDimZ, 0, cu.NoStream, args)
	return
}

func (e *Engine) ModScalar(a tensor.Tensor, b interface{}, leftTensor bool, opts ...tensor.FuncOpt) (tensor.Tensor, error) {
	dt := a.Dtype()
	var name string
	if leftTensor {
		name = fmt.Sprintf("%v.mod_vs_f%d", elemBinOpMod, int(dt.Size()*8))
	} else {
		name = fmt.Sprintf("%v.mod_ss_f%d", elemBinOpMod, int(dt.Size()*8))
	}

	if err = unaryCheck(a); err != nil {
		return nil, errors.Wrap(err, "Basic checks failed for ModScalar")
	}

	var reuse tensor.DenseTensor
	var safe, toReuse bool
	if reuse, safe, toReuse, _, _, err = handleFuncOpts(a.Shape(), a.Dtype(), a.DataOrder(), true, opts...); err != nil {
		return nil, errors.Wrap(err, "Unable to handle funcOpts")
	}

	var mem, memB cu.DevicePtr
	var size int64

	switch {
	case toReuse:
		mem = cu.DevicePtr(reuse.Uintptr())
		memA := cu.DevicePtr(a.Uintptr())
		memSize := int64(a.MemSize())
		e.Memcpy(mem, memA)

		size = int64(logicalSize(reuse.Shape()))
		retVal = reuse
	case !safe:
		mem = cu.DevicePtr(a.Uintptr())
		retVal = reuse
		size = int64(logicalSize(a.Shape()))
	default:
		// error
	}

	fn := e.f[name]
	gridDimX, gridDimY, gridDimZ, blockDimX, blockDimY, blockDimZ := e.ElemGridSize(int(size))
	args := []unsafe.Pointer{
		unsafe.Pointer(&mem),
		unsafe.Pointer(&memB),
		unsafe.Pointer(&size),
	}
	ctx.LaunchAndSync(fn, gridDimX, gridDimY, gridDimZ, blockDimX, blockDimY, blockDimZ, 0, cu.NoStream, args)
	return
}
