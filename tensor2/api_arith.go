package tensor

// exported API for arithmetics and the stupidly crazy amount of overloaded semantics
// Add performs a pointwise a+b. a and b can either be float64 or Tensor
//
// If both operands are Tensor, shape is checked first.
// Even though the underlying data may have the same size (say (2,2) vs (4,1)), if they have different shapes, it will error out.
//
// If the Unsafe flag is passed in, the data of the first tensor will be overwritten

func Add(a, b interface{}, opts ...FuncOpt) (retVal Tensor, err error) {
	ad, adok := a.(*Dense)
	bd, bdok := b.(*Dense)

	switch {
	case adok && bdok:
		return addDD(ad, bd, opts...)
	case adok && !bdok:
		return addDS(ad, b, opts...)
	case !adok && bdok:
		return addSD(a, bd, opts...)
	}
	panic("Unreachable")
}

func Sub(a, b interface{}, opts ...FuncOpt) (retVal Tensor, err error) {
	ad, adok := a.(*Dense)
	bd, bdok := b.(*Dense)

	switch {
	case adok && bdok:
		return subDD(ad, bd, opts...)
	case adok && !bdok:
		return subDS(ad, b, opts...)
	case !adok && bdok:
		return subSD(a, bd, opts...)
	}
	panic("Unreachable")
}

func Mul(a, b interface{}, opts ...FuncOpt) (retVal Tensor, err error) {
	ad, adok := a.(*Dense)
	bd, bdok := b.(*Dense)

	switch {
	case adok && bdok:
		return mulDD(ad, bd, opts...)
	case adok && !bdok:
		return mulDS(ad, b, opts...)
	case !adok && bdok:
		return mulSD(a, bd, opts...)
	}
	panic("Unreachable")
}

func Div(a, b interface{}, opts ...FuncOpt) (retVal Tensor, err error) {
	ad, adok := a.(*Dense)
	bd, bdok := b.(*Dense)

	switch {
	case adok && bdok:
		return divDD(ad, bd, opts...)
	case adok && !bdok:
		return divDS(ad, b, opts...)
	case !adok && bdok:
		return divSD(a, bd, opts...)
	}
	panic("Unreachable")
}

func Pow(a, b interface{}, opts ...FuncOpt) (retVal Tensor, err error) {
	ad, adok := a.(*Dense)
	bd, bdok := b.(*Dense)

	switch {
	case adok && bdok:
		return powDD(ad, bd, opts...)
	case adok && !bdok:
		return powDS(ad, b, opts...)
	case !adok && bdok:
		return powSD(a, bd, opts...)
	}
	panic("Unreachable")
}
