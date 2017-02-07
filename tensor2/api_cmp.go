package tensor

func ElEq(a, b interface{}, opts ...FuncOpt) (retVal Tensor, err error) {
	ad, adok := a.(*Dense)
	bd, bdok := b.(*Dense)

	switch {
	case adok && bdok:
		return elEqDD(ad, bd, opts...)
	case adok && !bdok:
		return elEqDS(ad, b, opts...)
	case !adok && bdok:
		return elEqSD(a, bd, opts...)
	}
	panic("Unreachable")
}

func Gt(a, b interface{}, opts ...FuncOpt) (retVal Tensor, err error) {
	ad, adok := a.(*Dense)
	bd, bdok := b.(*Dense)

	switch {
	case adok && bdok:
		return gtDD(ad, bd, opts...)
	case adok && !bdok:
		return gtDS(ad, b, opts...)
	case !adok && bdok:
		return gtSD(a, bd, opts...)
	}
	panic("Unreachable")
}

func Gte(a, b interface{}, opts ...FuncOpt) (retVal Tensor, err error) {
	ad, adok := a.(*Dense)
	bd, bdok := b.(*Dense)

	switch {
	case adok && bdok:
		return gteDD(ad, bd, opts...)
	case adok && !bdok:
		return gteDS(ad, b, opts...)
	case !adok && bdok:
		return gteSD(a, bd, opts...)
	}
	panic("Unreachable")
}

func Lt(a, b interface{}, opts ...FuncOpt) (retVal Tensor, err error) {
	ad, adok := a.(*Dense)
	bd, bdok := b.(*Dense)

	switch {
	case adok && bdok:
		return ltDD(ad, bd, opts...)
	case adok && !bdok:
		return ltDS(ad, b, opts...)
	case !adok && bdok:
		return ltSD(a, bd, opts...)
	}
	panic("Unreachable")
}

func Lte(a, b interface{}, opts ...FuncOpt) (retVal Tensor, err error) {
	ad, adok := a.(*Dense)
	bd, bdok := b.(*Dense)

	switch {
	case adok && bdok:
		return lteDD(ad, bd, opts...)
	case adok && !bdok:
		return lteDS(ad, b, opts...)
	case !adok && bdok:
		return lteSD(a, bd, opts...)
	}
	panic("Unreachable")
}
