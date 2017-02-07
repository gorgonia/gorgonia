package tensor

// public API for comparison ops

// Lt performs a elementwise less than comparison (a < b). a and b can either be float64 or *Dense.
// It returns the same Tensor type as its input.
//
// If both operands are *Dense, shape is checked first.
// Even though the underlying data may have the same size (say (2,2) vs (4,1)), if they have different shapes, it will error out.
func Lt(a, b interface{}, opts ...FuncOpt) (retVal Tensor, err error) {
	ad, adok := a.(*Dense)
	bd, bdok := b.(*Dense)
	switch {
	case adok && bdok:
		return ad.ltDD(bd, opts...)
	case adok && !bdok:
		return ad.ltDS(b, opts...)
	case !adok && bdok:
		return bd.gtDS(a, opts...)
	}

	panic("unreachable")
}

// Gt performs a elementwise greater than comparison (a > b). a and b can either be float64 or *Dense.
// It returns the same Tensor type as its input.
//
// If both operands are *Dense, shape is checked first.
// Even though the underlying data may have the same size (say (2,2) vs (4,1)), if they have different shapes, it will error out.
func Gt(a, b interface{}, opts ...FuncOpt) (retVal Tensor, err error) {
	ad, adok := a.(*Dense)
	bd, bdok := b.(*Dense)
	switch {
	case adok && bdok:
		return ad.gtDD(bd, opts...)
	case adok && !bdok:
		return ad.gtDS(b, opts...)
	case !adok && bdok:
		return bd.ltDS(a, opts...)
	}

	panic("unreachable")
}

// Lte performs a elementwise less than eq comparison (a <= b). a and b can either be float64 or *Dense.
// It returns the same Tensor type as its input.
//
// If both operands are *Dense, shape is checked first.
// Even though the underlying data may have the same size (say (2,2) vs (4,1)), if they have different shapes, it will error out.
func Lte(a, b interface{}, opts ...FuncOpt) (retVal Tensor, err error) {
	ad, adok := a.(*Dense)
	bd, bdok := b.(*Dense)
	switch {
	case adok && bdok:
		return ad.lteDD(bd, opts...)
	case adok && !bdok:
		return ad.lteDS(b, opts...)
	case !adok && bdok:
		return bd.gteDS(a, opts...)
	}

	panic("unreachable")
}

// Gte performs a elementwise greater than eq comparison (a >= b). a and b can either be float64 or *Dense.
// It returns the same Tensor type as its input.
//
// If both operands are *Dense, shape is checked first.
// Even though the underlying data may have the same size (say (2,2) vs (4,1)), if they have different shapes, it will error out.
func Gte(a, b interface{}, opts ...FuncOpt) (retVal Tensor, err error) {
	ad, adok := a.(*Dense)
	bd, bdok := b.(*Dense)
	switch {
	case adok && bdok:
		return ad.gteDD(bd, opts...)
	case adok && !bdok:
		return ad.gteDS(b, opts...)
	case !adok && bdok:
		return bd.lteDS(a, opts...)
	}

	panic("unreachable")
}

// ElEq performs a elementwise equality comparison (a == b). a and b can either be float64 or *Dense.
// It returns the same Tensor type as its input.
//
// If both operands are *Dense, shape is checked first.
// Even though the underlying data may have the same size (say (2,2) vs (4,1)), if they have different shapes, it will error out.
func ElEq(a, b interface{}, opts ...FuncOpt) (retVal Tensor, err error) {
	ad, adok := a.(*Dense)
	bd, bdok := b.(*Dense)
	switch {
	case adok && bdok:
		return ad.eqDD(bd, opts...)
	case adok && !bdok:
		return ad.eqDS(b, opts...)
	case !adok && bdok:
		return bd.eqDS(a, opts...)
	}

	panic("unreachable")
}

// ElNe performs a elementwise equality comparison (a != b). a and b can either be float64 or *Dense.
// It returns the same Tensor type as its input.
//
// If both operands are *Dense, shape is checked first.
// Even though the underlying data may have the same size (say (2,2) vs (4,1)), if they have different shapes, it will error out.
func ElNe(a, b interface{}, opts ...FuncOpt) (retVal Tensor, err error) {
	ad, adok := a.(*Dense)
	bd, bdok := b.(*Dense)
	switch {
	case adok && bdok:
		return ad.neDD(bd, opts...)
	case adok && !bdok:
		return ad.neDS(b, opts...)
	case !adok && bdok:
		return bd.neDS(a, opts...)
	}

	panic("unreachable")
}
