package tensor

// public API for comparison ops

// Lt performs a elementwise less than comparison (a < b). a and b can either be float64 or *Dense.
// It returns the same Tensor type as its input.
//
// If both operands are *Dense, shape is checked first.
// Even though the underlying data may have the same size (say (2,2) vs (4,1)), if they have different shapes, it will error out.
func Lt(a, b interface{}, opts ...FuncOpt) (retVal Tensor, err error) {
	boolT := !parseAsFloat64(opts...)

	ad, atok := a.(*Dense)
	bd, btok := b.(*Dense)
	switch {
	case adok && bdok:
		return a.ltDD(bd, opts...)
	case adok && !bdok:
		return a.ltDS(b, opts...)
	case !adok && bdok:
		return b.gtDS(a, opts...)
	}

	panic("unreachable")
}

// Gt performs a elementwise greater than comparison (a > b). a and b can either be float64 or *Dense.
// It returns the same Tensor type as its input.
//
// If both operands are *Dense, shape is checked first.
// Even though the underlying data may have the same size (say (2,2) vs (4,1)), if they have different shapes, it will error out.
func Gt(a, b interface{}, opts ...FuncOpt) (retVal Tensor, err error) {
	boolT := !parseAsFloat64(opts...)

	ad, atok := a.(*Dense)
	bd, btok := b.(*Dense)
	switch {
	case adok && bdok:
		return a.gtDD(bd, opts...)
	case adok && !bdok:
		return a.gtDS(b, opts...)
	case !adok && bdok:
		return b.ltDS(a, opts...)
	}

	panic("unreachable")
}

// Lte performs a elementwise less than eq comparison (a <= b). a and b can either be float64 or *Dense.
// It returns the same Tensor type as its input.
//
// If both operands are *Dense, shape is checked first.
// Even though the underlying data may have the same size (say (2,2) vs (4,1)), if they have different shapes, it will error out.
func Lte(a, b interface{}, opts ...FuncOpt) (retVal Tensor, err error) {
	boolT := !parseAsFloat64(opts...)

	ad, atok := a.(*Dense)
	bd, btok := b.(*Dense)
	switch {
	case adok && bdok:
		return a.lteDD(bd, opts...)
	case adok && !bdok:
		return a.lteDS(b, opts...)
	case !adok && bdok:
		return b.gteDS(a, opts...)
	}

	panic("unreachable")
}

// Gte performs a elementwise greater than eq comparison (a >= b). a and b can either be float64 or *Dense.
// It returns the same Tensor type as its input.
//
// If both operands are *Dense, shape is checked first.
// Even though the underlying data may have the same size (say (2,2) vs (4,1)), if they have different shapes, it will error out.
func Gte(a, b interface{}, opts ...FuncOpt) (retVal Tensor, err error) {
	boolT := !parseAsFloat64(opts...)

	ad, atok := a.(*Dense)
	bd, btok := b.(*Dense)
	switch {
	case adok && bdok:
		return a.gteDD(bd, opts...)
	case adok && !bdok:
		return a.gteDS(b, opts...)
	case !adok && bdok:
		return b.lteDS(a, opts...)
	}

	panic("unreachable")
}

// Eq performs a elementwise equality comparison (a == b). a and b can either be float64 or *Dense.
// It returns the same Tensor type as its input.
//
// If both operands are *Dense, shape is checked first.
// Even though the underlying data may have the same size (say (2,2) vs (4,1)), if they have different shapes, it will error out.
func Eq(a, b interface{}, opts ...FuncOpt) (retVal Tensor, err error) {
	boolT := !parseAsFloat64(opts...)

	ad, atok := a.(*Dense)
	bd, btok := b.(*Dense)
	switch {
	case adok && bdok:
		return a.eqDD(bd, opts...)
	case adok && !bdok:
		return a.eqDS(b, opts...)
	case !adok && bdok:
		return b.eqDS(a, opts...)
	}

	panic("unreachable")
}
