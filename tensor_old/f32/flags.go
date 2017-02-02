package tensorf32

import "github.com/chewxy/gorgonia/tensor/types"

// SafeOp has precedence over unsafe (it's default)
// Incr has precedence over Reuse
func parseSafeReuse(opts ...types.FuncOpt) (safe, incr bool, reuse *Tensor) {
	safe = true
	for _, opt := range opts {
		flag, val := opt()
		switch flag {
		case types.SafeOp:
			if !safe {
				safe = true
			}
		case types.UnsafeOp:
			safe = false
		case types.Incr:
			incr = true
			reuse = val.(*Tensor)
		case types.Reuse:
			if reuse == nil {
				reuse = val.(*Tensor)
			}
		}
	}
	return
}

func parseSafe(opts ...types.FuncOpt) bool {
	for _, opt := range opts {
		if flag, _ := opt(); flag == types.SafeOp {
			return true
		}
	}
	return false
}

func parseUnsafe(opts ...types.FuncOpt) bool {
	for _, opt := range opts {
		if flag, _ := opt(); flag == types.UnsafeOp {
			return true
		}
	}
	return false
}

func parseReuseIncr(opts ...types.FuncOpt) (reuse, incr *Tensor) {
	for _, opt := range opts {
		flag, iface := opt()
		switch flag {
		case types.Reuse:
			reuse = iface.(*Tensor)
		case types.Incr:
			incr = iface.(*Tensor)
		}
	}
	return
}

func parseAsFloat32(opts ...types.FuncOpt) bool {
	for _, opt := range opts {
		if flag, _ := opt(); flag == types.AsTensorF32 || flag == types.AsSame {
			return true
		}
	}
	return false
}
