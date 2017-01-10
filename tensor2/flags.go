package tensor

// SafeOp has precedence over unsafe (it's default)
// Incr has precedence over Reuse
func parseSafeReuse(opts ...FuncOpt) (safe, incr bool, reuse Tensor) {
	safe = true
	for _, opt := range opts {
		flag, val := opt()
		switch flag {
		case SafeOp:
			if !safe {
				safe = true
			}
		case UnsafeOp:
			safe = false
		case Incr:
			incr = true
			reuse = val.(Tensor)
		case Reuse:
			if reuse == nil {
				reuse = val.(Tensor)
			}
		}
	}
	return
}

func parseSafe(opts ...FuncOpt) bool {
	for _, opt := range opts {
		if flag, _ := opt(); flag == SafeOp {
			return true
		}
	}
	return false
}

func parseUnsafe(opts ...FuncOpt) bool {
	for _, opt := range opts {
		if flag, _ := opt(); flag == UnsafeOp {
			return true
		}
	}
	return false
}

func parseReuseIncr(opts ...FuncOpt) (reuse, incr Tensor) {
	for _, opt := range opts {
		flag, iface := opt()
		switch flag {
		case Reuse:
			reuse = iface.(Tensor)
		case Incr:
			incr = iface.(Tensor)
		}
	}
	return
}

func parseAsFloat64(opts ...FuncOpt) bool {
	for _, opt := range opts {
		if flag, _ := opt(); flag == AsTensorF64 || flag == AsSame {
			return true
		}
	}
	return false
}
