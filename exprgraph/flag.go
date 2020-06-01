package exprgraph

// flag is a flag to identify what kind of node it is
// The default flag is:
//	is an expression (isStmt = 0)
//	is not a constant value (isConst = 0)
//	is mutable (isImmutable = 0)
// 	is not a argument (isArg = 0)
//	is not a root node (isRoot = 0)
// 	is deterministic (isRandom = 0)
//
// Despite having 256 possible combinations, there are only very few valid states. See the isValid() method
type flag byte

const (
	isStmt      flag = 1 << iota // does this node represent a statement?
	isConst                      // does this node represent a constant value?
	isImmutable                  // does this node's Op represent an immutable (in-place) value?
	isArg                        // is this node an argument node (i.e. Op == nil)
	isRoot                       // is this node the root node?
	isRandom                     // does this node represent a non-determinism?
)

// there are very few valid states despite the combinatorial combination
func (f flag) isValid() bool {
	isStmt := (f&isStmt == 1)
	isConst := (f&isConst == 1)
	isImmutable := (f&isImmutable == 1)
	isArg := (f&isArg == 1)
	isRoot := (f&isRoot == 1)
	isRandom := (f&isRandom == 1)
	switch {
	case isStmt:
		// statements cannot be anything else other than statements.
		if isConst || isImmutable || isArg || isRandom || isRoot {
			return false
		}
		return true
	case isConst:
		if isStmt || isArg || isRandom {
			return false
		}
		return true
	case isImmutable:
		if isStmt || isConst || isArg || isRandom {
			return false
		}
		return true
	case isArg:
		if isStmt || isConst || isRandom {
			return false
		}
		return true
	case isRandom:
		if isStmt || isConst || isArg {
			return false
		}
		return true
	default:
		return false
	}
}
