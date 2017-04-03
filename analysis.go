package gorgonia

// dataflow analysis

type dataflow struct {
	uniques map[uint32]*Node

	replacements map[*Node]*Node
	devices      map[*Node]Device
	intervals    map[*Node]*interval

	// tracks the special nodes' children and parents
	devTransChildren map[*Node]Nodes
	devTransRepl     map[*Node]*Node
}

func newdataflow() *dataflow {
	df := new(dataflow)
	df.uniques = make(map[uint32]*Node)
	df.devices = make(map[*Node]Device)
	df.devTransChildren = make(map[*Node]Nodes)
	df.devTransRepl = make(map[*Node]*Node)
	return df
}

// equivalent to the value numbering algorithm
// it returns true if it is unique
func (df *dataflow) vn(n *Node) (retVal *Node, unique bool) {
	compileLogf("Value numbering")
	enterLoggingContext()
	defer leaveLoggingContext()

	node, ok := df.uniques[n.Hashcode()]

	if ok {
		return node, false
	}

	compileLogf("adding a new unique")
	// otherwise, add it to uniques, and then return itself
	df.uniques[n.Hashcode()] = n

	return n, true
}

// analyzeDevice records which node is supposed to be executed on which device.
//
// Currently it will only use Device 0. In the future, we can be smart about which device to use
func (df *dataflow) analyzeDevice(n *Node) {
	switch n.op.(type) {
	case CUDADoer:
		df.devices[n] = Device(0)
	case CLDoer:
		df.devices[n] = Device(0)
	default:
		df.devices[n] = CPU
	}
}

func analyze(g *ExprGraph, sorted Nodes) *dataflow {
	compileLogf("Performing dataflow analysis")
	enterLoggingContext()
	defer leaveLoggingContext()

	compileLogf("Finding unique leaves")
	df := newdataflow()
	for _, n := range g.leaves {
		df.uniques[n.Hashcode()] = n
	}

	// compileLogf("Common subexpression elimination")
	// compileLogf("analyzing devices")
	replacements := make(map[*Node]*Node)
	for _, n := range sorted {
		r, _ := df.vn(n)
		replacements[n] = r // CSE
		df.analyzeDevice(n) // Device targeting
	}
	df.replacements = replacements
	compileLogf("replacements: %-p", FmtNodeMap(replacements))
	compileLogf("Devices: %-v", FmtNodeMap(df.devices))

	// TODO
	// constant propagation
	/*
		for _, node := range g.nodes {
			n := node.(*Node)
			if len(n.Children) > 0 {
				allConst := true
				for _, child := range n.Children {
					if _, ok := child.Op.(constant); !ok {
						allConst = false
						break
					}
				}
			}
		}
	*/
	return df
}

func newDevTransNode(read, write *Node, from, to Device) *Node {
	op := devTrans{from, to}
	n := borrowNode()
	n.op = op
	n.shape = read.shape.Clone()
	n.t = read.t
	n.isStmt = true
	n.children = Nodes{read}
	return n
}

func (df *dataflow) insertDeviceInstr(sorted Nodes) Nodes {
	compileLogf("Inserting Device Transport Instructions")
	enterLoggingContext()
	defer leaveLoggingContext()
	// input -> output
	for i := 0; i < len(sorted); i++ {
		node := sorted[i]
		n := df.replacements[node]
		dev := df.devices[n]

		compileLogf("Working on %v. Replacement %v. Device %v", node, n, dev)
		var incr int
		replacementChildren := make(Nodes, len(n.children))
		enterLoggingContext()
		for j, child := range n.children {
			c := df.replacements[child]
			childDev := df.devices[c]

			compileLogf("Working on child :%v. Device: %v, Parent Device %v", c, childDev, dev)
			if childDev != dev {
				if repl, ok := df.devTransRepl[c]; ok {
					replacementChildren[j] = repl
					continue
				}
				transport := newDevTransNode(c, n, childDev, dev)
				sorted = append(sorted, nil)
				copy(sorted[i+1:], sorted[i:])
				sorted[i] = transport
				incr++
				compileLogf("Inserted %v", transport)

				// other stateful stuff
				df.devTransRepl[c] = transport
				df.replacements[transport] = transport
				replacementChildren[j] = transport
			} else {
				replacementChildren[j] = child
			}
		}
		leaveLoggingContext()

		if incr > 0 {
			df.devTransChildren[n] = replacementChildren
		}

		i += incr
	}
	return sorted
}

/*
	Notes on handling the live set:

	1. We load all the SSAs listed in the block's LiveIn
	2. Then we load all the SSAs used as input in this block Phi nodes
		- The reason for this is so that those SSAs can have intervals created
		  that are live in this block (well, they are kinda live)
	3. These input SSAs are temporary only, because a path-dependent liveset will be calculated below

	Consider a CFG that looks like this:

                           BLOCK 1           BLOCK 3
                           +-------+        +-------+
                     +---->| x = 1 +------->| y = 3 +----------------+
        BLOCK 0      |     +-------+        | use x |                v  BLOCK 4
       +-------+     |                      +-------+              +-------------+
       |       |+----+                                             | x = ϕ(1, 2) |
       +-------+     |     BLOCK 2                                 +-------------+
                     |     +-------+                                 ^
                     +---->| x = 2 +---------------------------------+
                           +-------+

	`x = 1` needs to be live in BLOCK 1, BLOCK 3 and BLOCK 4
	`x = 2` needs to be live in BLOCK 2 and BLOCK 4.

	The solution: in BLOCK 4, load `x = 1` and `x = 2` so they can be considered live in Block 4.

	The interval building process comes to BLOCK 3 next. It considers the SSAs that are live in BLOCK 4.
	If `x = 2` is live in BLOCK 4, it's Bad News with capital letters (see comment below).

	The solution: remove the InputSSAs of the Phi nodes when we're leaving this block.
*/
// TODO: rephrase above to fit this package's function.
// It's like the above, but without basic blocks, phi nodes, etc, making it a LOT simpler
func (df *dataflow) buildIntervals(sorted Nodes) {
	cudaLogf("Building intervals for %v", sorted)
	enterLoggingContext()
	defer leaveLoggingContext()

	intervals := make(map[*Node]*interval)

	var g *ExprGraph
	for _, n := range sorted {
		if g == nil && n.g != nil {
			g = n.g
		}

		intervals[n] = newInterval()
	}

	instructions := len(sorted)
	for i := len(sorted) - 1; i >= 0; i-- {
		n := sorted[i]
		logf("Working on %v", n)
		instrNum := i
		nInter := intervals[n]

		// inputs will be live the entire program
		if n.isInput() {
			nInter.addRange(instrNum, instructions)
			continue
		}
		nInter.addRange(instrNum, instrNum)

		// check for special cases requiring copying from device to device

		var children Nodes
		var ok bool
		if children, ok = df.devTransChildren[n]; !ok {
			children = n.children
		}

		logf("children of %v :%v", n, children)

		for _, child := range children {
			iv, ok := intervals[child]
			if !ok {
				// do something
				// parents := g.to[n]
				// for i, from := range parents {
				// 	ioutil.WriteFile(fmt.Sprintf("n_%d.dot", i), []byte(from.ToDot()), 0644)
				// }
			}
			logf("child %v", child)
			iv.addUsePositions(instrNum)
			// iv.setTo(instrNum)
		}
		// assume all derivations of input
		if len(n.derivOf) > 0 {
			for _, d := range n.derivOf {
				if d.isInput() {
					nInter.addUsePositions(instructions)
					break
				}
			}
		}
	}

	for _, iv := range intervals {
		iv.fix()
	}

	df.intervals = intervals
	return
}
