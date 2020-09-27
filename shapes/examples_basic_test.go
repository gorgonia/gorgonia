package shapes

import "fmt"

func ExampleMatMul() {
	expr := Arrow{
		Arrow{
			Abstract{Var('a'), Var('b')},
			Abstract{Var('b'), Var('c')},
		},
		Abstract{Var('a'), Var('c')},
	}
	fmt.Printf("%v", expr)

	// Output:
	// (a, b) → (b, c) → (a, c)
}

func ExampleAdd() {
	expr := Arrow{
		Arrow{
			Var('a'),
			Var('a'),
		},
		Var('a'),
	}
	fmt.Printf("%v", expr)

	// Output:
	// a → a → a

}

func ExampleRavel() {
	expr := Arrow{
		Var('a'),
		UnaryOp{Prod, Var('a')},
	}
	fmt.Printf("%v", expr)

	// Output:
	// a → Π a
}

func ExampleTranspose() {
	simple := Arrow{
		Arrow{
			Var('a'),
			Axes{0, 1, 3, 2},
		},
		TransposeOf{
			Axes{0, 1, 3, 2},
			Var('a'),
		},
	}
	fmt.Printf("%v\n", simple)

	st := SubjectTo{
		Eq,
		UnaryOp{Dims, Axes{0, 1, 3, 2}},
		UnaryOp{Dims, Var('a')},
	}
	correct := Compound{
		Expr:      simple,
		SubjectTo: st,
	}
	fmt.Printf("%v", correct)

	// Output:
	// a → X[0 1 3 2] → Tr X[0 1 3 2] a
	// a → X[0 1 3 2] → Tr X[0 1 3 2] a s.t. (D X[0 1 3 2] = D a)
}

func ExampleSlice() {
	simple := Arrow{
		Arrow{
			Var('a'),
			Sli{0, 2, 1},
		},
		SliceOf{
			Sli{0, 2, 1},
			Var('a'),
		},
	}

	fmt.Printf("%v", simple)

	// Output:
	// a → [0:2] → a[0:2]
}

func ExampleReshape() {
	expr := Compound{
		Arrow{
			Arrow{
				Var('a'),
				Var('b'),
			},
			Var('b'),
		},
		SubjectTo{
			Eq,
			UnaryOp{Prod, Var('a')},
			UnaryOp{Prod, Var('b')},
		},
	}

	fmt.Printf("%v", expr)

	// Output:
	// a → b → b s.t. (Π a = Π b)
}
