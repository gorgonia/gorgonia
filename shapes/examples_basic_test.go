package shapes

import (
	"fmt"
)

func Example_matMul() {
	matmul := Arrow{
		Abstract{Var('a'), Var('b')},
		Arrow{
			Abstract{Var('b'), Var('c')},
			Abstract{Var('a'), Var('c')},
		},
	}
	fmt.Printf("MatMul: %v\n", matmul)

	// Apply the first input to MatMul
	fst := Shape{2, 3}
	expr2, err := InferApp(matmul, fst)
	if err != nil {
		fmt.Printf("Error: %v\n", err)
	}
	fmt.Printf("Applying %v to MatMul:\n", fst)
	fmt.Printf("%v @ %v ↠ %v\n", matmul, fst, expr2)

	// Apply the second input
	snd := Shape{3, 4}
	expr3, err := InferApp(expr2, snd)
	if err != nil {
		fmt.Printf("Error: %v\n", err)
	}
	fmt.Printf("Applying %v to the result:\n", snd)
	fmt.Printf("%v @ %v ↠ %v\n", expr2, snd, expr3)

	// Bad example:
	bad2nd := Shape{4, 5}
	_, err = InferApp(expr2, bad2nd)
	fmt.Printf("What happens when you pass in a bad value (e.g. %v instead of %v):\n", bad2nd, snd)
	fmt.Printf("%v @ %v ↠ %v", expr2, bad2nd, err)

	// Output:
	// MatMul: (a, b) → (b, c) → (a, c)
	// Applying (2, 3) to MatMul:
	// (a, b) → (b, c) → (a, c) @ (2, 3) ↠ (3, c) → (2, c)
	// Applying (3, 4) to the result:
	// (3, c) → (2, c) @ (3, 4) ↠ (2, 4)
	// What happens when you pass in a bad value (e.g. (4, 5) instead of (3, 4)):
	// (3, c) → (2, c) @ (4, 5) ↠ Failed to solve [{(3, c) → (2, c) = (4, 5) → d}] | d: Unification Fail. 3 ~ 4 cannot proceed

}

func Example_add() {
	add := Arrow{
		Var('a'),
		Arrow{
			Var('a'),
			Var('a'),
		},
	}
	fmt.Printf("Add: %v\n", add)

	// pass in the first input
	fst := Shape{5, 2, 3, 1, 10}
	retExpr, err := InferApp(add, fst)
	if err != nil {
		fmt.Printf("Error %v\n", err)
	}
	fmt.Printf("Applying %v to Add:\n", fst)
	fmt.Printf("%v @ %v ↠ %v\n", add, fst, retExpr)

	// pass in the second input
	snd := Shape{5, 2, 3, 1, 10}
	retExpr2, err := InferApp(retExpr, snd)
	if err != nil {
		fmt.Printf("Error %v\n", err)
	}
	fmt.Printf("Applying %v to the result\n", snd)
	fmt.Printf("%v @ %v ↠ %v\n", retExpr, snd, retExpr2)

	// bad example:
	bad2nd := Shape{2, 3}
	_, err = InferApp(retExpr, bad2nd)

	fmt.Printf("Passing in a bad second input\n")
	fmt.Printf("%v @ %v ↠ %v", retExpr, bad2nd, err)

	// Output:
	// Add: a → a → a
	// Applying (5, 2, 3, 1, 10) to Add:
	// a → a → a @ (5, 2, 3, 1, 10) ↠ (5, 2, 3, 1, 10) → (5, 2, 3, 1, 10)
	// Applying (5, 2, 3, 1, 10) to the result
	// (5, 2, 3, 1, 10) → (5, 2, 3, 1, 10) @ (5, 2, 3, 1, 10) ↠ (5, 2, 3, 1, 10)
	// Passing in a bad second input
	// (5, 2, 3, 1, 10) → (5, 2, 3, 1, 10) @ (2, 3) ↠ Failed to solve [{(5, 2, 3, 1, 10) → (5, 2, 3, 1, 10) = (2, 3) → a}] | a: Unification Fail. (5, 2, 3, 1, 10) ~ (2, 3) cannot proceed as they do not contain the same amount of sub-expressions. (5, 2, 3, 1, 10) has 5 subexpressions while (2, 3) has 2 subexpressions

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

func ExampleColwiseSumMatrix() {
	expr := Compound{
		Arrow{
			Var('a'),
			Var('b'),
		},
		SubjectTo{
			Eq,
			UnaryOp{Dims, Var('b')},
			BinOp{
				Sub,
				UnaryOp{Dims, Var('a')},
				Size(1),
			},
		},
	}
	fmt.Printf("%v\n", expr)

	expr2 := Arrow{
		Abstract{Var('a'), Var('b')},
		Abstract{Var('a')},
	}
	fmt.Printf("%v", expr2)

	// Output:
	// a → b s.t. (D b = D a - 1)
	// (a, b) → (a)
}
