package gorgonia

import (
	"fmt"
	"strings"

	"gorgonia.org/tensor"
)

func ExampleSoftMax() {
	g := NewGraph()
	t := tensor.New(tensor.WithShape(2, 3), tensor.WithBacking([]float64{1, 3, 2, 3, 2, 1}))
	u := t.Clone().(*tensor.Dense)
	v := tensor.New(tensor.WithShape(2, 2, 3), tensor.WithBacking([]float64{
		1, 3, 2,
		4, 2, 1,

		3, 5, 3,
		2, 1, 5,
	}))

	a := NodeFromAny(g, t, WithName("a"))
	b := NodeFromAny(g, u, WithName("b"))
	c := NodeFromAny(g, v, WithName("c"))

	sm1 := Must(SoftMax(a))
	sm0 := Must(SoftMax(b, 0))
	sm := Must(SoftMax(c))
	m := NewTapeMachine(g)
	if err := m.RunAll(); err != nil {
		panic(err)
	}

	fmt.Printf("a:\n%v\nsoftmax(a) - along last axis (default behaviour):\n%1.2f", a.Value(), sm1.Value())
	fmt.Printf("b:\n%v\nsoftmax(b) - along axis 0:\n%1.2f", b.Value(), sm0.Value())

	tmp := fmt.Sprintf("c %v:\n%v\nsoftmax(c) - along last axis (default behaviour) %v:\n%1.2f", c.Value().Shape(), c.Value(), sm.Value().Shape(), sm.Value())

	fmt.Println(strings.Replace(tmp, "\n\n\n", "\n\n", -1))

	// the requirement to use tmp and strings.Replace is because when Go runs example tests, it strips excess newlines.

	// Output:
	// a:
	// ⎡1  3  2⎤
	// ⎣3  2  1⎦
	//
	// softmax(a) - along last axis (default behaviour):
	// ⎡0.09  0.67  0.24⎤
	// ⎣0.67  0.24  0.09⎦
	// b:
	// ⎡1  3  2⎤
	// ⎣3  2  1⎦
	//
	// softmax(b) - along axis 0:
	// ⎡0.12  0.73  0.73⎤
	// ⎣0.88  0.27  0.27⎦
	// c (2, 2, 3):
	// ⎡1  3  2⎤
	// ⎣4  2  1⎦
	//
	// ⎡3  5  3⎤
	// ⎣2  1  5⎦
	//
	//
	// softmax(c) - along last axis (default behaviour) (2, 2, 3):
	// ⎡0.09  0.67  0.24⎤
	// ⎣0.84  0.11  0.04⎦
	//
	// ⎡0.11  0.79  0.11⎤
	// ⎣0.05  0.02  0.94⎦

}

func ExampleConcat() {
	g := NewGraph()
	x := NewTensor(g, Float64, 4, WithShape(2, 3, 4, 5), WithInit(RangedFrom(0)), WithName("x"))
	y := NewTensor(g, Float64, 4, WithShape(2, 3, 4, 5), WithInit(RangedFrom(120)), WithName("y"))

	z, err := Concat(2, x, y)
	if err != nil {
		panic(err)
	}

	m := NewTapeMachine(g)
	if err := m.RunAll(); err != nil {
		panic(err)
	}
	tmp := fmt.Sprintf("z %v\n%v", z.Value().Shape(), z.Value())
	fmt.Println(strings.Replace(tmp, "\n\n", "\n", -1)) // this is because

	// Output:
	//z (2, 3, 8, 5)
	//⎡  0    1    2    3    4⎤
	//⎢  5    6    7    8    9⎥
	//⎢ 10   11   12   13   14⎥
	//⎢ 15   16   17   18   19⎥
	//⎢120  121  122  123  124⎥
	//⎢125  126  127  128  129⎥
	//⎢130  131  132  133  134⎥
	//⎣135  136  137  138  139⎦
	//
	//
	//⎡ 20   21   22   23   24⎤
	//⎢ 25   26   27   28   29⎥
	//⎢ 30   31   32   33   34⎥
	//⎢ 35   36   37   38   39⎥
	//⎢140  141  142  143  144⎥
	//⎢145  146  147  148  149⎥
	//⎢150  151  152  153  154⎥
	//⎣155  156  157  158  159⎦
	//
	//
	//⎡ 40   41   42   43   44⎤
	//⎢ 45   46   47   48   49⎥
	//⎢ 50   51   52   53   54⎥
	//⎢ 55   56   57   58   59⎥
	//⎢160  161  162  163  164⎥
	//⎢165  166  167  168  169⎥
	//⎢170  171  172  173  174⎥
	//⎣175  176  177  178  179⎦
	//
	//
	//⎡ 60   61   62   63   64⎤
	//⎢ 65   66   67   68   69⎥
	//⎢ 70   71   72   73   74⎥
	//⎢ 75   76   77   78   79⎥
	//⎢180  181  182  183  184⎥
	//⎢185  186  187  188  189⎥
	//⎢190  191  192  193  194⎥
	//⎣195  196  197  198  199⎦
	//
	//
	//⎡ 80   81   82   83   84⎤
	//⎢ 85   86   87   88   89⎥
	//⎢ 90   91   92   93   94⎥
	//⎢ 95   96   97   98   99⎥
	//⎢200  201  202  203  204⎥
	//⎢205  206  207  208  209⎥
	//⎢210  211  212  213  214⎥
	//⎣215  216  217  218  219⎦
	//
	//
	//⎡100  101  102  103  104⎤
	//⎢105  106  107  108  109⎥
	//⎢110  111  112  113  114⎥
	//⎢115  116  117  118  119⎥
	//⎢220  221  222  223  224⎥
	//⎢225  226  227  228  229⎥
	//⎢230  231  232  233  234⎥
	//⎣235  236  237  238  239⎦
}

func ExampleUnconcat() {
	g := NewGraph()
	x := NewTensor(g, Float64, 4, WithShape(2, 3, 4, 5), WithInit(RangedFrom(0)), WithName("x"))
	y := NewTensor(g, Float64, 4, WithShape(2, 3, 4, 5), WithInit(RangedFrom(120)), WithName("y"))

	z, err := Concat(2, x, y)
	if err != nil {
		panic(err)
	}

	unconcats, err := Unconcat(z, 2, 2)
	if err != nil {
		panic(err)
	}
	a, b := unconcats[0], unconcats[1]

	m := NewTapeMachine(g)
	if err := m.RunAll(); err != nil {
		panic(err)
	}
	tmp := fmt.Sprintf("a %v\n%v\nb %v\n%v", a.Value().Shape(), a.Value(), b.Value().Shape(), b.Value())
	fmt.Println(strings.Replace(tmp, "\n\n", "\n", -1))

	// Output:
	// a (2, 3, 4, 5)
	// ⎡  0    1    2    3    4⎤
	// ⎢  5    6    7    8    9⎥
	// ⎢ 10   11   12   13   14⎥
	// ⎣ 15   16   17   18   19⎦
	//
	//
	// ⎡ 20   21   22   23   24⎤
	// ⎢ 25   26   27   28   29⎥
	// ⎢ 30   31   32   33   34⎥
	// ⎣ 35   36   37   38   39⎦
	//
	//
	// ⎡ 40   41   42   43   44⎤
	// ⎢ 45   46   47   48   49⎥
	// ⎢ 50   51   52   53   54⎥
	// ⎣ 55   56   57   58   59⎦
	//
	//
	// ⎡ 60   61   62   63   64⎤
	// ⎢ 65   66   67   68   69⎥
	// ⎢ 70   71   72   73   74⎥
	// ⎣ 75   76   77   78   79⎦
	//
	//
	// ⎡ 80   81   82   83   84⎤
	// ⎢ 85   86   87   88   89⎥
	// ⎢ 90   91   92   93   94⎥
	// ⎣ 95   96   97   98   99⎦
	//
	//
	// ⎡100  101  102  103  104⎤
	// ⎢105  106  107  108  109⎥
	// ⎢110  111  112  113  114⎥
	// ⎣115  116  117  118  119⎦
	//
	//
	//
	// b (2, 3, 4, 5)
	// ⎡120  121  122  123  124⎤
	// ⎢125  126  127  128  129⎥
	// ⎢130  131  132  133  134⎥
	// ⎣135  136  137  138  139⎦
	//
	//
	// ⎡140  141  142  143  144⎤
	// ⎢145  146  147  148  149⎥
	// ⎢150  151  152  153  154⎥
	// ⎣155  156  157  158  159⎦
	//
	//
	// ⎡160  161  162  163  164⎤
	// ⎢165  166  167  168  169⎥
	// ⎢170  171  172  173  174⎥
	// ⎣175  176  177  178  179⎦
	//
	//
	// ⎡180  181  182  183  184⎤
	// ⎢185  186  187  188  189⎥
	// ⎢190  191  192  193  194⎥
	// ⎣195  196  197  198  199⎦
	//
	//
	// ⎡200  201  202  203  204⎤
	// ⎢205  206  207  208  209⎥
	// ⎢210  211  212  213  214⎥
	// ⎣215  216  217  218  219⎦
	//
	//
	// ⎡220  221  222  223  224⎤
	// ⎢225  226  227  228  229⎥
	// ⎢230  231  232  233  234⎥
	// ⎣235  236  237  238  239⎦
}
