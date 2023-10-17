module gorgonia.org/gorgonia

go 1.12

replace gorgonia.org/tensor => /home/chewxy/workspace/gorgoniaws/src/gorgonia.org/tensor

replace gorgonia.org/dtype => /home/chewxy/workspace/gorgoniaws/src/gorgonia.org/dtype

replace gorgonia.org/shapes => /home/chewxy/workspace/gorgoniaws/src/gorgonia.org/shapes

require (
	github.com/chewxy/hm v1.0.0
	github.com/chewxy/math32 v1.10.1
	github.com/fatih/color v1.7.0 // indirect
	github.com/go-gota/gota v0.10.1
	github.com/mattn/go-colorable v0.1.4 // indirect
	github.com/pkg/errors v0.9.1
	github.com/stretchr/testify v1.8.3
	github.com/xtgo/set v1.0.0
	gonum.org/v1/gonum v0.13.0
	gonum.org/v1/netlib v0.0.0-20200317120129-c5a04cffd98a
	gopkg.in/check.v1 v1.0.0-20180628173108-788fd7840127 // indirect
	gopkg.in/cheggaaa/pb.v1 v1.0.27
	gorgonia.org/cu v0.9.2
	gorgonia.org/dawson v1.2.0
	gorgonia.org/dtype v0.10.0
	gorgonia.org/internal v0.0.0-20210804133310-438f7f1f5027
	gorgonia.org/shapes v0.0.0-20220805023001-db33330e8e09
	gorgonia.org/tensor v0.9.20
)
