package gorgonia

import (
	"gonum.org/v1/gonum/blas"
	"gorgonia.org/tensor"
	"gorgonia.org/vecf32"
	"gorgonia.org/vecf64"
)

func (op *BatchNormOp) f64sOld(input, output *tensor.Dense) (err error) {
	n := input.Shape()[0]
	channels := input.Shape()[1]
	nc := channels * n
	spatialDim := input.Shape().TotalSize() / (nc)

	inputF64s := input.Float64s()
	outputF64s := output.Float64s()
	copy(outputF64s, inputF64s)

	meanTmp := op.runningMean.Float64s()
	mean := op.mean.Float64s()
	varianceTmp := op.runningVariance.Float64s()
	variance := op.variance.Float64s()
	tmp := op.tmpSpace.Float64s()
	ssm := op.spatialSumMultiplier.Float64s()
	nbc := op.numByChans.Float64s()
	bsm := op.batchSumMultiplier.Float64s()

	momentum := op.momentum
	eps := op.epsilon

	if !op.training {
		// use stored mean/variance estimates
		scaleFactor := float64(1)
		if fst := op.ma.Float64s()[0]; fst != 1 {
			scaleFactor = fst
		}
		copy(meanTmp, mean)
		whichblas.Dscal(len(meanTmp), scaleFactor, meanTmp, 1)
		copy(varianceTmp, variance)
		whichblas.Dscal(len(varianceTmp), scaleFactor, varianceTmp, 1)
	} else {
		// compute mean
		alpha := 1.0 / float64(n*spatialDim)
		whichblas.Dgemv(blas.NoTrans, nc, spatialDim, alpha, inputF64s, spatialDim, ssm, 1, 0, nbc, 1)
		whichblas.Dgemv(blas.Trans, n, channels, 1, nbc, channels, bsm, 1, 0, meanTmp, 1)
	}

	// subtract mean
	whichblas.Dgemm(blas.NoTrans, blas.NoTrans, n, channels, 1, 1, bsm, 1, meanTmp, channels, 0, nbc, channels)
	whichblas.Dgemm(blas.NoTrans, blas.NoTrans, nc, spatialDim, 1, -1, nbc, 1, ssm, spatialDim, 1, outputF64s, spatialDim)

	if op.training {
		// compute variance using var(X) = E(X-EX)²)
		copy(tmp, outputF64s)
		vecf64.Mul(tmp, tmp) // (X-EX) ^ 2

		whichblas.Dgemv(blas.NoTrans, nc, spatialDim, 1.0/(float64(n*spatialDim)), tmp, spatialDim, ssm, 1, 0, nbc, 1)
		whichblas.Dgemv(blas.Trans, n, channels, 1.0, nbc, channels, bsm, 1, 0, varianceTmp, 1) // E((X_EX)²)

		// compute and save moving average
		op.ma.Float64s()[0] *= momentum
		op.ma.Float64s()[0]++

		// TODO: write axpby for gonum
		whichblas.Dscal(len(mean), momentum, mean, 1)
		whichblas.Daxpy(len(meanTmp), 1.0, meanTmp, 1, mean, 1)

		m := len(inputF64s) / channels
		correctionFactor := float64(1)
		if m > 1 {
			correctionFactor = float64(m) / (float64(m - 1))
		}
		whichblas.Dscal(len(variance), momentum, variance, 1)
		whichblas.Daxpy(len(varianceTmp), correctionFactor, varianceTmp, 1, variance, 1)
	}

	// normalize variance
	vecf64.Trans(varianceTmp, eps)
	vecf64.Sqrt(varianceTmp)

	// replicate variance to inputsize
	whichblas.Dgemm(blas.NoTrans, blas.NoTrans, n, channels, 1, 1, bsm, 1, varianceTmp, channels, 0, nbc, channels)
	whichblas.Dgemm(blas.NoTrans, blas.NoTrans, nc, spatialDim, 1, 1, nbc, 1, ssm, spatialDim, 0, tmp, spatialDim)
	vecf64.Div(outputF64s, tmp)
	copy(op.xNorm.Float64s(), outputF64s) // caching

	return nil
}

func (op *BatchNormOp) f32sOld(input, output *tensor.Dense) (err error) {
	n := input.Shape()[0]
	channels := input.Shape()[1]
	nc := channels * n
	spatialDim := input.Shape().TotalSize() / (nc)

	inputF32s := input.Float32s()
	outputF32s := output.Float32s()
	copy(outputF32s, inputF32s)

	meanTmp := op.runningMean.Float32s()
	mean := op.mean.Float32s()
	varianceTmp := op.runningVariance.Float32s()
	variance := op.variance.Float32s()
	tmp := op.tmpSpace.Float32s()
	ssm := op.spatialSumMultiplier.Float32s()
	nbc := op.numByChans.Float32s()
	bsm := op.batchSumMultiplier.Float32s()

	momentum := float32(op.momentum)
	eps := float32(op.epsilon)

	if !op.training {
		// use stored mean/variance estimates
		scaleFactor := float32(1)
		if fst := op.ma.Float32s()[0]; fst != 1 {
			scaleFactor = fst
		}
		copy(meanTmp, mean)
		whichblas.Sscal(len(meanTmp), scaleFactor, meanTmp, 1)
		copy(varianceTmp, variance)
		whichblas.Sscal(len(varianceTmp), scaleFactor, varianceTmp, 1)
	} else {
		// compute mean
		alpha := 1.0 / float32(n*spatialDim)
		whichblas.Sgemv(blas.NoTrans, nc, spatialDim, alpha, inputF32s, spatialDim, ssm, 1, 0, nbc, 1)
		whichblas.Sgemv(blas.Trans, n, channels, 1, nbc, channels, bsm, 1, 0, meanTmp, 1)
	}

	// subtract mean
	whichblas.Sgemm(blas.NoTrans, blas.NoTrans, n, channels, 1, 1, bsm, 1, meanTmp, channels, 0, nbc, channels)
	whichblas.Sgemm(blas.NoTrans, blas.NoTrans, nc, spatialDim, 1, -1, nbc, 1, ssm, spatialDim, 1, outputF32s, spatialDim)

	if op.training {
		// compute variance using var(X) = E(X-EX)²)
		copy(tmp, outputF32s)
		vecf32.Mul(tmp, tmp) // (X-EX) ^ 2

		whichblas.Sgemv(blas.NoTrans, nc, spatialDim, 1.0/(float32(n*spatialDim)), tmp, spatialDim, ssm, 1, 0, nbc, 1)
		whichblas.Sgemv(blas.Trans, n, channels, 1.0, nbc, channels, bsm, 1, 0, varianceTmp, 1) // E((X_EX)²)

		// compute and save moving average
		op.ma.Float32s()[0] *= momentum
		op.ma.Float32s()[0]++

		// TODO: write axpby for gonum
		whichblas.Sscal(len(mean), momentum, mean, 1)
		whichblas.Saxpy(len(meanTmp), 1.0, meanTmp, 1, mean, 1)

		m := len(inputF32s) / channels
		correctionFactor := float32(1)
		if m > 1 {
			correctionFactor = float32(m) / (float32(m - 1))
		}
		whichblas.Sscal(len(variance), momentum, variance, 1)
		whichblas.Saxpy(len(varianceTmp), correctionFactor, varianceTmp, 1, variance, 1)
	}

	// normalize variance
	vecf32.Trans(varianceTmp, eps)
	vecf32.Sqrt(varianceTmp)

	// replicate variance to inputsize
	whichblas.Sgemm(blas.NoTrans, blas.NoTrans, n, channels, 1, 1, bsm, 1, varianceTmp, channels, 0, nbc, channels)
	whichblas.Sgemm(blas.NoTrans, blas.NoTrans, nc, spatialDim, 1, 1, nbc, 1, ssm, spatialDim, 0, tmp, spatialDim)
	vecf32.Div(outputF32s, tmp)
	copy(op.xNorm.Float32s(), outputF32s) // caching

	return nil
}

func (op *batchnormDiffOp) f64sOld(input, inGrad, outGrad *tensor.Dense) (err error) {
	in := input.Float64s()
	ig := inGrad.Float64s()
	og := outGrad.Float64s()
	tmp := op.tmpSpace.Float64s()
	out := op.xNorm.Float64s()
	ssm := op.spatialSumMultiplier.Float64s()
	nbc := op.numByChans.Float64s()
	bsm := op.batchSumMultiplier.Float64s()
	meanTmp := op.runningMean.Float64s()

	if !op.training {
		copy(ig, og)
		vecf64.Div(og, tmp)
		return nil
	}

	n := input.Shape()[0]
	channels := input.Shape()[1]
	nc := n * channels
	spatialDim := len(in) / nc

	// if Y = (X-mean(X))/(sqrt(var(X)+eps)), then
	//
	// dE(Y)/dX =
	//   (dE/dY - mean(dE/dY) - mean(dE/dY ⋅ Y) ⋅ Y)
	//     ./ sqrt(var(X) + eps)
	//
	// where ⋅ and ./ are hadamard product and elementwise division,
	// respectively, dE/dY is the top diff, and mean/var/sum are all computed
	// along all dimensions except the channels dimension.  In the above
	// equation, the operations allow for expansion (i.e. broadcast) along all
	// dimensions except the channels dimension where required.

	// sum(dE/dY ⋅ Y)
	copy(ig, out)
	vecf64.Mul(ig, og)
	whichblas.Dgemv(blas.NoTrans, nc, spatialDim, 1, ig, spatialDim, ssm, 1, 0, nbc, 1)
	whichblas.Dgemv(blas.Trans, n, channels, 1, nbc, channels, bsm, 1, 0, meanTmp, 1)

	// reshape (broadcast) the above
	whichblas.Dgemm(blas.NoTrans, blas.NoTrans, n, channels, 1, 1, bsm, 1, meanTmp, channels, 0, nbc, channels)
	whichblas.Dgemm(blas.NoTrans, blas.NoTrans, nc, spatialDim, 1, 1, nbc, 1, ssm, spatialDim, 0, ig, spatialDim)

	// sum(dE/dY ⋅ Y) ⋅ Y
	vecf64.Mul(ig, out)

	// sum(dE/dY)-sum(dE/dY ⋅ Y) ⋅ Y
	whichblas.Dgemv(blas.NoTrans, nc, spatialDim, 1, og, spatialDim, ssm, 1, 0, nbc, 1)
	whichblas.Dgemv(blas.Trans, n, channels, 1, nbc, channels, bsm, 1, 0, meanTmp, 1)

	// reshape (broadcast) the above to make
	// sum(dE/dY)-sum(dE/dY ⋅ Y) ⋅ Y
	whichblas.Dgemm(blas.NoTrans, blas.NoTrans, n, channels, 1, 1, bsm, 1, meanTmp, channels, 0, nbc, channels)
	whichblas.Dgemm(blas.NoTrans, blas.NoTrans, nc, spatialDim, 1, 1, nbc, 1, ssm, spatialDim, 1, ig, spatialDim)

	// dE/dY - mean(dE/dY)-mean(dE/dY ⋅ Y) ⋅ Y
	beta := (-1.0 / float64(nc))

	vecf64.Scale(ig, beta)
	vecf64.Add(ig, og)

	// note: temp_ still contains sqrt(var(X)+eps), computed during the forward
	// pass.
	vecf64.Div(ig, tmp)
	return nil

}

func (op *batchnormDiffOp) f32sOld(input, inGrad, outGrad *tensor.Dense) (err error) {
	in := input.Float32s()
	ig := inGrad.Float32s()
	og := outGrad.Float32s()
	tmp := op.tmpSpace.Float32s()
	out := op.xNorm.Float32s()
	ssm := op.spatialSumMultiplier.Float32s()
	nbc := op.numByChans.Float32s()
	bsm := op.batchSumMultiplier.Float32s()
	meanTmp := op.runningMean.Float32s()

	if !op.training {
		copy(ig, og)
		vecf32.Div(og, tmp)
		return nil
	}

	n := input.Shape()[0]
	channels := input.Shape()[1]
	nc := n * channels
	spatialDim := len(in) / nc

	// if Y = (X-mean(X))/(sqrt(var(X)+eps)), then
	//
	// dE(Y)/dX =
	//   (dE/dY - mean(dE/dY) - mean(dE/dY ⋅ Y) ⋅ Y)
	//     ./ sqrt(var(X) + eps)
	//
	// where ⋅ and ./ are hadamard product and elementwise division,
	// respectively, dE/dY is the top diff, and mean/var/sum are all computed
	// along all dimensions except the channels dimension.  In the above
	// equation, the operations allow for expansion (i.e. broadcast) along all
	// dimensions except the channels dimension where required.

	// sum(dE/dY ⋅ Y)
	copy(ig, out)
	vecf32.Mul(ig, og)
	whichblas.Sgemv(blas.NoTrans, nc, spatialDim, 1, ig, spatialDim, ssm, 1, 0, nbc, 1)
	whichblas.Sgemv(blas.Trans, n, channels, 1, nbc, channels, bsm, 1, 0, meanTmp, 1)

	// reshape (broadcast) the above
	whichblas.Sgemm(blas.NoTrans, blas.NoTrans, n, channels, 1, 1, bsm, 1, meanTmp, channels, 0, nbc, channels)
	whichblas.Sgemm(blas.NoTrans, blas.NoTrans, nc, spatialDim, 1, 1, nbc, 1, ssm, spatialDim, 0, ig, spatialDim)

	// sum(dE/dY ⋅ Y) ⋅ Y
	vecf32.Mul(ig, out)

	// sum(dE/dY)-sum(dE/dY ⋅ Y) ⋅ Y
	whichblas.Sgemv(blas.NoTrans, nc, spatialDim, 1, og, spatialDim, ssm, 1, 0, nbc, 1)
	whichblas.Sgemv(blas.Trans, n, channels, 1, nbc, channels, bsm, 1, 0, meanTmp, 1)

	// reshape (broadcast) the above to make
	// sum(dE/dY)-sum(dE/dY ⋅ Y) ⋅ Y
	whichblas.Sgemm(blas.NoTrans, blas.NoTrans, n, channels, 1, 1, bsm, 1, meanTmp, channels, 0, nbc, channels)
	whichblas.Sgemm(blas.NoTrans, blas.NoTrans, nc, spatialDim, 1, 1, nbc, 1, ssm, spatialDim, 1, ig, spatialDim)

	// dE/dY - mean(dE/dY)-mean(dE/dY ⋅ Y) ⋅ Y
	beta := (-1.0 / float32(n*spatialDim))
	vecf32.Scale(ig, beta)
	vecf32.Add(ig, og)

	// note: temp_ still contains sqrt(var(X)+eps), computed during the forward
	// pass.
	vecf32.Div(ig, tmp)
	return nil

}
