#include <stdlib.h>
#include <stdint.h>
#include <stdio.h>
#include "_cgo_export.h"
#include "work.h"
#include "cblas.h"

uintptr_t processFn(struct fnargs* args) {
	uintptr_t ret = 0;
	
	switch (args->fn) {
		case fn_undefined:
			abort(); // coredump all the things!
			break;
		case fn_cblas_cdotu_sub:
			abort();
			break;
		case fn_cblas_cdotc_sub:
			abort();
			break;
		case fn_cblas_zdotu_sub:
			abort();
			break;
		case fn_cblas_zdotc_sub:
			abort();
			break;
		case fn_cblas_sswap:
			abort();
			break;
		case fn_cblas_scopy:
			abort();
			break;
		case fn_cblas_saxpy:
			abort();
			break;
		case fn_catlas_saxpby:
			abort();
			break;
		case fn_cblas_dswap:
			abort();
			break;
		case fn_cblas_dcopy:
			abort();
			break;
		case fn_cblas_daxpy:
			abort();
			break;
		case fn_catlas_daxpby:
			abort();
			break;
		case fn_cblas_cswap:
			abort();
			break;
		case fn_cblas_ccopy:
			abort();
			break;
		case fn_cblas_caxpy:
			abort();
			break;
		case fn_catlas_caxpby:
			abort();
			break;
		case fn_cblas_zswap:
			abort();
			break;
		case fn_cblas_zcopy:
			abort();
			break;
		case fn_cblas_zaxpy:
			abort();
			break;
		case fn_catlas_zaxpby:
			abort();
			break;
		case fn_cblas_srotg:
			abort();
			break;
		case fn_cblas_srotmg:
			abort();
			break;
		case fn_cblas_srot:
			abort();
			break;
		case fn_cblas_srotm:
			abort();
			break;
		case fn_cblas_drotg:
			abort();
			break;
		case fn_cblas_drotmg:
			abort();
			break;
		case fn_cblas_drot:
			abort();
			break;
		case fn_cblas_drotm:
			abort();
			break;
		case fn_cblas_sscal:
			abort();
			break;
		case fn_cblas_dscal:
			abort();
			break;
		case fn_cblas_cscal:
			abort();
			break;
		case fn_cblas_zscal:
			abort();
			break;
		case fn_cblas_csscal:
			abort();
			break;
		case fn_cblas_zdscal:
			abort();
			break;
		case fn_cblas_crotg:
			abort();
			break;
		case fn_cblas_zrotg:
			abort();
			break;
		case fn_cblas_csrot:
			abort();
			break;
		case fn_cblas_zdrot:
			abort();
			break;
		case fn_cblas_sgemv:
			abort();
			break;
		case fn_cblas_sgbmv:
			abort();
			break;
		case fn_cblas_strmv:
			abort();
			break;
		case fn_cblas_stbmv:
			abort();
			break;
		case fn_cblas_stpmv:
			abort();
			break;
		case fn_cblas_strsv:
			abort();
			break;
		case fn_cblas_stbsv:
			abort();
			break;
		case fn_cblas_stpsv:
			abort();
			break;
		case fn_cblas_dgemv:
			cblas_dgemv(args->order, // order
			    args->tA, // transA
			    args->i0, // m
			    args->i1, // n
			    args->d0,  // alpha
			    (double*)args->a0, // a
			    args->i2, // lda
			    (double*)args->a1, // x
			    args->i3, // incX
			    args->d1, // beta
			    (double*)args->a2, // y
			    args->i4 //incY
			);
			break;
		case fn_cblas_dgbmv:
			abort();
			break;
		case fn_cblas_dtrmv:
			abort();
			break;
		case fn_cblas_dtbmv:
			abort();
			break;
		case fn_cblas_dtpmv:
			abort();
			break;
		case fn_cblas_dtrsv:
			abort();
			break;
		case fn_cblas_dtbsv:
			abort();
			break;
		case fn_cblas_dtpsv:
			abort();
			break;
		case fn_cblas_cgemv:
			abort();
			break;
		case fn_cblas_cgbmv:
			abort();
			break;
		case fn_cblas_ctrmv:
			abort();
			break;
		case fn_cblas_ctbmv:
			abort();
			break;
		case fn_cblas_ctpmv:
			abort();
			break;
		case fn_cblas_ctrsv:
			abort();
			break;
		case fn_cblas_ctbsv:
			abort();
			break;
		case fn_cblas_ctpsv:
			abort();
			break;
		case fn_cblas_zgemv:
			abort();
			break;
		case fn_cblas_zgbmv:
			abort();
			break;
		case fn_cblas_ztrmv:
			abort();
			break;
		case fn_cblas_ztbmv:
			abort();
			break;
		case fn_cblas_ztpmv:
			abort();
			break;
		case fn_cblas_ztrsv:
			abort();
			break;
		case fn_cblas_ztbsv:
			abort();
			break;
		case fn_cblas_ztpsv:
			abort();
			break;
		case fn_cblas_ssymv:
			abort();
			break;
		case fn_cblas_ssbmv:
			abort();
			break;
		case fn_cblas_sspmv:
			abort();
			break;
		case fn_cblas_sger:
			abort();
			break;
		case fn_cblas_ssyr:
			abort();
			break;
		case fn_cblas_sspr:
			abort();
			break;
		case fn_cblas_ssyr2:
			abort();
			break;
		case fn_cblas_sspr2:
			abort();
			break;
		case fn_cblas_dsymv:
			abort();
			break;
		case fn_cblas_dsbmv:
			abort();
			break;
		case fn_cblas_dspmv:
			abort();
			break;
		case fn_cblas_dger:
			abort();
			break;
		case fn_cblas_dsyr:
			abort();
			break;
		case fn_cblas_dspr:
			abort();
			break;
		case fn_cblas_dsyr2:
			abort();
			break;
		case fn_cblas_dspr2:
			abort();
			break;
		case fn_cblas_chemv:
			abort();
			break;
		case fn_cblas_chbmv:
			abort();
			break;
		case fn_cblas_chpmv:
			abort();
			break;
		case fn_cblas_cgeru:
			abort();
			break;
		case fn_cblas_cgerc:
			abort();
			break;
		case fn_cblas_cher:
			abort();
			break;
		case fn_cblas_chpr:
			abort();
			break;
		case fn_cblas_cher2:
			abort();
			break;
		case fn_cblas_chpr2:
			abort();
			break;
		case fn_cblas_zhemv:
			abort();
			break;
		case fn_cblas_zhbmv:
			abort();
			break;
		case fn_cblas_zhpmv:
			abort();
			break;
		case fn_cblas_zgeru:
			abort();
			break;
		case fn_cblas_zgerc:
			abort();
			break;
		case fn_cblas_zher:
			abort();
			break;
		case fn_cblas_zhpr:
			abort();
			break;
		case fn_cblas_zher2:
			abort();
			break;
		case fn_cblas_zhpr2:
			abort();
			break;
		case fn_cblas_sgemm:
			abort();
			break;
		case fn_cblas_ssymm:
			abort();
			break;
		case fn_cblas_ssyrk:
			abort();
			break;
		case fn_cblas_ssyr2k:
			abort();
			break;
		case fn_cblas_strmm:
			abort();
			break;
		case fn_cblas_strsm:
			abort();
			break;
		case fn_cblas_dgemm:
			cblas_dgemm(args->order, // order
			    args->tA, // transA
			    args->tB,  // transB
			    args->i0, // m
			    args->i1, // n
			    args->i2, // k
			    args->d0,  // alpha
			    (double*)args->a0, // A
			    args->i3, // lda
			    (double*)args->a1, // b
			    args->i4, // ldb
			    args->d1, // beta
			    (double*)args->a2, // C
			    args->i5 //ldc
			);
			break;
		case fn_cblas_dsymm:
			abort();
			break;
		case fn_cblas_dsyrk:
			abort();
			break;
		case fn_cblas_dsyr2k:
			abort();
			break;
		case fn_cblas_dtrmm:
			abort();
			break;
		case fn_cblas_dtrsm:
			abort();
			break;
		case fn_cblas_cgemm:
			abort();
			break;
		case fn_cblas_csymm:
			abort();
			break;
		case fn_cblas_csyrk:
			abort();
			break;
		case fn_cblas_csyr2k:
			abort();
			break;
		case fn_cblas_ctrmm:
			abort();
			break;
		case fn_cblas_ctrsm:
			abort();
			break;
		case fn_cblas_zgemm:
			abort();
			break;
		case fn_cblas_zsymm:
			abort();
			break;
		case fn_cblas_zsyrk:
			abort();
			break;
		case fn_cblas_zsyr2k:
			abort();
			break;
		case fn_cblas_ztrmm:
			abort();
			break;
		case fn_cblas_ztrsm:
			abort();
			break;
		case fn_cblas_chemm:
			abort();
			break;
		case fn_cblas_cherk:
			abort();
			break;
		case fn_cblas_cher2k:
			abort();
			break;
		case fn_cblas_zhemm:
			abort();
			break;
		case fn_cblas_zherk:
			abort();
			break;
		case fn_cblas_zher2k:
			abort();
			break;
	}
	return ret;
}

