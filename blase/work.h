#include <stdint.h>
#include <stdlib.h>
#include "cblas.h"

typedef enum {
	fn_undefined,

	fn_cblas_cdotu_sub,
	fn_cblas_cdotc_sub,
	fn_cblas_zdotu_sub,
	fn_cblas_zdotc_sub,
	fn_cblas_sswap,
	fn_cblas_scopy,
	fn_cblas_saxpy,
	fn_catlas_saxpby,
	fn_cblas_dswap,
	fn_cblas_dcopy,
	fn_cblas_daxpy,
	fn_catlas_daxpby,
	fn_cblas_cswap,
	fn_cblas_ccopy,
	fn_cblas_caxpy,
	fn_catlas_caxpby,
	fn_cblas_zswap,
	fn_cblas_zcopy,
	fn_cblas_zaxpy,
	fn_catlas_zaxpby,
	fn_cblas_srotg,
	fn_cblas_srotmg,
	fn_cblas_srot,
	fn_cblas_srotm,
	fn_cblas_drotg,
	fn_cblas_drotmg,
	fn_cblas_drot,
	fn_cblas_drotm,
	fn_cblas_sscal,
	fn_cblas_dscal,
	fn_cblas_cscal,
	fn_cblas_zscal,
	fn_cblas_csscal,
	fn_cblas_zdscal,
	fn_cblas_crotg,
	fn_cblas_zrotg,
	fn_cblas_csrot,
	fn_cblas_zdrot,
	fn_cblas_sgemv,
	fn_cblas_sgbmv,
	fn_cblas_strmv,
	fn_cblas_stbmv,
	fn_cblas_stpmv,
	fn_cblas_strsv,
	fn_cblas_stbsv,
	fn_cblas_stpsv,
	fn_cblas_dgemv,
	fn_cblas_dgbmv,
	fn_cblas_dtrmv,
	fn_cblas_dtbmv,
	fn_cblas_dtpmv,
	fn_cblas_dtrsv,
	fn_cblas_dtbsv,
	fn_cblas_dtpsv,
	fn_cblas_cgemv,
	fn_cblas_cgbmv,
	fn_cblas_ctrmv,
	fn_cblas_ctbmv,
	fn_cblas_ctpmv,
	fn_cblas_ctrsv,
	fn_cblas_ctbsv,
	fn_cblas_ctpsv,
	fn_cblas_zgemv,
	fn_cblas_zgbmv,
	fn_cblas_ztrmv,
	fn_cblas_ztbmv,
	fn_cblas_ztpmv,
	fn_cblas_ztrsv,
	fn_cblas_ztbsv,
	fn_cblas_ztpsv,
	fn_cblas_ssymv,
	fn_cblas_ssbmv,
	fn_cblas_sspmv,
	fn_cblas_sger,
	fn_cblas_ssyr,
	fn_cblas_sspr,
	fn_cblas_ssyr2,
	fn_cblas_sspr2,
	fn_cblas_dsymv,
	fn_cblas_dsbmv,
	fn_cblas_dspmv,
	fn_cblas_dger,
	fn_cblas_dsyr,
	fn_cblas_dspr,
	fn_cblas_dsyr2,
	fn_cblas_dspr2,
	fn_cblas_chemv,
	fn_cblas_chbmv,
	fn_cblas_chpmv,
	fn_cblas_cgeru,
	fn_cblas_cgerc,
	fn_cblas_cher,
	fn_cblas_chpr,
	fn_cblas_cher2,
	fn_cblas_chpr2,
	fn_cblas_zhemv,
	fn_cblas_zhbmv,
	fn_cblas_zhpmv,
	fn_cblas_zgeru,
	fn_cblas_zgerc,
	fn_cblas_zher,
	fn_cblas_zhpr,
	fn_cblas_zher2,
	fn_cblas_zhpr2,
	fn_cblas_sgemm,
	fn_cblas_ssymm,
	fn_cblas_ssyrk,
	fn_cblas_ssyr2k,
	fn_cblas_strmm,
	fn_cblas_strsm,
	fn_cblas_dgemm,
	fn_cblas_dsymm,
	fn_cblas_dsyrk,
	fn_cblas_dsyr2k,
	fn_cblas_dtrmm,
	fn_cblas_dtrsm,
	fn_cblas_cgemm,
	fn_cblas_csymm,
	fn_cblas_csyrk,
	fn_cblas_csyr2k,
	fn_cblas_ctrmm,
	fn_cblas_ctrsm,
	fn_cblas_zgemm,
	fn_cblas_zsymm,
	fn_cblas_zsyrk,
	fn_cblas_zsyr2k,
	fn_cblas_ztrmm,
	fn_cblas_ztrsm,
	fn_cblas_chemm,
	fn_cblas_cherk,
	fn_cblas_cher2k,
	fn_cblas_zhemm,
	fn_cblas_zherk,
	fn_cblas_zher2k,	
} cblasFn;

struct fnargs {
	cblasFn fn; 

	cblas_order order;
	cblas_transpose tA;
	cblas_transpose tB;

	uintptr_t a0;
	uintptr_t a1;
	uintptr_t a2;
	uintptr_t a3;


	int i0;
	int i1;
	int i2;
	int i3;
	int i4;
	int i5;

	double d0;
	double d1;
	double d2;
	double d3;
};

extern uintptr_t processFn(struct fnargs* args);

extern void prrintfnargs(struct fnargs* args);