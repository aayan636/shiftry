// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#include <iostream>
#include <cmath>

#include "datatypes.h"
#include "library_fixed.h"

// This file contains implementations of the linear algebra operators supported by SeeDot.
// Each function takes the scaling factors as arguments along with the pointers to the operands.

// C = A + B
void MatAddNN(MYINT *A, MYINT *B, MYINT *C, MYINT I, MYINT J, MYINT shrA, MYINT shrB, MYINT shrC)
{
	for (MYITE i = 0; i < I; i++)
	{
		for (MYITE j = 0; j < J; j++)
		{
			MYINT a = A[i * J + j];
			MYINT b = B[i * J + j];

			a = a / shrA;
			b = b / shrB;

			MYINT c = Saturate<MYINT>(a / shrC + b / shrC);

			C[i * J + j] = c;
		}
	}
	return;
}

// C = A + B
void MatAddCN(const MYINT *A, MYINT *B, MYINT *C, MYINT I, MYINT J, MYINT shrA, MYINT shrB, MYINT shrC)
{
	for (MYITE i = 0; i < I; i++)
	{
		for (MYITE j = 0; j < J; j++)
		{
			MYINT a = A[i * J + j];
			MYINT b = B[i * J + j];

			a = a / shrA;
			b = b / shrB;

			MYINT c = Saturate<MYINT>(a / shrC + b / shrC);

			C[i * J + j] = c;
		}
	}
	return;
}

// C = A + B
void MatAddNC(MYINT *A, const MYINT *B, MYINT *C, MYINT I, MYINT J, MYINT shrA, MYINT shrB, MYINT shrC)
{
	for (MYITE i = 0; i < I; i++)
	{
		for (MYITE j = 0; j < J; j++)
		{
			MYINT a = A[i * J + j];
			MYINT b = B[i * J + j];

			a = a / shrA;
			b = b / shrB;

			MYINT c = Saturate<MYINT>(a / shrC + b / shrC);

			C[i * J + j] = c;
		}
	}
	return;
}

// C = A + B
void MatAddCC(const MYINT *A, const MYINT *B, MYINT *C, MYINT I, MYINT J, MYINT shrA, MYINT shrB, MYINT shrC)
{
	for (MYITE i = 0; i < I; i++)
	{
		for (MYITE j = 0; j < J; j++)
		{
			MYINT a = A[i * J + j];
			MYINT b = B[i * J + j];

			a = a / shrA;
			b = b / shrB;

			MYINT c = Saturate<MYINT>(a / shrC + b / shrC);

			C[i * J + j] = c;
		}
	}
	return;
}

// C = a + B
void MatAddBroadCastA(MYINT *A, MYINT *B, MYINT *C, MYINT I, MYINT J, MYINT shrA, MYINT shrB, MYINT shrC)
{
	for (MYITE i = 0; i < I; i++)
	{
		for (MYITE j = 0; j < J; j++)
		{
			MYINT a = *A;
			MYINT b = B[i * J + j];

			a = a / shrA;
			b = b / shrB;

			MYINT c = Saturate<MYINT>(a / shrC + b / shrC);

			C[i * J + j] = c;
		}
	}
	return;
}

// C = A + b
void MatAddBroadCastB(MYINT *A, MYINT *B, MYINT *C, MYINT I, MYINT J, MYINT shrA, MYINT shrB, MYINT shrC)
{
	for (MYITE i = 0; i < I; i++)
	{
		for (MYITE j = 0; j < J; j++)
		{
			MYINT a = A[i * J + j];
			MYINT b = *B;

			a = a / shrA;
			b = b / shrB;

			MYINT c = Saturate<MYINT>(a / shrC + b / shrC);

			C[i * J + j] = c;
		}
	}
	return;
}

// C = A - B
// TODO: shrB is int32_t because in 8-bit/16-bit code, shrB is usually very high and int8_t/int16_t will overflow.
void MatSub(MYINT *A, const MYINT *B, MYINT *C, MYINT I, MYINT J, MYINT shrA, int32_t shrB, MYINT shrC)
{
	for (MYITE i = 0; i < I; i++)
	{
		for (MYITE j = 0; j < J; j++)
		{
			MYINT a = A[i * J + j];
			MYINT b = B[i * J + j];

			a = a / shrA;
			b = b / shrB;

			MYINT c = Saturate<MYINT>(a / shrC - b / shrC);

			C[i * J + j] = c;
		}
	}
	return;
}

// C = a - B
// TODO: shrB is int32_t because in 8-bit/16-bit code, shrB is usually very high and int8_t/int16_t will overflow.
void MatSubBroadCastA(MYINT *A, MYINT *B, MYINT *C, MYINT I, MYINT J, MYINT shrA, int32_t shrB, MYINT shrC)
{
	for (MYITE i = 0; i < I; i++)
	{
		for (MYITE j = 0; j < J; j++)
		{
			MYINT a = *A;
			MYINT b = B[i * J + j];

			a = a / shrA;
			b = b / shrB;

			MYINT c = Saturate<MYINT>(a / shrC - b / shrC);

			C[i * J + j] = c;
		}
	}
	return;
}

// C = A - b
// TODO: shrB is int32_t because in 8-bit/16-bit code, shrB is usually very high and int8_t/int16_t will overflow.
void MatSubBroadCastB(MYINT *A, MYINT *B, MYINT *C, MYINT I, MYINT J, MYINT shrA, int32_t shrB, MYINT shrC)
{
	for (MYITE i = 0; i < I; i++)
	{
		for (MYITE j = 0; j < J; j++)
		{
			MYINT a = A[i * J + j];
			MYINT b = *B;

			a = a / shrA;
			b = b / shrB;

			MYINT c = Saturate<MYINT>(a / shrC - b / shrC);

			C[i * J + j] = c;
		}
	}
	return;
}

// C = A * B
void MatMulNN(MYINT *A, MYINT *B, MYINT *C, MYINT *tmp, MYINT I, MYINT K, MYINT J, MYINT shrA, MYINT shrB, MYINT H1, MYINT H2)
{

	for (MYITE i = 0; i < I; i++)
	{
		for (MYITE j = 0; j < J; j++)
		{
			for (MYITE k = 0; k < K; k++)
			{
				MYINT a = A[i * K + k];
				MYINT b = B[k * J + j];

#ifdef FASTAPPROX
				a = a / shrA;
				b = b / shrB;

				tmp[k] = a * b;
#else
				int64_t prod = ((int64_t)a * (int64_t)b);
				tmp[k] = Saturate<MYINT>((prod / ((int64_t)shrB * (int64_t)shrA)));
#endif
			}

			MYITE count = K, depth = 0;
			bool shr = true;

			while (depth < (H1 + H2))
			{
				if (depth >= H1)
					shr = false;

				for (MYITE p = 0; p < (K / 2 + 1); p++)
				{
					MYINT sum;
					if (p < (count >> 1))
					{
						if (shr)
							sum = tmp[2 * p] / 2 + tmp[(2 * p) + 1] / 2;
						else
							sum = tmp[2 * p] + tmp[(2 * p) + 1];
					}
					else if ((p == (count >> 1)) && ((count & 1) == 1))
					{
						if (shr)
							sum = tmp[2 * p] / 2;
						else
							sum = tmp[2 * p];
					}
					else
						sum = 0;

					tmp[p] = sum;
				}
				count = (count + 1) >> 1;

				depth++;
			}

			C[i * J + j] = tmp[0];
		}
	}
	return;
}

// C = A * B
void MatMulCN(const MYINT *A, MYINT *B, MYINT *C, MYINT *tmp, MYINT I, MYINT K, MYINT J, MYINT shrA, MYINT shrB, MYINT H1, MYINT H2)
{

	for (MYITE i = 0; i < I; i++)
	{
		for (MYITE j = 0; j < J; j++)
		{
			for (MYITE k = 0; k < K; k++)
			{
				MYINT a = A[i * K + k];
				MYINT b = B[k * J + j];

#ifdef FASTAPPROX
				a = a / shrA;
				b = b / shrB;

				tmp[k] = a * b;
#else
				int64_t prod = ((int64_t)a * (int64_t)b);
				tmp[k] = Saturate<MYINT>((prod / ((int64_t)shrB * (int64_t)shrA)));
#endif
			}

			MYITE count = K, depth = 0;
			bool shr = true;

			while (depth < (H1 + H2))
			{
				if (depth >= H1)
					shr = false;

				for (MYITE p = 0; p < (K / 2 + 1); p++)
				{
					MYINT sum;
					if (p < (count >> 1))
					{
						if (shr)
							sum = tmp[2 * p] / 2 + tmp[(2 * p) + 1] / 2;
						else
							sum = tmp[2 * p] + tmp[(2 * p) + 1];
					}
					else if ((p == (count >> 1)) && ((count & 1) == 1))
					{
						if (shr)
							sum = tmp[2 * p] / 2;
						else
							sum = tmp[2 * p];
					}
					else
						sum = 0;

					tmp[p] = sum;
				}
				count = (count + 1) >> 1;

				depth++;
			}

			C[i * J + j] = tmp[0];
		}
	}
	return;
}

// C = A * B
void MatMulNC(MYINT *A, const MYINT *B, MYINT *C, MYINT *tmp, MYINT I, MYINT K, MYINT J, MYINT shrA, MYINT shrB, MYINT H1, MYINT H2)
{

	for (MYITE i = 0; i < I; i++)
	{
		for (MYITE j = 0; j < J; j++)
		{
			for (MYITE k = 0; k < K; k++)
			{
				MYINT a = A[i * K + k];
				MYINT b = B[k * J + j];

#ifdef FASTAPPROX
				a = a / shrA;
				b = b / shrB;

				tmp[k] = a * b;
#else
				int64_t prod = ((int64_t)a * (int64_t)b);
				tmp[k] = Saturate<MYINT>((prod / ((int64_t)shrB * (int64_t)shrA)));
#endif
			}

			MYITE count = K, depth = 0;
			bool shr = true;

			while (depth < (H1 + H2))
			{
				if (depth >= H1)
					shr = false;

				for (MYITE p = 0; p < (K / 2 + 1); p++)
				{
					MYINT sum;
					if (p < (count >> 1))
					{
						if (shr)
							sum = tmp[2 * p] / 2 + tmp[(2 * p) + 1] / 2;
						else
							sum = tmp[2 * p] + tmp[(2 * p) + 1];
					}
					else if ((p == (count >> 1)) && ((count & 1) == 1))
					{
						if (shr)
							sum = tmp[2 * p] / 2;
						else
							sum = tmp[2 * p];
					}
					else
						sum = 0;

					tmp[p] = sum;
				}
				count = (count + 1) >> 1;

				depth++;
			}

			C[i * J + j] = tmp[0];
		}
	}
	return;
}

// C = A * B
void MatMulCC(const MYINT *A, const MYINT *B, MYINT *C, MYINT *tmp, MYINT I, MYINT K, MYINT J, MYINT shrA, MYINT shrB, MYINT H1, MYINT H2)
{

	for (MYITE i = 0; i < I; i++)
	{
		for (MYITE j = 0; j < J; j++)
		{
			for (MYITE k = 0; k < K; k++)
			{
				MYINT a = A[i * K + k];
				MYINT b = B[k * J + j];

#ifdef FASTAPPROX
				a = a / shrA;
				b = b / shrB;

				tmp[k] = a * b;
#else
				int64_t prod = ((int64_t)a * (int64_t)b);
				tmp[k] = Saturate<MYINT>((prod / ((int64_t)shrB * (int64_t)shrA)));
#endif
			}

			MYITE count = K, depth = 0;
			bool shr = true;

			while (depth < (H1 + H2))
			{
				if (depth >= H1)
					shr = false;

				for (MYITE p = 0; p < (K / 2 + 1); p++)
				{
					MYINT sum;
					if (p < (count >> 1))
					{
						if (shr)
							sum = tmp[2 * p] / 2 + tmp[(2 * p) + 1] / 2;
						else
							sum = tmp[2 * p] + tmp[(2 * p) + 1];
					}
					else if ((p == (count >> 1)) && ((count & 1) == 1))
					{
						if (shr)
							sum = tmp[2 * p] / 2;
						else
							sum = tmp[2 * p];
					}
					else
						sum = 0;

					tmp[p] = sum;
				}
				count = (count + 1) >> 1;

				depth++;
			}

			C[i * J + j] = tmp[0];
		}
	}
	return;
}

// C = A |*| B
// TODO: K is int16_t because K is usually very high and int8_t will overflow in 8-bit code.
void SparseMatMul(const MYINT *Aidx, const MYINT *Aval, MYINT **B, MYINT *C, int16_t K, MYINT shrA, MYINT shrB, MYINT shrC)
{

	MYITE ite_idx = 0, ite_val = 0;
	for (MYITE k = 0; k < K; k++)
	{
		// MYINT b = getIntFeature(k);
		MYINT b = B[k * 1][0];
#ifdef FASTAPPROX
		b = b / shrB;
#endif

		MYITE idx = Aidx[ite_idx];
		while (idx != 0)
		{
			MYINT a = Aval[ite_val];
#ifdef FASTAPPROX
			a = a / shrA;

			MYINT c = a * b;
			c = c / shrC;
#else
			MYINT c = Saturate<MYINT>(((int64_t)a * (int64_t)b) / ((int64_t)shrC * (int64_t)shrA * (int64_t)shrB));
#endif

			C[idx - 1] += c;

			ite_idx++;
			ite_val++;

			idx = Aidx[ite_idx];
		}
		ite_idx++;
	}

	return;
}

// C = A <*> B
void MulCir(MYINT *A, MYINT *B, MYINT *C, MYINT I, MYINT J, MYINT shrA, MYINT shrB)
{
	for (MYITE i = 0; i < I; i++)
	{
		for (MYITE j = 0; j < J; j++)
		{
			MYINT a = A[i * J + j];
			MYINT b = B[i * J + j];

#ifdef FASTAPPROX
			a = a / shrA;
			b = b / shrB;

			C[i * J + j] = a * b;
#else
			int64_t prod = ((int64_t)a * (int64_t)b);
			C[i * J + j] = Saturate<MYINT>(prod / ((int64_t)shrB * (int64_t)shrA));
#endif
		}
	}
	return;
}

// A = tanh(A)
void TanH(MYINT *A, MYINT I, MYINT J, MYINT scale_in, MYINT scale_out, MYINT *B)
{
	for (MYITE i = 0; i < I; i++)
	{
		for (MYITE j = 0; j < J; j++)
		{
#ifdef FLOATEXP
			float x = float(A[i * J + j]) / scale_in;

			float y = tanh(x);

			MYINT z = MYINT(y * scale_out);

			B[i * J + j] = z;
#else
			MYINT x = A[i * J + j], y;

			if (x >= scale_in)
				y = scale_in;
			else if (x <= -scale_in)
				y = -scale_in;
			else
				y = x;

			MYINT scale_diff = scale_out / scale_in;

			y *= scale_diff;

			B[i * J + j] = y;
#endif
		}
	}
	return;
}

// index = argmax(A)
void ArgMax(MYINT *A, MYINT I, MYINT J, MYINT *index)
{

	MYINT max = A[0];
	MYITE maxIndex = 0, counter = 0;
	for (MYITE i = 0; i < I; i++)
	{
		for (MYITE j = 0; j < J; j++)
		{
			MYINT x = A[i * J + j];

			if (max < x)
			{
				maxIndex = counter;
				max = x;
			}

			counter++;
		}
	}

	*index = maxIndex;

	return;
}

// A = A^T
void Transpose(MYINT *A, MYINT *B, MYINT I, MYINT J)
{
	for (MYITE i = 0; i < I; i++)
	{
		for (MYITE j = 0; j < J; j++)
		{
			B[i * J + j] = A[j * I + i];
		}
	}
	return;
}

// C = a * B
void ScalarMul(MYINT *A, MYINT *B, MYINT *C, MYINT I, MYINT J, MYINT shrA, MYINT shrB)
{

	MYINT a = *A;
#ifdef FASTAPPROX
	a = a / shrA;
#endif

	for (MYITE i = 0; i < I; i++)
	{
		for (MYITE j = 0; j < J; j++)
		{
			MYINT b = B[i * J + j];

#ifdef FASTAPPROX
			b = b / shrB;

			C[i * J + j] = a * b;
#else
			int64_t prod = ((int64_t)a * (int64_t)b);
			C[i * J + j] = Saturate<MYINT>(prod / ((int64_t)shrA * (int64_t)shrB));
#endif
		}
	}

	return;
}

// A = A <+> B
// A[N][H][W][C], B[C]
void AddOrSubCir4D(MYINT *A, const MYINT *B, MYINT *X, MYINT N, MYINT H, MYINT W, MYINT C, MYINT shrA, MYINT shrB, MYINT shrC, bool add)
{

	for (MYITE n = 0; n < N; n++)
	{
		for (MYITE h = 0; h < H; h++)
		{
			for (MYITE w = 0; w < W; w++)
			{
				for (MYITE c = 0; c < C; c++)
				{
					MYINT a = A[n * H * W * C + h * W * C + w * C + c];
					a = a / shrA;

					MYINT b = B[c];
					b = b / shrB;

					MYINT res;
					if (add)
						res = Saturate<MYINT>(a / shrC + b / shrC);
					else
						res = Saturate<MYINT>(a / shrC - b / shrC);
					X[n * H * W * C + h * W * C + w * C + c] = res;
				}
			}
		}
	}

	return;
}

// A = A <+> B
// A[N][H][W][C], B[C]
void AddOrSubCir2D(MYINT *A, const MYINT *B, MYINT *X, MYINT H, MYINT W, MYINT shrA, MYINT shrB, MYINT shrC, bool add)
{

	for (MYITE h = 0; h < H; h++)
	{
		for (MYITE w = 0; w < W; w++)
		{
			MYINT a = A[h * W + w];
			a = a / shrA;

			MYINT b = B[w];
			b = b / shrB;

			MYINT res;
			if (add)
				res = Saturate<MYINT>(a / shrC + b / shrC);
			else
				res = Saturate<MYINT>(a / shrC - b / shrC);
			X[h * W + w] = res;
		}
	}

	return;
}

// A = relu(A)
// A[N][H][W][C]
void Relu4D(MYINT *A, MYINT N, MYINT H, MYINT W, MYINT C)
{

	for (MYITE n = 0; n < N; n++)
	{
		for (MYITE h = 0; h < H; h++)
		{
			for (MYITE w = 0; w < W; w++)
			{
				for (MYITE c = 0; c < C; c++)
				{
					MYINT a = A[n * H * W * C + h * W * C + w * C + c];
					if (a < 0)
						a = 0;

					A[n * H * W * C + h * W * C + w * C + c] = a;
				}
			}
		}
	}

	return;
}

// A = relu(A)
// A[N][H][W][C]
void Relu2D(MYINT *A, MYINT H, MYINT W)
{

	for (MYITE h = 0; h < H; h++)
	{
		for (MYITE w = 0; w < W; w++)
		{
			MYINT a = A[h * W + w];
			if (a < 0)
				a = 0;

			A[h * W + w] = a;
		}
	}

	return;
}

// B = maxpool(A)
// A[N][H][W][C], B[N][H][W][C]
void Maxpool(MYINT *A, MYINT *B, MYINT N, MYINT H, MYINT W, MYINT C, MYINT stride)
{
	MYITE HO = H / stride;
	MYITE WO = W / stride;

	for (MYITE n = 0; n < N; n++)
	{
		for (MYITE ho = 0; ho < HO; ho++)
		{
			for (MYITE wo = 0; wo < WO; wo++)
			{
				for (MYITE c = 0; c < C; c++)
				{

					MYINT max = A[n * H * W * C + (stride * ho) * W * C + (stride * wo) * C + c];
					for (MYITE hs = 0; hs < stride; hs++)
					{
						for (MYITE ws = 0; ws < stride; ws++)
						{
							MYINT a = A[n * H * W * C + ((stride * ho) + hs) * W * C + ((stride * wo) + ws) * C + c];
							if (a > max)
								max = a;
						}
					}

					B[n * HO * WO * C + ho * WO * C + wo * C + c] = max;
				}
			}
		}
	}

	return;
}

// B = exp(A)
void Exp(MYINT *A, MYINT I, MYINT J, MYINT shrA, MYINT shrB, MYINT *B)
{

	for (MYITE i = 0; i < I; i++)
	{
		for (MYITE j = 0; j < J; j++)
		{
			B[i * J + j] = ((MYINT)(exp(((float)A[i * J + j]) / shrA) * shrB));
		}
	}

	return;
}

// B = Sigmoid(A)
void Sigmoid(MYINT *A, MYINT I, MYINT J, MYINT div, MYINT add, MYINT sigmoid_limit, MYINT scale_in, MYINT scale_out, MYINT *B)
{

	MYINT scale_diff = scale_out / scale_in;

	for (MYITE i = 0; i < I; i++)
	{
		for (MYITE j = 0; j < J; j++)
		{
#ifdef FLOATEXP
			float x = float(A[i * J + j]) / scale_in;

			float y = 1 / (1 + exp(-x));

			MYINT z = MYINT(y * scale_out);

			B[i * J + j] = z;
#else
			MYINT x = A[i * J + j];

			x = (x / div) + add;

			MYINT y;
			if (x >= sigmoid_limit)
				y = sigmoid_limit;
			else if (x <= 0)
				y = 0;
			else
				y = x;

			y = y * scale_diff;

			B[i * J + j] = y;
#endif
		}
	}

	return;
}

// A = AdjustScaleShr(A)
void AdjustScaleShr(MYINT *A, MYINT I, MYINT J, MYINT scale)
{

	for (MYITE i = 0; i < I; i++)
	{
		for (MYITE j = 0; j < J; j++)
		{
			MYINT a = A[i * J + j];
			A[i * J + j] = a / scale;
		}
	}

	return;
}

// A = AdjustScaleShl(A)
void AdjustScaleShl(MYINT *A, MYINT I, MYINT J, MYINT scale)
{

	for (MYITE i = 0; i < I; i++)
	{
		for (MYITE j = 0; j < J; j++)
		{
			MYINT a = A[i * J + j];
			A[i * J + j] = a * scale;
		}
	}

	return;
}

MYINT treeSum(MYINT *arr, MYINT count, MYINT height_shr, MYINT height_noshr)
{
	if (count == 1)
		return arr[0];

	bool shr = true;

	for (MYITE depth = 0; depth < (height_shr + height_noshr); depth++)
	{
		if (depth >= height_shr)
			shr = false;

		for (MYITE index = 0; index < (count / 2); index++)
		{
			MYINT sum = arr[2 * index] + arr[(2 * index) + 1];

			if (shr)
				arr[index] = sum / 2;
			else
				arr[index] = sum;
		}

		if (count % 2 == 1)
		{
			MYITE index = (count / 2) + 1;
			if (shr)
				arr[index - 1] = arr[count - 1] / 2;
			else
				arr[index - 1] = arr[count - 1];
		}

		// Debugging
		if (count % 2 == 1)
			arr[count / 2 + 1] = 0;
		else
			arr[count / 2] = 0;

		count = (count + 1) >> 1;
	}

	return arr[0];
}
