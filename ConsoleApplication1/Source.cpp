#include <iostream>

#include <iomanip>

#include <stdlib.h>

#include <time.h>

#include <omp.h>

#define dim 200

int q = 200;

using namespace std;

double GetNorm(double ** A)

{

	double norm = 0;

	for (int i = 0; i < dim; i++)

	{

		for (int j = 0; j < dim; j++)

		{

			norm += A[i][j] * A[i][j];

		}

	}

	return sqrt(norm);

}

void PrintAngle(double **matr)

{

	cout << matr[0][0] << " " << matr[0][dim - 1] << endl;

	cout << matr[dim - 1][0] << " " << matr[dim - 1][dim - 1] << endl;

}

void Init(double **A, double **B)
{
	srand(time(NULL));
	for (int i = 0; i < dim; i++)
	{
		for (int j = 0; j < dim; j++)
		{
			A[i][j] = -0.5 + (rand() % 100) / (100 * 1.0);
			B[i][j] = -0.5 + (rand() % 100) / (100 * 1.0);
		}
	}
}
void min(double **A, double **B, double **C)
{
	for (int i = 0; i < dim; i++)
		for (int j = 0; j < dim; j++)
			for (int k = 0; k < dim; k++)
			{
				C[i][j] += A[i][k] * B[k][j];
			}
}


void min_i(double **A, double **B, double **C)
{
	omp_set_num_threads(4);
#pragma omp parallel for
	for (int i = 0; i < dim; i++)
		for (int j = 0; j < dim; j++)
			for (int k = 0; k < dim; k++)
			{
				C[i][j] += A[i][k] * B[k][j];
			}
}

void min_j(double **A, double **B, double **C)
{
	omp_set_num_threads(4);
	for (int i = 0; i < dim; i++)
#pragma omp parallel for
		for (int j = 0; j < dim; j++)
			for (int k = 0; k < dim; k++)
			{
				C[i][j] += A[i][k] * B[k][j];
			}
}

void min_k(double **A, double **B, double **C)
{
	int k;
	double c=0;
	omp_set_num_threads(4);
	for (int i = 0; i < dim; i++)
		for (int j = 0; j < dim; j++)
		{
			c = 0;
#pragma omp parallel for reduction(+:c)
			for (k = 0; k < dim; k++)
			{
				c += A[i][k] * B[k][j];
			}
			C[i][j] = c;
		}
}

int main()

{

	double

		**A = (double**)calloc(dim, sizeof(double)),

		**B = (double**)calloc(dim, sizeof(double)),

		**C = (double**)calloc(dim, sizeof(double));

	for (int i = 0; i < dim; i++)

	{

		A[i] = (double*)calloc(dim, sizeof(double));

		B[i] = (double*)calloc(dim, sizeof(double));

		C[i] = (double*)calloc(dim, sizeof(double));

	}

	Init(A, B);

	cout << "Matrix A:" << endl;

	PrintAngle(A);

	cout << endl;

	cout << "Matrix B:" << endl;

	PrintAngle(B);

	cout << endl;

	double time;

	time = clock();

	for (int i = 0; i < q; i++)

	{

		min(A, B, C);

	}

	time = clock() - time;

	cout << "Time:" << time / CLOCKS_PER_SEC << endl << endl;
	cout << "Norm: " << GetNorm(C) << endl << endl;
	for (int i = 0; i < dim; i++)
	{
		for (int j = 0; j < dim; j++)
		{
			C[i][j] = 0;
		}
	}

	//-----------------------------------------------

	time = clock();

	for (int i = 0; i < q; i++)

	{

		min_i(A, B, C);

	}

	time = clock() - time;

	cout << "Time:" << time / CLOCKS_PER_SEC << endl << endl;
	cout << "Norm: " << GetNorm(C) << endl << endl;
	for (int i = 0; i < dim; i++)
	{
		for (int j = 0; j < dim; j++)
		{
			C[i][j] = 0;
		}
	}

	//-----------------------------------------------

	time = clock();

	for (int i = 0; i < q; i++)

	{

		min_j(A, B, C);

	}

	time = clock() - time;

	cout << "Time:" << time / CLOCKS_PER_SEC << endl << endl;
	cout << "Norm: " << GetNorm(C) << endl << endl;
	for (int i = 0; i < dim; i++)

	{

		for (int j = 0; j < dim; j++)

		{

			C[i][j] = 0;

		}

	}

	//-----------------------------------------------
	time = clock();
	for (int i = 0; i < q; i++)
	{
		min_k(A, B, C);
	}
	time = clock() - time;
	cout << "Time:" << time / CLOCKS_PER_SEC << endl << endl;
	cout << "Norm: " << GetNorm(C) << endl << endl;
	cout << "Result:" << endl;
	PrintAngle(C);
	cout << endl;
	free(A);
	free(B);
	free(C);
	return 0;

}