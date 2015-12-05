package jml.operation;

import org.apache.commons.math.linear.RealMatrix;

/***
 * The {@code ArrayOperation} includes frequently used operation 
 * functions on {@code double} arrays. The argument vector is 
 * required to have been allocated memory before being used in 
 * the array operations.
 * 
 * @author Mingjie Qian
 * @version 1.0, Feb. 21st, 2013
 */
public class ArrayOperation {

	/**
	 * Compute the maximum argument.
	 * 
	 * @param V a {@code double} array
	 * 
	 * @return maximum argument
	 * 
	 */
	public static int argmax(double[] V) {

		int maxIdx = 0;
		double maxVal = V[0];
		for (int i = 1; i < V.length; i++) {
			if (maxVal < V[i]) {
				maxVal = V[i];
				maxIdx = i;
			}
		}
		return maxIdx;

	}
	
	/**
	 * Assign a 1D {@code double} array by a real scalar.
	 * 
	 * @param V a 1D {@code double} array
	 * 
	 * @param v a real scalar
	 * 
	 */
	public static void assignVector(double[] V, double v) {
		for (int i = 0; i < V.length; i++)
			V[i] = v;
	}
	
	/**
	 * Assign a 1D {@code int} array by a real scalar.
	 * 
	 * @param V a 1D {@code int} array
	 * 
	 * @param v a real scalar
	 * 
	 */
	public static void assignIntegerVector(int[] V, int v) {
		for (int i = 0; i < V.length; i++)
			V[i] = v;
	}

	/**
	 * Clear all elements of a 1D {@code double} array to zero.
	 * 
	 * @param V a {@code double} array
	 * 
	 */
	public static void clearVector(double[] V) {
		assignVector(V, 0);
	}

	/**
	 * Clear all elements of a 2D {@code double} array to zero.
	 * 
	 * @param M a 2D {@code double} array
	 * 
	 */
	public static void clearMatrix(double[][] M) {
		for (int i = 0; i < M.length; i++) {
			assignVector(M[i], 0);
		}
	}

	/**
	 * Allocate continuous memory block for a 1D {@code double}
	 * array.
	 * 
	 * @param n number of elements to be allocated
	 * 
	 * @return a 1D {@code double} array of length n
	 * 
	 */
	public static double[] allocateVector(int n) {
		double[] res = new double[n];
		assignVector(res, 0);
		return res;
	}
	
	/**
	 * Allocate continuous memory block for a 1D {@code double}
	 * array and assign all elements with a given value.
	 * 
	 * @param n number of elements to be allocated
	 * 
	 * @param v a real scalar to assign the 1D {@code double} array
	 * 
	 * @return a 1D {@code double} array of length n
	 * 
	 */
	public static double[] allocateVector(int n, double v) {
		double[] res = new double[n];
		assignVector(res, v);
		return res;
	}
	
	/**
	 * Allocate continuous memory block for a 1D {@code int}
	 * array.
	 * 
	 * @param n number of elements to be allocated
	 * 
	 * @return a 1D {@code int} array of length n
	 * 
	 */
	public static int[] allocateIntegerVector(int n) {
		int[] res = new int[n];
		assignIntegerVector(res, 0);
		return res;
	}

	/**
	 * Allocate memory for a 2D {@code double} array.
	 * 
	 * @param nRows number of rows
	 * 
	 * @param nCols number of columns
	 * 
	 * @return a nRows by nCols 2D {@code double} array
	 * 
	 */
	public static double[][] allocateMatrix(int nRows, int nCols) {
		double[][] res = new double[nRows][];
		for (int i = 0; i < nRows; i++) {
			res[i] = allocateVector(nCols);
		}
		return res;
	}

	/**
	 * Element-wise division and assignment operation. It divides
	 * the first argument with the second argument and assign
	 * the result to the first argument, i.e., V = V / v.
	 * 
	 * @param V a 1D {@code double} array
	 * 
	 * @param v a real scalar
	 * 
	 */
	public static void divideAssign(double[] V, double v) {
		for (int i = 0; i < V.length; i++)
			V[i] = V[i] / v;
	}
	
	/**
	 * Element-wise division and assignment operation. It divides
	 * the first argument with the second argument and assign
	 * the result to the first argument, i.e., V1 = V1 ./ V2.
	 * 
	 * @param V1 a 1D {@code double} array
	 * 
	 * @param V2 a 1D {@code double} array
	 * 
	 */
	public static void divideAssign(double[] V1, double[] V2) {
		for (int i = 0; i < V1.length; i++)
			V1[i] = V1[i] / V2[i];
	}

	/**
	 * Element-wise multiplication and assignment operation.
	 * It multiplies the first argument with the second argument
	 * and assign the result to the first argument, i.e., V = V * v.
	 * 
	 * @param V a 1D {@code double} array
	 * 
	 * @param v a real scalar
	 * 
	 */
	public static void timesAssign(double[] V, double v) {
		for (int i = 0; i < V.length; i++)
			V[i] = V[i] * v;
	}

	/**
	 * Element-wise multiplication and assignment operation.
	 * It multiplies the first argument with the second argument
	 * and assign the result to the first argument, i.e., V1 = V1 .* V2.
	 * 
	 * @param V1 a 1D {@code double} array
	 * 
	 * @param V2 a 1D {@code double} array
	 * 
	 */
	public static void timesAssign(double[] V1, double[] V2) {
		for (int i = 0; i < V1.length; i++)
			V1[i] = V1[i] * V2[i];
	}

	/**
	 * Compute the sum of a 1D {@code double} array.
	 * 
	 * @param V a 1D {@code double} array
	 * 
	 * @return sum(V)
	 * 
	 */
	public static double sum(double[] V) {
		double res = 0;
		for (int i = 0; i < V.length; i++)
			res += V[i];
		return res;
	}

	/**
	 * Sum a 1D {@code double} array to one, i.e., V[i] = V[i] / sum(V).
	 * 
	 * @param V a 1D {@code double} array
	 */
	public static void sum2one(double[] V) {
		divideAssign(V, sum(V));
	}
	
	/**
	 * Element-wise addition and assignment operation.
	 * It adds the first argument by the second argument
	 * and assign the result to the first argument, i.e., V = V + v.
	 * 
	 * @param V a 1D {@code double} array
	 * 
	 * @param v a real scalar
	 * 
	 */
	public static void plusAssign(double[] V, double v) {
		for (int i = 0; i < V.length; i++)
			V[i] = V[i] + v;
	}

	/**
	 * Element-wise addition and assignment operation.
	 * It adds the first argument by the second argument
	 * and assign the result to the first argument, i.e., V1 = V1 + V2.
	 * 
	 * @param V1 a 1D {@code double} array
	 * 
	 * @param V2 a 1D {@code double} array
	 * 
	 */
	public static void plusAssign(double[] V1, double[] V2) {
		for (int i = 0; i < V1.length; i++)
			V1[i] = V1[i] + V2[i];
	}
	
	/**
	 * Element-wise subtraction and assignment operation.
	 * It subtracts the first argument by the second argument
	 * and assign the result to the first argument, i.e., V = V - v.
	 * 
	 * @param V a 1D {@code int} array
	 * 
	 * @param v an integer
	 * 
	 */
	public static void minusAssign(int[] V, int v) {
		for (int i = 0; i < V.length; i++)
			V[i] = V[i] - v;
	}
	
	/**
	 * Element-wise subtraction and assignment operation.
	 * It subtracts the first argument by the second argument
	 * and assign the result to the first argument, i.e., V = V - v.
	 * 
	 * @param V a 1D {@code double} array
	 * 
	 * @param v a real scalar
	 * 
	 */
	public static void minusAssign(double[] V, double v) {
		for (int i = 0; i < V.length; i++)
			V[i] = V[i] - v;
	}

	/**
	 * Element-wise subtraction and assignment operation.
	 * It subtracts the first argument by the second argument
	 * and assign the result to the first argument, i.e., V1 = V1 - V2.
	 * 
	 * @param V1 a 1D {@code double} array
	 * 
	 * @param V2 a 1D {@code double} array
	 * 
	 */
	public static void minusAssign(double[] V1, double[] V2) {
		for (int i = 0; i < V1.length; i++)
			V1[i] = V1[i] - V2[i];
	}

	/**
	 * Element-wise assignment operation. It assigns the first argument
	 * with the second argument, i.e., V1 = V2.
	 * 
	 * @param V1 a 1D {@code double} array
	 * 
	 * @param V2 a 1D {@code double} array
	 * 
	 */
	public static void assignVector(double[] V1, double[] V2) {
		for (int i = 0; i < V1.length; i++)
			V1[i] = V2[i];
	}
	
	/**
	 * V1 = A * V2.
	 * 
	 * @param V1 a 1D {@code double} array
	 * 
	 * @param A a 2D {@code double} array
	 * 
	 * @param V2 a 1D {@code double} array
	 * 
	 */
	public static void operate(double[] V1, double[][] A, double[] V2) {
		
		double s = 0;
		for (int i = 0; i < V1.length; i++) {
			s = 0;
			for (int j = 0; j < V2.length; j++) {
				s += A[i][j] * V2[j];
			}
			V1[i] = s;
		}
		
	}
	
	/**
	 * V1 = A * V2.
	 * 
	 * @param V1 a 1D {@code double} array
	 * 
	 * @param A a real matrix
	 * 
	 * @param V2 a 1D {@code double} array
	 * 
	 */
	public static void operate(double[] V1, RealMatrix A, double[] V2) {
		
		double s = 0;
		for (int i = 0; i < V1.length; i++) {
			s = 0;
			for (int j = 0; j < V2.length; j++) {
				s += A.getEntry(i, j) * V2[j];
			}
			V1[i] = s;
		}
		
	}
	
	/**
	 * V1' = V2' * A.
	 * 
	 * @param V1 a 1D {@code double} array
	 * 
	 * @param V2 a 1D {@code double} array
	 * 
	 * @param A a 2D {@code double} array
	 * 
	 */
	public static void operate(double[] V1, double[] V2, double[][] A) {
		
		double s = 0;
		for (int j = 0; j < V1.length; j++) {
			s = 0;
			for (int i = 0; i < V2.length; i++) {
				s += V2[i] * A[i][j];
			}
			V1[j] = s;
		}
		
	}
	
	/**
	 * V1' = V2' * A.
	 * 
	 * @param V1 a 1D {@code double} array
	 * 
	 * @param V2 a 1D {@code double} array
	 * 
	 * @param A a 2D {@code double} array
	 * 
	 */
	public static void operate(double[] V1, double[] V2, RealMatrix A) {
		
		double s = 0;
		for (int j = 0; j < V1.length; j++) {
			s = 0;
			for (int i = 0; i < V2.length; i++) {
				s += V2[i] * A.getEntry(i, j);
			}
			V1[j] = s;
		}
		
	}
	
}
