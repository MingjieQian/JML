package jml.kernel;

import org.apache.commons.math.linear.RealMatrix;

import static jml.matlab.Matlab.*;

/***
 * Java implementation of commonly used kernel functions.
 * 
 * @version 1.0 Mar. 29th, 2013
 * @author Mingjie Qian
 */
public class Kernel {

	/**
	 * @param args
	 */
	public static void main(String[] args) {

	}
	
	/**
	 * Computes Gram matrix of a specified kernel. Given a data matrix
	 * X (d x n), it returns Gram matrix K (n x n).
	 * 
	 * @param kernelType 'linear' | 'poly' | 'rbf' | 'cosine'
	 * 
	 * @param kernelParam   --    | degree | sigma |    --
	 * 
	 * @param X a matrix

	 * @return Gram matrix (n x n)
	 * 
	 */
	public static RealMatrix calcKernel(String kernelType, 
			double kernelParam, RealMatrix X) {
		return calcKernel(kernelType, kernelParam, X, X);
	}

	/**
	 * Computes Gram matrix of a specified kernel. Given two data matrices
	 * X1 (d x n1), X2 (d x n2), it returns Gram matrix K (n1 x n2).
	 * 
	 * @param kernelType 'linear' | 'poly' | 'rbf' | 'cosine'
	 * 
	 * @param kernelParam   --    | degree | sigma |    --
	 * 
	 * @param X1 a matrix
	 * 
	 * @param X2 a matrix
	 * 
	 * @return Gram matrix (n1 x n2)
	 * 
	 */
	public static RealMatrix calcKernel(String kernelType, 
			double kernelParam, RealMatrix X1, RealMatrix X2) {
		
		RealMatrix K = null;
		if (kernelType.equals("linear")) {
			K = X1.transpose().multiply(X2);
		} else if (kernelType.equals("cosine")) {
			RealMatrix A = X1;
			RealMatrix B = X2;
			RealMatrix AA = sum(times(A, A));
			RealMatrix BB = sum(times(B, B));
			RealMatrix AB = A.transpose().multiply(B);
			K = dotMultiply(scalarDivide(1, sqrt(kron(AA.transpose(), BB))), AB);
		} else if (kernelType.equals("poly")) {
			K = power(X1.transpose().multiply(X2), kernelParam);
		} else if (kernelType.equals("rbf")) {
			K = exp(l2DistanceSquare(X1, X2).scalarMultiply(-1 / (2 * Math.pow(kernelParam, 2))));
		}
		return K;

	}

}
