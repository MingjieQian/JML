package jml.recovery;

import static jml.matlab.Matlab.abs;
import static jml.matlab.Matlab.disp;
import static jml.matlab.Matlab.fprintf;
import static jml.matlab.Matlab.max;
import static jml.matlab.Matlab.minus;
import static jml.matlab.Matlab.mtimes;
import static jml.matlab.Matlab.norm;
import static jml.matlab.Matlab.plus;
import static jml.matlab.Matlab.rand;
import static jml.matlab.Matlab.randn;
import static jml.matlab.Matlab.randperm;
import static jml.matlab.Matlab.rank;
import static jml.matlab.Matlab.rdivide;
import static jml.matlab.Matlab.reshape;
import static jml.matlab.Matlab.setSubMatrix;
import static jml.matlab.Matlab.size;
import static jml.matlab.Matlab.svd;
import static jml.matlab.Matlab.times;
import static jml.matlab.Matlab.vec;
import static jml.matlab.Matlab.zeros;
import static jml.optimization.ShrinkageOperator.shrinkage;
import static jml.utils.Time.*;

import org.apache.commons.math.linear.RealMatrix;

/***
 * A Java implementation of robust PCA which solves the 
 * following convex optimization problem:
 * </br>
 * min ||A||_* + lambda ||E||_1</br>
 * s.t. D = A + E</br>
 * where ||.||_* denotes the nuclear norm of a matrix (i.e., 
 * the sum of its singular values), and ||.||_1 denotes the 
 * sum of the absolute values of matrix entries.</br>
 * </br>
 * Inexact augmented Lagrange multipliers is used to solve the optimization
 * problem due to its empirically fast convergence speed and proved convergence 
 * to the true optimal solution.
 * 
 * <b>Input:</b></br>
 *    D: an observation matrix with columns as data vectors</br>
 *    lambda: a positive weighting parameter</br>
 *    
 * <b>Output:</b></br>
 *    A: a low-rank matrix recovered from the corrupted data matrix D</br>
 *    E: error matrix between D and A</br>
 * 
 * @author Mingjie Qian
 * @version 1.0 Nov. 20th, 2013
 */
public class RobustPCA {
	
	/**
	 * @param args
	 */
	public static void main(String[] args) {
		
		int m = 8;
		int r = m / 4;
		
		RealMatrix L = randn(m, r);
		RealMatrix R = randn(m, r);
		
		RealMatrix A_star = mtimes(L, R.transpose());
		RealMatrix E_star = zeros(size(A_star));
		int[] indices = randperm(m * m);
		int nz = m * m / 20;
		int[] nz_indices = new int[nz];
		for (int i = 0; i < nz; i++) {
			nz_indices[i] = indices[i] - 1;
		}
		RealMatrix E_vec = vec(E_star);
		setSubMatrix(E_vec, nz_indices, new int[] {0}, (minus(rand(nz, 1), 0.5).scalarMultiply(100)));
		E_star = reshape(E_vec, size(E_star));
		
		// Input
		RealMatrix D = A_star.add(E_star);
		double lambda = 1 * Math.pow(m, -0.5);
		
		// Run Robust PCA
		RobustPCA robustPCA = new RobustPCA(lambda);
		robustPCA.feedData(D);
		tic();
		robustPCA.run();
		fprintf("Elapsed time: %.2f seconds.%n", toc());
		
		// Output
		RealMatrix A_hat = robustPCA.GetLowRankEstimation();
		RealMatrix E_hat = robustPCA.GetErrorMatrix();
		
		fprintf("A*:\n");
		disp(A_star, 4);
		fprintf("A^:\n");
		disp(A_hat, 4);
		fprintf("E*:\n");
		disp(E_star, 4);
		fprintf("E^:\n");
		disp(E_hat, 4);
		fprintf("rank(A*): %d\n", rank(A_star));
		fprintf("rank(A^): %d\n", rank(A_hat));
		fprintf("||A* - A^||_F: %.4f\n", norm(A_star.subtract(A_hat), "fro"));
		fprintf("||E* - E^||_F: %.4f\n", norm(E_star.subtract(E_hat), "fro"));
		
	}
	
	/**
	 * A positive weighting parameter.
	 */
	double lambda;

	/**
	 * Observation real matrix.
	 */
	RealMatrix D;
	
	/**
	 * A low-rank matrix recovered from the corrupted data observation matrix D.
	 */
	RealMatrix A;
	
	/**
	 * Error matrix between the original observation matrix D and the low-rank
	 * recovered matrix A.
	 */
	RealMatrix E;
	
	/**
	 * Constructor for Robust PCA.
	 * 
	 * @param lambda a positive weighting parameter, larger value leads to sparser
	 * 				 error matrix
	 */
	public RobustPCA(double lambda) {
		this.lambda = lambda;
	}
	
	/**
	 * Feed an observation matrix.
	 * 
	 * @param D a real matrix
	 */
	public void feedData(RealMatrix D) {
		this.D = D;
	}
	
	/**
	 * Run robust PCA.
	 */
	public void run() {
		RealMatrix[] res = robustPCA(D, lambda);
		A = res[0];
		E = res[1];
	}
	
	/**
	 * Get the low-rank matrix recovered from the corrupted data 
	 * observation matrix.
	 * 
	 * @return low-rank approximation
	 */
	public RealMatrix GetLowRankEstimation() {
		return A;
	}
	
	/**
	 * Get the error matrix between the original observation matrix 
	 * and its low-rank recovered matrix.
	 * 
	 * @return error matrix
	 */
	public RealMatrix GetErrorMatrix() {
		return E;
	}
	
	/**
	 * Compute robust PCA for an observation matrix which solves the 
	 * following convex optimization problem:
	 * </br>
	 * min ||A||_* + lambda ||E||_1</br>
	 * s.t. D = A + E</br>
	 * where ||.||_* denotes the nuclear norm of a matrix (i.e., 
	 * the sum of its singular values), and ||.||_1 denotes the 
	 * sum of the absolute values of matrix entries.</br>
	 * </br>
	 * Inexact augmented Lagrange multipliers is used to solve the optimization
	 * problem due to its empirically fast convergence speed and proved convergence 
	 * to the true optimal solution.
	 * 
	 * @param D a real observation matrix
	 * 
	 * @param lambda a positive weighting parameter, larger value leads to sparser
	 * 				 error matrix
	 * @return a {@code RealMatrix} array [A, E] where A is the recovered low-rank
	 * 		   approximation of D, and E is the error matrix between A and D
	 * 
	 */
	public RealMatrix[] robustPCA(RealMatrix D, double lambda) {
		
		RealMatrix Y = rdivide(D, J(D, lambda));
		RealMatrix E = zeros(size(D));
		RealMatrix A = minus(D, E);
		double mu = 1.25 / norm(D, 2);
		double rou = 1.6;
		int k = 0;
		double norm_D = norm(D, "fro");
		double e1 = 1e-7;
		double e2 = 1e-6;
		double c1 = 0;
		double c2 = 0;
		double mu_old = 0;
		RealMatrix E_old = null;
		RealMatrix[] SVD = null;
		
		while (true) {
			
			// Stopping criteria
			if (k > 0) {
				c1 = norm(D.subtract(A).subtract(E), "fro") / norm_D;
				c2 = mu_old * norm(E.subtract(E_old), "fro") / norm_D;
				// fprintf("k = %d, c2: %.4f%n", k, c2);
				if (c1 <= e1 && c2 <= e2)
					break;
			}
			
			E_old = E;
			mu_old = mu;
			
			// E_{k+1} = argmin_E L(A_k, E, Y_k, mu_k)
		    E = shrinkage(plus(minus(D, A), rdivide(Y, mu)), lambda / mu);
		    
		    // A_{k+1} = argmin_A L(A, E_{k+1}, Y_k, mu_k)
		    SVD = svd(plus(minus(D, E), rdivide(Y, mu)));
		    A = SVD[0].multiply(shrinkage(SVD[1], 1 / mu)).multiply(SVD[2].transpose());

		    Y = Y.add(times(mu, D.subtract(A).subtract(E)));
		    
		    // Update mu_k to mu_{k+1}

		    if (norm(E.subtract(E_old), "fro") * mu / norm_D < e2)
		    	mu = rou * mu;

		    k = k + 1;

		}

		RealMatrix[] res = new RealMatrix[2];
		res[0] = A;
		res[1] = E;
		return res;
		
	}
	
	private static double J(RealMatrix Y, double lambda) {
		return Math.max(norm(Y, 2), max(max(abs(Y))).getEntry(0, 0) / lambda);
	}

}
