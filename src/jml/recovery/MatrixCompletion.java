package jml.recovery;

import static jml.matlab.Matlab.colon;
import static jml.matlab.Matlab.disp;
import static jml.matlab.Matlab.fprintf;
import static jml.matlab.Matlab.gt;
import static jml.matlab.Matlab.linearIndexing;
import static jml.matlab.Matlab.linearIndexingAssignment;
import static jml.matlab.Matlab.logicalIndexingAssignment;
import static jml.matlab.Matlab.minus;
import static jml.matlab.Matlab.mtimes;
import static jml.matlab.Matlab.norm;
import static jml.matlab.Matlab.plus;
import static jml.matlab.Matlab.randn;
import static jml.matlab.Matlab.randperm;
import static jml.matlab.Matlab.rank;
import static jml.matlab.Matlab.rdivide;
import static jml.matlab.Matlab.size;
import static jml.matlab.Matlab.sumAll;
import static jml.matlab.Matlab.svd;
import static jml.matlab.Matlab.times;
import static jml.matlab.Matlab.zeros;
import static jml.operation.ArrayOperation.minusAssign;
import static jml.optimization.ShrinkageOperator.shrinkage;
import static jml.utils.Time.tic;
import static jml.utils.Time.toc;
import jml.data.Data;

import org.apache.commons.math.linear.OpenMapRealMatrix;
import org.apache.commons.math.linear.RealMatrix;

/***
 * A Java implementation of matrix completion which solves the 
 * following convex optimization problem:
 * </br>
 * min ||A||_*</br>
 * s.t. D = A + E</br>
 *      E(Omega) = 0</br>
 * where ||.||_* denotes the nuclear norm of a matrix (i.e., 
 * the sum of its singular values).</br>
 * </br>
 * Inexact augmented Lagrange multiplers is used to solve the optimization
 * problem due to its empirically fast convergence speed and proved convergence 
 * to the true optimal solution.
 * 
 * <b>Input:</b></br>
 *    D: an observation matrix with columns as data vectors</br>
 *    Omega: a sparse or dense logical matrix indicating the indices of samples</br>
 *    
 * <b>Output:</b></br>
 *    A: a low-rank matrix completed from the data matrix D</br>
 *    E: error matrix between D and A</br>
 * 
 * @author Mingjie Qian
 * @version 1.0 Nov. 22nd, 2013
 */
public class MatrixCompletion {

	/**
	 * @param args
	 */
	public static void main(String[] args) {
		
		int m = 6;
		int r = 1;
		int p = (int) Math.round(m * m * 0.3);
		
		RealMatrix L = randn(m, r);
		RealMatrix R = randn(m, r);
		RealMatrix A_star = mtimes(L, R.transpose());
		
		int[] indices = randperm(m * m);
		minusAssign(indices, 1);
		indices = linearIndexing(indices, colon(0, p - 1));
		
		RealMatrix Omega = zeros(size(A_star));
		linearIndexingAssignment(Omega, indices, 1);
		
		RealMatrix D = zeros(size(A_star));
		linearIndexingAssignment(D, indices, linearIndexing(A_star, indices));
				
		RealMatrix E_star = D.subtract(A_star);
		logicalIndexingAssignment(E_star, Omega, 0);
		
		D = Data.loadMatrix("D.txt");
		Omega = Data.loadMatrix("Omega.txt");
		
		// Run matrix completion
		MatrixCompletion matrixCompletion = new MatrixCompletion();
		matrixCompletion.feedData(D);
		matrixCompletion.feedIndices(Omega);
		tic();
		matrixCompletion.run();
		fprintf("Elapsed time: %.2f seconds.%n", toc());
		
		// Output
		RealMatrix A_hat = matrixCompletion.GetLowRankEstimation();
		
		fprintf("A*:\n");
		disp(A_star, 4);
		fprintf("A^:\n");
		disp(A_hat, 4);
		fprintf("D:\n");
		disp(D, 4);
		fprintf("rank(A*): %d\n", rank(A_star));
		fprintf("rank(A^): %d\n", rank(A_hat));
		fprintf("||A* - A^||_F: %.4f\n", norm(A_star.subtract(A_hat), "fro"));
		
	}

	/**
	 * Observation real matrix.
	 */
	RealMatrix D;
	
	/**
	 * a sparse or dense logical matrix indicating the indices of samples
	 */
	RealMatrix Omega;
	
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
	 * Constructor.
	 */
	public MatrixCompletion() {
		
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
	 * Feed indices of samples.
	 * 
	 * @param Omega a sparse or dense logical matrix indicating the indices of samples
	 * 
	 */
	public void feedIndices(RealMatrix Omega) {
		this.Omega = Omega;
	}
	
	/**
	 * Feed indices of samples.
	 * 
	 * @param indices an {@code int} array for the indices of samples
	 * 
	 */
	public void feedIndices(int[] indices) {
		Omega = new OpenMapRealMatrix(size(D, 1), size(D, 2));
		linearIndexingAssignment(Omega, indices, 1);
	}
	
	/**
	 * Run matrix completion.
	 */
	public void run() {
		RealMatrix[] res = matrixCompletion(D, Omega);
		A = res[0];
		E = res[1];
	}
	
	/**
	 * Get the low-rank completed matrix.
	 * 
	 * @return the low-rank completed matrix
	 */
	public RealMatrix GetLowRankEstimation() {
		return A;
	}
	
	/**
	 * Get the error matrix between the original observation matrix and 
	 * its low-rank completed matrix.
	 * 
	 * @return error matrix
	 */
	public RealMatrix GetErrorMatrix() {
		return E;
	}

	/**
	 * Do matrix completion which solves the following convex 
	 * optimization problem:
	 * </br>
	 * min ||A||_*</br>
	 * s.t. D = A + E</br>
	 *      E(Omega) = 0</br>
	 * where ||.||_* denotes the nuclear norm of a matrix (i.e., 
	 * the sum of its singular values).</br>
	 * </br>
	 * Inexact augmented Lagrange multipliers is used to solve the optimization
	 * problem due to its empirically fast convergence speed and proved convergence 
	 * to the true optimal solution.
	 * 
	 * @param D a real observation matrix
	 * 
	 * @param Omega a sparse or dense logical matrix indicating the indices of samples
	 * 
	 * @return a {@code RealMatrix} array [A, E] where A is the low-rank
	 * 		   completion from D, and E is the error matrix between A and D
	 * 
	 */
	public RealMatrix[] matrixCompletion(RealMatrix D, RealMatrix Omega) {
		
		RealMatrix Y = zeros(size(D));
		RealMatrix E = zeros(size(D));
		RealMatrix A = minus(D, E);
		int m = size(D, 1);
		int n = size(D, 2);
		double mu = 1 / norm(D, 2);
		// Sampling density
		double rou_s = sumAll(gt(Omega, 0)) / (m * n);
		// The relation between rou and rou_s is obtained by regression
		double rou = 1.2172 + 1.8588 * rou_s;
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
			if (k > 1) {
				c1 = norm(D.subtract(A).subtract(E), "fro") / norm_D;
				c2 = mu_old * norm(E.subtract(E_old), "fro") / norm_D;
				// fprintf("k = %d, c2: %.4f%n", k, c2);
				if (c1 <= e1 && c2 <= e2)
					break;
			}
			
			E_old = E;
			mu_old = mu;
		    
		    // A_{k+1} = argmin_A L(A, E_k, Y_k, mu_k)
		    SVD = svd(plus(minus(D, E), rdivide(Y, mu)));
		    // disp(full(SVD[1]));
		    A = SVD[0].multiply(shrinkage(SVD[1], 1 / mu)).multiply(SVD[2].transpose());

		    // E_{k+1} = argmin_E L(A_{k+1}, E, Y_k, mu_k)
		    E = D.subtract(A);
		    logicalIndexingAssignment(E, Omega, 0);
		    
		    Y = Y.add(times(mu, D.subtract(A).subtract(E)));
		    /*disp("Y:");
		    disp(Y);*/
		    
		    // Update mu_k to mu_{k+1}

		    if (norm(E.subtract(E_old), "fro") * mu / norm_D < e2)
		    	mu = rou * mu;

		    // fprintf("mu: %f%n", mu);
		    
		    k = k + 1;

		}

		RealMatrix[] res = new RealMatrix[2];
		res[0] = A;
		res[1] = E;
		return res;
		
	}
	
	/**
	 * Do matrix completion which solves the following convex 
	 * optimization problem:
	 * </br>
	 * min ||A||_*</br>
	 * s.t. D = A + E</br>
	 *      E(Omega) = 0</br>
	 * where ||.||_* denotes the nuclear norm of a matrix (i.e., 
	 * the sum of its singular values).</br>
	 * </br>
	 * Inexact augmented Lagrange multipliers is used to solve the optimization
	 * problem due to its empirically fast convergence speed and proved convergence 
	 * to the true optimal solution.
	 * 
	 * @param D a real observation matrix
	 * 
	 * @param indices an {@code int} array for the indices of samples
	 * 
	 * @return a {@code RealMatrix} array [A, E] where A is the low-rank
	 * 		   completion from D, and E is the error matrix between A and D
	 * 
	 */
	public RealMatrix[] matrixCompletion(RealMatrix D, int[] indices) {
		RealMatrix Omega = new OpenMapRealMatrix(size(D, 1), size(D, 2));
		linearIndexingAssignment(Omega, indices, 1);
		return matrixCompletion(D, Omega);
	}
	
}
