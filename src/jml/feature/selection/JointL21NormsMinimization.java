package jml.feature.selection;

import static jml.matlab.Matlab.*;

import org.apache.commons.math.linear.RealMatrix;

/***
 * Joint l_{2,1}-norms minimization.
 * 
 * Optimization problem:
 * min_W gamma \ || X' * W - Y ||_{2,1} + || W ||_{2,1}
 * 
 * @author Mingjie Qian
 * @version 1.0, Feb. 4th, 2012
 */
public class JointL21NormsMinimization extends SupervisedFeatureSelection{

	public double gamma;

	public JointL21NormsMinimization(double gamma) {
		super();
		this.gamma = gamma;
	}

	@Override
	public void run() {

		int n = X.getColumnDimension();
		int d = X.getRowDimension();
		RealMatrix I_n = eye(n);
		RealMatrix XI = horzcat(X.transpose(), times(gamma, I_n));
		RealMatrix A = XI;
		int m = n + d;
		int maxIter = 100;
		RealMatrix D = eye(m);

		RealMatrix D_inv = null;
		RealMatrix T = null;
		RealMatrix U = null;
		RealMatrix D_pre = null;
		RealMatrix U_pre = null;

		boolean verbose = !true;
		double ofv = 0;

		int k = 0;
		while (true) {

			if (k > maxIter)
				break;

			D_inv = repmat(rdivide(1, diag(D)), 1, n);

			T = times(D_inv, A.transpose());
			U = mtimes(T, mldivide(mtimes(A, T), Y));

			D_pre = D;
			D = diag(rdivide(0.5, plus(l2NormByRows(U), eps)));

			if (verbose) {
				fprintf("||D_{k+1} - D_{k}||: %f\n", norm(minus(D_pre, D)));
			}

			W = U.getSubMatrix(colon(0, d - 1), colon(0, U.getColumnDimension() - 1));
			
			if (verbose) {
				ofv = sum(l2NormByRows(X.transpose().multiply(W).subtract(Y))).getEntry(0, 0)/ gamma + sum(l2NormByRows(W)).getEntry(0, 0);
				fprintf("ofv: %f\n", ofv);
			}

			if (k > 0)
				fprintf("Iter %d: ||U_{k+1} - U_{k}||: %f\n", k, norm(minus(U_pre, U)));

			U_pre = U;

			k = k + 1;

		}

	}

	/**
	 * @param args
	 */
	public static void main(String[] args) {

		double[][] data = { {3.5, 4.4, 1.3},
							{5.3, 2.2, 0.5},
							{0.2, 0.3, 4.1},
							{-1.2, 0.4, 3.2} };

		double[][] labels = { {1, 0, 0},
							  {0, 1, 0},
							  {0, 0, 1} };

		SupervisedFeatureSelection robustFS = new JointL21NormsMinimization(2.0);
		robustFS.feedData(data);
		robustFS.feedLabels(labels);
		long start = System.currentTimeMillis();
		robustFS.run();
		
		double elapsedTime = (System.currentTimeMillis() - start) / 1000d;
		System.out.format("Elapsed time: %.3f seconds\n", elapsedTime);
		
		System.out.println("Projection matrix:");
		display(robustFS.getW());
		
	}

}
