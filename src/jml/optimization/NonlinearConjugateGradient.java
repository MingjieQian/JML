package jml.optimization;

import static jml.matlab.Matlab.innerProduct;
import static jml.matlab.Matlab.minus;
import static jml.matlab.Matlab.norm;
import static jml.matlab.Matlab.plus;
import static jml.matlab.Matlab.setMatrix;
import static jml.matlab.Matlab.times;
import static jml.matlab.Matlab.uminus;

import java.util.ArrayList;
import org.apache.commons.math.linear.RealMatrix;

/**
 * A Java implementation for the nonlinear conjugate gradient method.
 * It is a general algorithm interface, only gradient and objective
 * function value are needed to compute outside the class.
 * </p>
 * A simple example: </br></br>
 * <code>
 * W = repmat(zeros(nFea, 1), new int[]{1, K}); </br>
 * A = X.transpose().multiply(W); </br>
 * V = sigmoid(A); </br>
 * G = X.multiply(V.subtract(Y)).scalarMultiply(1.0 / nSample); </br>
 * fval = -sum(sum(times(Y, log(plus(V, eps))))).getEntry(0, 0) / nSample; </br>
 * </br>
 * boolean flags[] = null; </br>
 * while (true) { </br>
 * &nbsp flags = NonlinearConjugateGradient.run(G, fval, epsilon, W); </br>
 * &nbsp if (flags[0]) </br>
 * &nbsp &nbsp break; </br>
 * &nbsp A = X.transpose().multiply(W); </br>
 * &nbsp V = sigmoid(A); </br>
 * &nbsp fval = -sum(sum(times(Y, log(plus(V, eps))))).getEntry(0, 0) / nSample; </br>
 * &nbsp if (flags[1]) </br>
 * &nbsp &nbsp G = rdivide(X.multiply(V.subtract(Y)), nSample); </br>
 * } </br>
 * </br>
 * </code>
 * 
 * @version 1.0 Feb. 25th, 2013
 * 
 * @author Mingjie Qian
 */
public class NonlinearConjugateGradient {

	/**
	 * Current gradient.
	 */
	private static RealMatrix G = null;

	/**
	 * Last gradient.
	 */
	private static RealMatrix G_pre = null;

	/**
	 * Current matrix variable that we want to optimize.
	 */
	private static RealMatrix X = null;

	/**
	 * Decreasing step.
	 */
	private static RealMatrix p = null;

	/**
	 * The last objective function value.
	 */
	private static double fval = 0;

	/**
	 * If gradient is required for the next step.
	 */
	private static boolean gradientRequired = false;

	/**
	 * If the algorithm converges or not.
	 */
	private static boolean converge = false;

	/**
	 * State for the automata machine.
	 * 0: Initialization
	 * 1: Before backtracking line search
	 * 2: Backtracking line search
	 * 3: After backtracking line search
	 * 4: Convergence
	 */
	private static int state = 0;

	/**
	 * Step length for backtracking line search.
	 */
	private static double t = 1;

	/**
	 * A temporary variable holding the inner product of the decreasing step p
	 * and the gradient G, it should be always non-positive.
	 */
	private static double z = 0;

	/**
	 * Iteration counter.
	 */
	private static int k = 0;

	private static double alpha = 0.05;

	private static double rou = 0.9;

	/**
	 * Formula used to calculate beta.
	 * 1: FR
	 * 2: PR
	 * 3: PR+
	 * 4: HS
	 */
	private static int formula = 4;

	/**
	 * An array holding the sequence of objective function values. 
	 */
	private static ArrayList<Double> J = new ArrayList<Double>();

	/**
	 * Main entry for the algorithm. The matrix variable to be 
	 * optimized will be updated in place to a better solution 
	 * point with lower objective function value.
	 * 
	 * @param Grad_t gradient at original X_t, required on the
	 *               first revocation
	 * 
	 * @param fval_t objective function value on original X_t
	 * 
	 * @param epsilon convergence precision
	 * 
	 * @param X_t current matrix variable to be optimized, will be
	 *            updated in place to a better solution point with
	 *            lower objective function value.
	 * 
	 * @return a {@code boolean} array of two elements: {converge, gradientRequired}
	 * 
	 */
	public static boolean[] run(RealMatrix Grad_t, double fval_t, double epsilon, RealMatrix X_t) {

		// If the algorithm has converged, we do a new job
		if (state == 4) {
			J.clear();
			state = 0;
		}

		if (state == 0) {

			X = X_t.copy();
			if (Grad_t == null) {
				System.err.println("Gradient is required on the first call!");
				System.exit(1);
			}
			G = Grad_t.copy();
			fval = fval_t;
			if (Double.isNaN(fval)) {
				System.err.println("Object function value is nan!");
				System.exit(1);
			}
			System.out.format("Initial ofv: %g\n", fval);

			p = uminus(G);

			state = 1;

		}

		if (state == 1) {

			double norm_Grad = norm(G);
			if (norm_Grad < epsilon) {
				converge = true;
				gradientRequired = false;
				state = 4;
				System.out.printf("CG converges with norm(Grad) %f\n", norm_Grad);
				return new boolean[] {converge, gradientRequired};
			}		

			t = 1;
			// z is always less than 0
			z = innerProduct(G, p);

			state = 2;

			// X_t.setSubMatrix(plus(X, times(t, p)).getData(), 0, 0);
			setMatrix(X_t, plus(X, times(t, p)));

			converge = false;
			gradientRequired = false;

			return new boolean[] {converge, gradientRequired};

		}

		// Backtracking line search
		if (state == 2) {

			converge = false;

			if (fval_t <= fval + alpha * t * z) {
				gradientRequired = true;
				state = 3;
			} else {
				t = rou * t;
				gradientRequired = false;
				// X_t.setSubMatrix(plus(X, times(t, p)).getData(), 0, 0);
				setMatrix(X_t, plus(X, times(t, p)));
			}	

			// We don't need to compute X_t again since the X_t has already
			// satisfied the Armijo condition.
			// X_t.setSubMatrix(plus(X, times(t, p)).getData(), 0, 0);

			return new boolean[] {converge, gradientRequired};

		}

		if (state == 3) {

			// X_pre = X.copy();   
			G_pre = G.copy();

			/*if (Math.abs(fval_t - fval) < 1e-256) {
				converge = true;
				gradientRequired = false;
				System.out.printf("Objective function value doesn't decrease, iteration stopped!\n");
				System.out.format("Iter %d, ofv: %g, norm(Grad): %g\n", k + 1, fval, norm(G));
				return new boolean[] {converge, gradientRequired};
			}*/

			fval = fval_t;
			J.add(fval);
			System.out.format("Iter %d, ofv: %g, norm(Grad): %g\n", k + 1, fval, norm(G));

			X = X_t.copy();
			G = Grad_t.copy();

			RealMatrix y_k = null;
			y_k = minus(G, G_pre);
			double beta = 0;
			switch (formula) {
			case 1:
				beta = innerProduct(G, G) / innerProduct(G_pre, G);
			case 2:
				beta = innerProduct(G, y_k) / innerProduct(G_pre, G_pre);
			case 3:
				beta = Math.max(innerProduct(G, y_k) / innerProduct(G_pre, G_pre), 0);
			case 4:
				beta = innerProduct(G, y_k) / innerProduct(y_k, p);
			case 5:
				beta = innerProduct(G, G) / innerProduct(y_k, p);
			default:
				beta = innerProduct(G, y_k) / innerProduct(y_k, p);
			}
			
			p = uminus(G).add(times(beta, p));

			k = k + 1;

			state = 1;

		}

		converge = false;
		gradientRequired = false;
		return new boolean[] {converge, gradientRequired};

	}

}
