package jml.optimization;

import org.apache.commons.math.linear.RealMatrix;

/**
 * A Java implementation for the accelerated gradient descent method.
 * It is a general algorithm interface, only gradient and objective
 * function value are needed to compute outside the class.
 * </p>
 * A simple example: </br></br>
 * <code>
 * double epsilon = ...; // Convergence tolerance</br>
 * Matrix W = ...; // Initial matrix (vector) you want to optimize</br>
 * Matrix G = ...; // Gradient at the initial matrix (vector) you want to optimize</br>
 * double fval = ...; // Initial objective function value</br>
 * </br>
 * boolean flags[] = null; </br>
 * while (true) { </br>
 * &nbsp flags = AcceleratedGradientDescent.run(G, fval, epsilon, W); // Update W in place</br>
 * &nbsp if (flags[0]) // flags[0] indicates if it converges</br>
 * &nbsp &nbsp break; </br>
 * &nbsp fval = ...; // Compute the new objective function value at the updated W</br>
 * &nbsp if (flags[1])  // flags[1] indicates if gradient at the updated W is required</br>
 * &nbsp &nbsp G = ...; // Compute the gradient at the new W</br>
 * } </br>
 * </br>
 * </code>
 * 
 * @version 1.0 Mar. 11th, 2013
 * 
 * @author Mingjie Qian
 */
public class AcceleratedGradientDescent {
	
	private static ProximalMapping prox = new Prox();

	/**
	 * Main entry for the accelerated gradient descent algorithm. 
	 * The matrix variable to be optimized will be updated in place 
	 * to a better solution point with lower objective function value.
	 * 
	 * @param Grad_t gradient at X_t, required on the first revocation
	 * 
	 * @param fval_t objective function value at X_t: f(X_t)
	 * 
	 * @param epsilon convergence precision
	 * 
	 * @param X_t current matrix variable to be optimized, will be
	 *            updated in place to a better solution point with
	 *            lower objective function value
	 *
	 * @return a {@code boolean} array with two elements: {converge, gradientRequired}
	 * 
	 */
	public static boolean[] run(RealMatrix Grad_t, double fval_t, double epsilon, RealMatrix X_t) {
		AcceleratedProximalGradient.prox = prox;
		return AcceleratedProximalGradient.run(Grad_t, fval_t, 0, epsilon, X_t);
	}

}
