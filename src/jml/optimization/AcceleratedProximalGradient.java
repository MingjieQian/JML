package jml.optimization;

import static jml.matlab.Matlab.*;

import java.util.ArrayList;

import org.apache.commons.math.linear.RealMatrix;

/**
 * A Java implementation for the accelerated proximal gradient method.
 * We want to optimize the following optimization problem:</br>
 * <p>min_X g(X) + t * h(X).</p>
 * 
 * It is a general algorithm interface, only gradient and objective
 * function value are needed to compute outside the class.
 * </p>
 * A simple example: </br></br>
 * <code>
 * AcceleratedProximalGradient.prox = new ProxPlus(); // Set the proximal mapping function</br>
 * double epsilon = ...; // Convergence tolerance</br>
 * Matrix X = ...; // Initial matrix (vector) you want to optimize</br>
 * Matrix G = ...; // Gradient of g at the initial matrix (vector) you want to optimize</br>
 * double gval = ...; // Initial objective function value for g(X)</br>
 * double hval = ...; // Initial objective function value for t * h(X)</br>
 * </br>
 * boolean flags[] = null; </br>
 * while (true) { </br>
 * &nbsp flags = AcceleratedProximalGradient.run(G, gval, hval, epsilon, X); // Update X in place</br>
 * &nbsp if (flags[0]) // flags[0] indicates if it converges</br>
 * &nbsp &nbsp break; </br>
 * &nbsp gval = ...; // Compute the new objective function value for g(X) at the updated X</br>
 * &nbsp hval = ...; // Compute the new objective function value for t * h(X) at the updated X</br>
 * &nbsp if (flags[1])  // flags[1] indicates if gradient at the updated X is required</br>
 * &nbsp &nbsp G = ...; // Compute the gradient at the new X</br>
 * } </br>
 * </br>
 * </code>
 * 
 * @version 1.0 Mar. 11th, 2013
 * 
 * @author Mingjie Qian
 */
public class AcceleratedProximalGradient {
	
	/**
	 * @param args
	 */
	public static void main(String[] args) {
		
		int n = 10;
		RealMatrix t = rand(n);
		RealMatrix C = minus(t.multiply(t.transpose()), times(0.1, eye(n)));
		RealMatrix y = times(3, minus(0.5, rand(n, 1)));
		double epsilon = 1e-4;
		double gamma = 0.01;
		
		AcceleratedProximalGradient.prox = new ProxPlus();
		
		long start = System.currentTimeMillis();
		
		/*
		 *      min_x || C * x - y ||_2 + gamma * || x ||_2
	     * s.t. x >= 0
	     * 
	     * g(x) = || C * x - y ||_2 + gamma * || x ||_2
	     * h(x) = I_+(x)
		 */
		RealMatrix x0 = rdivide(ones(n, 1), n);
		RealMatrix x = x0.copy();
		
		RealMatrix r_x = null;
		double f_x = 0;
		double phi_x = 0;
		double gval = 0;
		double hval = 0;
		double fval = 0;
		
		r_x = C.multiply(x).subtract(y);
		f_x = norm(r_x);
		phi_x = norm(x);
		gval = f_x + gamma * phi_x;
		hval = 0;
		fval = gval + hval;
		
		RealMatrix Grad_f_x = null;
		RealMatrix Grad_phi_x = null;
		RealMatrix Grad = null;
		
		Grad_f_x = rdivide(C.transpose().multiply(r_x), f_x);
		Grad_phi_x = rdivide(x, phi_x);
		Grad = plus(Grad_f_x, times(gamma, Grad_phi_x));
		
		boolean flags[] = null;
		int k = 0;
		int maxIter = 10000;
		hval = 0;
		while (true) {
			
			// flags = AcceleratedProximalGradient.run(Grad, gval, hval, epsilon, x);
			flags = AcceleratedGradientDescent.run(Grad, gval, epsilon, x);
			
			if (flags[0])
				break;
			
			if (sumAll(isnan(x)) > 0) {
				int a = 1;
				a = a + 1;
			}
			
			/*
			 *  Compute the objective function value, if flags[1] is true
			 *  gradient will also be computed.
			 */
			r_x = C.multiply(x).subtract(y);
			f_x = norm(r_x);
			phi_x = norm(x);
			gval = f_x + gamma * phi_x;
			hval = 0;
			fval = gval + hval;
			
			if (flags[1]) {
				
				k = k + 1;
				
				// Compute the gradient
				if (k > maxIter)
					break;
				
				Grad_f_x = rdivide(C.transpose().multiply(r_x), f_x);
				if (phi_x != 0)
					Grad_phi_x = rdivide(x, phi_x);
				else
					Grad_phi_x = times(0, Grad_phi_x);
				Grad = plus(Grad_f_x, times(gamma, Grad_phi_x));
				
				/*if ( Math.abs(fval_pre - fval) < eps)
					break;
				fval_pre = fval;*/
				
			}
			
		}
		
		RealMatrix x_accelerated_proximal_gradient = x;
		double f_accelerated_proximal_gradient = fval;
		fprintf("fval_accelerated_proximal_gradient: %g\n\n", f_accelerated_proximal_gradient);
		fprintf("x_accelerated_proximal_gradient:\n");
		display(x_accelerated_proximal_gradient.transpose());
		
		double elapsedTime = (System.currentTimeMillis() - start) / 1000d;
		fprintf("Elapsed time: %.3f seconds\n", elapsedTime);

	}
	
	/**
	 * Proximity operator: prox_th(X).
	 */
	public static ProximalMapping prox = null;
	
	/**
	 * Current gradient.
	 */
	private static RealMatrix Grad_Y_k = null;
	
	/**
	 * Current matrix variable that we want to optimize.
	 */
	private static RealMatrix X = null;
	
	/**
	 * Last matrix variable that we want to optimize.
	 */
	private static RealMatrix X_pre = null;
	
	/**
	 * Current matrix variable that we want to optimize.
	 */
	private static RealMatrix Y = null;
	
	/**
	 * X_{k + 1} = prox_th(Y_k - t * Grad_Y_k) = y_k - t * G_Y_k.
	 */
	private static RealMatrix G_Y_k = null;
	
	/**
	 * g(Y_k).
	 */
	private static double gval_Y_k = 0;
	
	/**
	 * h(Y_k).
	 */
	private static double hval_Y_k = 0;
	
	/**
	 * f(Y_k) = g(Y_k) + h(Y_k).
	 */
	private static double fval_Y_k = 0;
	
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
	
	private static double beta = 0.95;
	
	/**
	 * Iteration counter.
	 */
	private static int k = 1;
	
	/**
	 * An array holding the sequence of objective function values. 
	 */
	private static ArrayList<Double> J = new ArrayList<Double>();
	
	/**
	 * Main entry for the accelerated proximal gradient algorithm. 
	 * The matrix variable to be optimized will be updated in place 
	 * to a better solution point with lower objective function value.
	 * 
	 * @param Grad_t gradient at X_t, required on the first revocation
	 * 
	 * @param gval_t g(X_t)
	 * 
	 * @param hval_t h(X_t)
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
	public static boolean[] run(RealMatrix Grad_t, double gval_t, double hval_t, double epsilon, RealMatrix X_t) {
		
		// If the algorithm has converged, we do a new job
		if (state == 4) {
			J.clear();
			state = 0;
		}
		
		if (state == 0) {

			X = X_t.copy();
			Y = X_t.copy();
			
			gval_Y_k = gval_t;
			hval_Y_k = hval_t;
			fval_Y_k = gval_Y_k + hval_Y_k;
			if (Double.isNaN(fval_Y_k)) {
				System.err.println("Object function value is nan!");
				System.exit(1);
			}
			System.out.format("Initial ofv: %g\n", fval_Y_k);

			k = 1;
			t = 1;
			state = 1;

		}
		
		if (state == 1) {
			
			if (Grad_t == null) {
				System.err.println("Gradient is required!");
				System.exit(1);
			}
			Grad_Y_k = Grad_t.copy();
			
			gval_Y_k = gval_t;
			hval_Y_k = hval_t;
			
			
			state = 2;
			
			// X_t.setSubMatrix(plus(X, times(t, p)).getData(), 0, 0);
			setMatrix(X_t, prox.compute(t, minus(Y, times(t, Grad_Y_k))));
			
			G_Y_k = rdivide(minus(Y, X_t), t);
			
			converge = false;
			gradientRequired = false;
			
			return new boolean[] {converge, gradientRequired};
			
		}
		
		// Backtracking line search
		if (state == 2) {

			converge = false;

			if (gval_t <= gval_Y_k - t * innerProduct(Grad_Y_k, G_Y_k) + t / 2 * innerProduct(G_Y_k, G_Y_k) + eps) {
				gradientRequired = true;
				state = 3;
			} else {
				t = beta * t;
				gradientRequired = false;
				setMatrix(X_t, prox.compute(t, minus(Y, times(t, Grad_Y_k))));
				G_Y_k = rdivide(minus(Y, X_t), t);
				return new boolean[] {converge, gradientRequired};
			}

		}
		
		if (state == 3) {
			
			double norm_G_Y = norm(G_Y_k);
			
			if (norm_G_Y < epsilon) {
				converge = true;
				gradientRequired = false;
				state = 4;
				System.out.printf("Accelerated proximal gradient method " +
						"converges with norm(G_Y_k) %f\n", norm_G_Y);
				return new boolean[] {converge, gradientRequired};
			}
			
			fval_Y_k = gval_Y_k + hval_Y_k;
		    J.add(fval_Y_k);
		    System.out.format("Iter %d, ofv: %g, norm(G_Y_k): %g\n", k, fval_Y_k, norm(G_Y_k));
			
			X_pre = X.copy();
			X = X_t.copy();
			Y = plus(X, times((double)(k) / (k + 3), minus(X, X_pre)));
			setMatrix(X_t, Y);
			
			k = k + 1;
		    
		    state = 1;
			
		}
		
		converge = false;
	    gradientRequired = true;
	    return new boolean[] {converge, gradientRequired};
		
	}
	
}

