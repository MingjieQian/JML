package jml.optimization;

// import static jml.matlab.Matlab.eps;
import static jml.matlab.Matlab.inf;
import static jml.matlab.Matlab.innerProduct;
import static jml.matlab.Matlab.minus;
import static jml.matlab.Matlab.norm;
import static jml.matlab.Matlab.plus;
import static jml.matlab.Matlab.setMatrix;
import static jml.matlab.Matlab.times;
import static jml.matlab.Matlab.uminus;

import java.util.ArrayList;
import java.util.Iterator;
import java.util.LinkedList;

import org.apache.commons.math.linear.RealMatrix;

/**
 * A Java implementation for the limited-memory BFGS algorithm.
 * It is a general algorithm interface, only gradient and objective
 * function value are needed to compute outside the class.
 * </p>
 * A simple example: </br></br>
 * <code>
 * double epsilon = ...; // Convergence tolerance</br>
 * RealMatrix W = ...; // Initial matrix (vector) you want to optimize</br>
 * RealMatrix G = ...; // Gradient at the initial matrix (vector) you want to optimize</br>
 * double fval = ...; // Initial objective function value</br>
 * </br>
 * boolean flags[] = null; </br>
 * while (true) { </br>
 * &nbsp flags = LBFGS.run(G, fval, epsilon, W); // Update W in place</br>
 * &nbsp if (flags[0]) // flags[0] indicates if L-BFGS converges</br>
 * &nbsp &nbsp break; </br>
 * &nbsp fval = ...; // Compute the new objective function value at the updated W</br>
 * &nbsp if (flags[1])  // flags[1] indicates if gradient at the updated W is required</br>
 * &nbsp &nbsp G = ...; // Compute the gradient at the new W</br>
 * } </br>
 * </br>
 * </code>
 * 
 * @version 1.0 Jan. 11th, 2013
 * 
 * @author Mingjie Qian
 */
public class LBFGS {
	
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
	 * Last matrix variable that we want to optimize.
	 */
	private static RealMatrix X_pre = null;
	
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
	
	private static double alpha = 0.2;
	
	private static double beta = 0.75;
	
	private static int m = 30;
	
	private static double H = 0;
		
	private static RealMatrix s_k = null;
	private static RealMatrix y_k = null;
	private static double rou_k;
	
	private static LinkedList<RealMatrix> s_ks = new LinkedList<RealMatrix>();
	private static LinkedList<RealMatrix> y_ks = new LinkedList<RealMatrix>();
	private static LinkedList<Double> rou_ks = new LinkedList<Double>();
	
	/**
	 * An array holding the sequence of objective function values. 
	 */
	private static ArrayList<Double> J = new ArrayList<Double>();
	
	/**
	 * Main entry for the LBFGS algorithm. The matrix variable to be 
	 * optimized will be updated in place to a better solution point 
	 * with lower objective function value.
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
	 *            lower objective function value
	 * 
	 * @return a {@code boolean} array with two elements: {converge, gradientRequired}
	 * 
	 */
	public static boolean[] run(RealMatrix Grad_t, double fval_t, double epsilon, RealMatrix X_t) {
		
		// If the algorithm has converged, we do a new job
		if (state == 4) {
			s_ks.clear();
			y_ks.clear();
			rou_ks.clear();
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
			
			k = 0;
			state = 1;
			
		}
		
		if (state == 1) {
			
			double norm_Grad = norm(G, inf);
			if (norm_Grad < epsilon) {
				converge = true;
				gradientRequired = false;
				state = 4;
				System.out.printf("L-BFGS converges with norm(Grad) %f\n", norm_Grad);
				return new boolean[] {converge, gradientRequired};
			}
			
			if (k == 0) {
				H = 1;
			} else {
				H = innerProduct(s_k, y_k) / innerProduct(y_k, y_k);
			}		
			
			RealMatrix s_k_i = null;
			RealMatrix y_k_i = null;
			Double rou_k_i = null;
			
			Iterator<RealMatrix> iter_s_ks = null;
			Iterator<RealMatrix> iter_y_ks = null;
			Iterator<Double> iter_rou_ks = null;
			
			double[] a = new double[m];
			double b = 0;
			
			RealMatrix q = null;
			RealMatrix r = null;
			
			q = G;
			iter_s_ks = s_ks.descendingIterator();
			iter_y_ks = y_ks.descendingIterator();
			iter_rou_ks = rou_ks.descendingIterator();
			for (int i = s_ks.size() - 1; i >= 0; i--) {
				s_k_i = iter_s_ks.next();
				y_k_i = iter_y_ks.next();
				rou_k_i = iter_rou_ks.next();
				a[i] = rou_k_i * innerProduct(s_k_i, q);
				q = q.subtract(times(a[i], y_k_i));
			}
			r = times(H, q);
			iter_s_ks = s_ks.iterator();
			iter_y_ks = y_ks.iterator();
			iter_rou_ks = rou_ks.iterator();
			for (int i = 0; i < s_ks.size(); i++) {
				s_k_i = iter_s_ks.next();
				y_k_i = iter_y_ks.next();
				rou_k_i = iter_rou_ks.next();
				b = rou_k_i * innerProduct(y_k_i, r);
				r = r.add(times(a[i] - b, s_k_i));
			}
			// p is a decreasing step
			p = uminus(r);
			
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
				t = beta * t;
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
			
			X_pre = X.copy();
		    G_pre = G.copy();
		    
		    if (Math.abs(fval_t - fval) < 1e-32) {
				converge = true;
				gradientRequired = false;
				System.out.printf("Objective function value doesn't decrease, iteration stopped!\n");
				System.out.format("Iter %d, ofv: %g, norm(Grad): %g\n", k + 1, fval, norm(G));
				return new boolean[] {converge, gradientRequired};
		    }
	        
		    fval = fval_t;
		    J.add(fval);
		    System.out.format("Iter %d, ofv: %g, norm(Grad): %g\n", k + 1, fval, norm(G));
		    
		    X = X_t.copy();
		    G = Grad_t.copy();
		    
		    s_k = X.subtract(X_pre);
		    y_k = minus(G, G_pre);
		    rou_k = 1 / innerProduct(y_k, s_k);
		    
		    // Now s_ks, y_ks, and rou_ks all have k elements
		    if (k >= m) {
		    	s_ks.removeFirst();
		    	y_ks.removeFirst();
		    	rou_ks.removeFirst();
		    }
		    s_ks.add(s_k);
	    	y_ks.add(y_k);
	    	rou_ks.add(rou_k);
		    
		    k = k + 1;
		    
		    state = 1;
		    
		}
		
		converge = false;
	    gradientRequired = false;
	    return new boolean[] {converge, gradientRequired};
		
	}

}
