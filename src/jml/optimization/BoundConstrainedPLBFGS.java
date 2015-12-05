package jml.optimization;

import static jml.matlab.Matlab.eps;
import static jml.matlab.Matlab.inf;
import static jml.matlab.Matlab.innerProduct;
import static jml.matlab.Matlab.minus;
import static jml.matlab.Matlab.norm;
import static jml.matlab.Matlab.plus;
import static jml.matlab.Matlab.times;
import static jml.matlab.Matlab.gt;
import static jml.matlab.Matlab.lt;
import static jml.matlab.Matlab.and;
import static jml.matlab.Matlab.eq;
import static jml.matlab.Matlab.uminus;
import static jml.matlab.Matlab.min;
import static jml.matlab.Matlab.max;
import static jml.matlab.Matlab.setMatrix;
import static jml.matlab.Matlab.logicalIndexing;
import static jml.matlab.Matlab.logicalIndexingAssignment;

import java.util.ArrayList;
import java.util.Iterator;
import java.util.LinkedList;

import org.apache.commons.math.linear.RealMatrix;

/**
 * A Java implementation for the projected limited-memory BFGS algorithm
 * with bound constraints.
 * 
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
 * &nbsp flags = LBFGS.run(G, fval, epsilon, W); </br>
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
 * @version 1.0 Jan. 11th, 2013
 * 
 * @author Mingjie Qian
 */
public class BoundConstrainedPLBFGS {
	
	/**
	 * Current gradient.
	 */
	private static RealMatrix G = null;
	
	/**
	 * Current projected gradient.
	 */
	private static RealMatrix PG = null;
	
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
	 * Lower bound.
	 *//*
	private static double l = 0;
	
	*//**
	 * Upper bound.
	 *//*
	private static double u = 1;
	
	*//**
	 * Lower bound matrix.
	 *//*
	private static RealMatrix L = null;
	
	*//**
	 * Upper bound matrix.
	 *//*
	private static RealMatrix U = null;*/
	
	/**
	 * A temporary variable holding the inner product of the decreasing step p
	 * and the gradient G, it should be always non-positive.
	 *//*
	private static double z = 0;*/
	
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
	 * Tolerance of convergence.
	 */
	private static double tol = 1;
	
	/**
	 * An array holding the sequence of objective function values. 
	 */
	private static ArrayList<Double> J = new ArrayList<Double>();
	
	/**
	 * Main entry for the LBFGS algorithm with bound constraints. The matrix 
	 * variable to be optimized will be updated in place to a better solution 
	 * point with lower objective function value.
	 * 
	 * @param Grad_t gradient at original X_t
	 * 
	 * @param fval_t objective function value on original X_t
	 * 
	 * @param l lower bound
	 * 
	 * @param u upper bound
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
	public static boolean[] run(RealMatrix Grad_t, double fval_t, double l, double u, double epsilon, RealMatrix X_t) {
		
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
			
			/*BoundConstrainedPLBFGS.l = l;
			BoundConstrainedPLBFGS.u = u;*/
			tol = epsilon * norm(G, inf);
			
			k = 0;
			state = 1;
			
		}
		
		if (state == 1) {
						
			RealMatrix I_b = null;
			RealMatrix I_l = null;
			RealMatrix I_u = null;
			// RealMatrix PG = null;
			
			/*I_b = l < x & x < u;
		    I_l = x == l;
		    I_u = x == u;
		    
			PG_x(I_b) = G_x(I_b);
		    PG_x(I_l) = min(G_x(I_l), 0);
		    PG_x(I_u) = max(G_x(I_u), 0);*/
			
			I_b = and(lt(l, X), lt(X, u));
			I_l = eq(X, l);
			I_u = eq(X, u);
			PG = G.copy();
			logicalIndexingAssignment(PG, I_b, logicalIndexing(G, I_b));
			logicalIndexingAssignment(PG, I_l, min(logicalIndexing(G, I_l), 0.0));
			logicalIndexingAssignment(PG, I_u, max(logicalIndexing(G, I_u), 0.0));
			
			// System.out.format("norm(PG): %g\n", norm(PG, inf));
			double norm_PGrad = norm(PG, inf);
			if (norm_PGrad < tol) {
				converge = true;
				gradientRequired = false;
				state = 4;
				System.out.printf("PLBFGS converges with norm(PGrad) %f\n", norm_PGrad);
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
			// p = uminus(r);
			
			/*HG_x = r;
		    I_k = HG_x < 0 | x > 0;
		    I_k_com = not(I_k);
		    PHG_x(I_k) = HG_x(I_k);
		    PHG_x(I_k_com) = 0;
		    
		    if (PHG_x' * G_x <= 0)
		        p = -PG_x;
		    else
		        p = -PHG_x;
		    end*/
			
			RealMatrix HG = r;
			RealMatrix PHG = HG.copy();
			
			/*PHG_x(I_b) = HG_x(I_b);
		    PHG_x(I_l) = min(HG_x(I_l), 0);
		    PHG_x(I_u) = max(HG_x(I_u), 0);*/
			
			logicalIndexingAssignment(PHG, I_b, logicalIndexing(HG, I_b));
			logicalIndexingAssignment(PHG, I_l, min(logicalIndexing(HG, I_l), 0.0));
			logicalIndexingAssignment(PHG, I_u, max(logicalIndexing(HG, I_u), 0.0));
			
			if (innerProduct(PHG, G) <= 0)
				p = uminus(PG);
			else
				p = uminus(PHG);
			
			t = 1;
			// z is always less than 0
			// z = innerProduct(G, p);
			
			state = 2;
			
			// X_t.setSubMatrix(subplus(plus(X, times(t, p))).getData(), 0, 0);
			setMatrix(X_t, project(plus(X, times(t, p)), l, u));
			
			converge = false;
			gradientRequired = false;
			
			return new boolean[] {converge, gradientRequired};
			
		}
		
		// Backtracking line search
		if (state == 2) {
			
			converge = false;

			if (fval_t <= fval + alpha * t * innerProduct(G, minus(X_t, X))) {
				gradientRequired = true;
				state = 3;
			} else {
				t = beta * t;
				gradientRequired = false;
				setMatrix(X_t, project(plus(X, times(t, p)), l, u));
			}	

			return new boolean[] {converge, gradientRequired};
			
		}
		
		if (state == 3) {
			
			X_pre = X.copy();   
		    G_pre = G.copy();
		    
		    if (Math.abs(fval_t - fval) < eps) {
				converge = true;
				gradientRequired = false;
				System.out.printf("Objective function value doesn't decrease, iteration stopped!\n");
				System.out.format("Iter %d, ofv: %g, norm(PGrad): %g\n", k + 1, fval, norm(PG, inf));
				return new boolean[] {converge, gradientRequired};
		    }
	        
		    fval = fval_t;
		    J.add(fval);
		    System.out.format("Iter %d, ofv: %g, norm(PGrad): %g\n", k + 1, fval, norm(PG, inf));
		    
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
	
	/**
	 * Main entry for the LBFGS algorithm with bound constraints. The matrix 
	 * variable to be optimized will be updated in place to a better solution 
	 * point with lower objective function value.
	 * 
	 * @param Grad_t gradient at original X_t
	 * 
	 * @param fval_t objective function value on original X_t
	 * 
	 * @param L lower bound matrix
	 * 
	 * @param U upper bound matrix
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
	public static boolean[] run(RealMatrix Grad_t, double fval_t, RealMatrix L, RealMatrix U, double epsilon, RealMatrix X_t) {
		
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
			
			/*BoundConstrainedPLBFGS.L = L.copy();
			BoundConstrainedPLBFGS.U = U.copy();*/
			tol = epsilon * norm(G, inf);
			
			state = 1;
			
		}
		
		if (state == 1) {
						
			RealMatrix I_b = null;
			RealMatrix I_l = null;
			RealMatrix I_u = null;
			// RealMatrix PG = null;
			
			/*I_b = l < x & x < u;
		    I_l = x == l;
		    I_u = x == u;
		    
			PG_x(I_b) = G_x(I_b);
		    PG_x(I_l) = min(G_x(I_l), 0);
		    PG_x(I_u) = max(G_x(I_u), 0);*/
			
			I_b = and(lt(L, X), lt(X, U));
			I_l = eq(X, L);
			I_u = eq(X, U);
			PG = G.copy();
			logicalIndexingAssignment(PG, I_b, logicalIndexing(G, I_b));
			logicalIndexingAssignment(PG, I_l, min(logicalIndexing(G, I_l), 0.0));
			logicalIndexingAssignment(PG, I_u, max(logicalIndexing(G, I_u), 0.0));
			
			// System.out.format("norm(PG): %g\n", norm(PG));
			double norm_PGrad = norm(PG, inf);
			if (norm_PGrad < tol) {
				converge = true;
				gradientRequired = false;
				state = 4;
				System.out.printf("PLBFGS converges with norm(PGrad) %f\n", norm_PGrad);
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
			// p = uminus(r);
			
			/*HG_x = r;
		    I_k = HG_x < 0 | x > 0;
		    I_k_com = not(I_k);
		    PHG_x(I_k) = HG_x(I_k);
		    PHG_x(I_k_com) = 0;
		    
		    if (PHG_x' * G_x <= 0)
		        p = -PG_x;
		    else
		        p = -PHG_x;
		    end*/
			
			RealMatrix HG = r;
			RealMatrix PHG = HG.copy();
			
			/*PHG_x(I_b) = HG_x(I_b);
		    PHG_x(I_l) = min(HG_x(I_l), 0);
		    PHG_x(I_u) = max(HG_x(I_u), 0);*/
			
			logicalIndexingAssignment(PHG, I_b, logicalIndexing(HG, I_b));
			logicalIndexingAssignment(PHG, I_l, min(logicalIndexing(HG, I_l), 0.0));
			logicalIndexingAssignment(PHG, I_u, max(logicalIndexing(HG, I_u), 0.0));
			
			if (innerProduct(PHG, G) <= 0)
				p = uminus(PG);
			else
				p = uminus(PHG);
			
			t = 1;
			// z is always less than 0
			// z = innerProduct(G, p);
			
			state = 2;
			
			// X_t.setSubMatrix(subplus(plus(X, times(t, p))).getData(), 0, 0);
			setMatrix(X_t, project(plus(X, times(t, p)), L, U));
			
			converge = false;
			gradientRequired = false;
			
			return new boolean[] {converge, gradientRequired};
			
		}
		
		// Backtracking line search
		if (state == 2) {
			
			converge = false;

			if (fval_t <= fval + alpha * t * innerProduct(G, minus(X_t, X))) {
				gradientRequired = true;
				state = 3;
			} else {
				t = beta * t;
				gradientRequired = false;
				setMatrix(X_t, project(plus(X, times(t, p)), L, U));
			}	

			return new boolean[] {converge, gradientRequired};
			
		}
		
		if (state == 3) {
			
			X_pre = X.copy();   
		    G_pre = G.copy();
		    
		    if (Math.abs(fval_t - fval) < eps) {
				converge = true;
				gradientRequired = false;
				System.out.printf("Objective function value doesn't decrease, iteration stopped!\n");
				System.out.format("Iter %d, ofv: %g, norm(PGrad): %g\n", k + 1, fval, norm(PG, inf));
				return new boolean[] {converge, gradientRequired};
		    }
	        
		    fval = fval_t;
		    J.add(fval);
		    System.out.format("Iter %d, ofv: %g, norm(PGrad): %g\n", k + 1, fval, norm(PG, inf));
		    
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
	
	/**
	 * Projection on a box region [l, u].
	 * 
	 * @param A a matrix to be projected
	 * 
	 * @param l lower bound
	 * 
	 * @param u upper bound
	 * 
	 * @return a matrix projected on a box region [l, u] 
	 * 
	 */
	private static RealMatrix project(RealMatrix A, double l, double u) {
		logicalIndexingAssignment(A, lt(A, l), l);
		logicalIndexingAssignment(A, gt(A, u), u);
		return A;
	}
	
	/**
	 * Projection on a box region [L_{ij}, U_{ij}] for each element.
	 * 
	 * @param A a matrix to be projected
	 * 
	 * @param L lower bound matrix
	 * 
	 * @param U upper bound matrix
	 * 
	 * @return a matrix projected on a box region [L_{ij}, U_{ij}] 
	 *         for each element 
	 * 
	 */
	private static RealMatrix project(RealMatrix A, RealMatrix L, RealMatrix U) {
		RealMatrix I_l = lt(A, L);
		RealMatrix I_u = gt(A, U);
		logicalIndexingAssignment(A, I_l, logicalIndexing(L, I_l));
		logicalIndexingAssignment(A, I_u, logicalIndexing(U, I_u));
		return A;
	}

}
