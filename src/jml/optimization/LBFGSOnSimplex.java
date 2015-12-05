package jml.optimization;

// import static jml.matlab.Matlab.eps;
import static jml.matlab.Matlab.innerProduct;
import static jml.matlab.Matlab.isnan;
import static jml.matlab.Matlab.minus;
import static jml.matlab.Matlab.norm;
import static jml.matlab.Matlab.plus;
import static jml.matlab.Matlab.rand;
import static jml.matlab.Matlab.times;
import static jml.matlab.Matlab.gt;
import static jml.matlab.Matlab.lt;
import static jml.matlab.Matlab.or;
import static jml.matlab.Matlab.not;
import static jml.matlab.Matlab.uminus;
import static jml.matlab.Matlab.subplus;
import static jml.matlab.Matlab.logicalIndexingAssignment;
import static jml.matlab.Matlab.max;
import static jml.matlab.Matlab.vertcat;
import static jml.matlab.Matlab.horzcat;
import static jml.matlab.Matlab.eye;
import static jml.matlab.Matlab.zeros;
import static jml.matlab.Matlab.ones;
import static jml.matlab.Matlab.size;
import static jml.matlab.Matlab.logicalIndexing;
import static jml.matlab.Matlab.sumAll;
import static jml.matlab.Matlab.rdivide;
import static jml.matlab.Matlab.fprintf;
import static jml.matlab.Matlab.display;

import java.io.File;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.LinkedList;

import jml.data.Data;

import org.apache.commons.math.linear.RealMatrix;

/**
 * A Java implementation for the projected limited-memory BFGS algorithm
 * with simplex constraints.
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
 * &nbsp flags = LBFGSOnSimplex.run(G, fval, epsilon, W); </br>
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
 * @version 1.0 Feb. 26th, 2013
 * 
 * @author Mingjie Qian
 */
public class LBFGSOnSimplex {
	
	/**
	 * @param args
	 */
	public static void main(String[] args) {

		int n = 10;
		RealMatrix t = rand(n);
		RealMatrix C = minus(t.multiply(t.transpose()), times(0.05, eye(n)));
		RealMatrix y = times(3, rand(n, 1));
		double epsilon = 1e-6;
		double gamma = 0.01;
		
		String path = "C:/Aaron/My Codes/Matlab/Convex Optimization";
		C = Data.loadMatrix(path + File.separator + "C.txt");
		y = Data.loadMatrix(path + File.separator + "y.txt");
		
		long start = System.currentTimeMillis();
		
		/*
		 *      min_x || C * x - y ||_2 + gamma * || x ||_2
	     * s.t. sum_i x_i = 1
         *      x_i >= 0
		 */
		RealMatrix x0 = rdivide(ones(n, 1), n);
		RealMatrix x = x0.copy();
		
		RealMatrix r_x = null;
		double f_x = 0;
		double phi_x = 0;
		double fval = 0;
		
		r_x = C.multiply(x).subtract(y);
		f_x = norm(r_x);
		phi_x = norm(x);
		fval = f_x + gamma * phi_x;
		
		RealMatrix Grad_f_x = null;
		RealMatrix Grad_phi_x = null;
		RealMatrix Grad = null;
		
		Grad_f_x = rdivide(C.transpose().multiply(r_x), f_x);
		Grad_phi_x = rdivide(x, phi_x);
		Grad = plus(Grad_f_x, times(gamma, Grad_phi_x));
		
		boolean flags[] = null;
		int k = 0;
		int maxIter = 1000;
		while (true) {
			flags = LBFGSOnSimplex.run(Grad, fval, epsilon, x);
			if (flags[0])
				break;
			// display(W);
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
			fval = f_x + gamma * phi_x;
			
			
			if (flags[1]) {
				k = k + 1;
				// Compute the gradient
				if (k > maxIter)
					break;
				
				Grad_f_x = rdivide(C.transpose().multiply(r_x), f_x);
				Grad_phi_x = rdivide(x, phi_x);
				Grad = plus(Grad_f_x, times(gamma, Grad_phi_x));
				
				/*if ( Math.abs(fval_pre - fval) < eps)
					break;
				fval_pre = fval;*/
			}
			
		}
		
		RealMatrix x_projected_LBFGS_Armijo = x;
		double f_projected_LBFGS_Armijo = fval;
		fprintf("fval_projected_LBFGS_Armijo: %g\n\n", f_projected_LBFGS_Armijo);
		fprintf("x_projected_LBFGS_Armijo:\n");
		display(x_projected_LBFGS_Armijo.transpose());
		
		double elapsedTime = (System.currentTimeMillis() - start) / 1000d;
		fprintf("Elapsed time: %.3f seconds\n", elapsedTime);
		
	}
	
	/**
	 * Current gradient.
	 */
	private static RealMatrix G = null;
	
	/**
	 * Current projected gradient.
	 *//*
	private static RealMatrix PG = null;*/
	
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
	 *//*
	private static RealMatrix p = null;*/
	
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
	
	private static RealMatrix z = null;
	
	private static RealMatrix z_t = null;
	
	private static RealMatrix p_z = null;
	
	private static RealMatrix I_z = null;
	
	private static RealMatrix G_z = null;
	
	private static RealMatrix PG_z = null;
	
	private static int i = -1;
	
	/**
	 * Tolerance of convergence.
	 */
	private static double tol = 1;
	
	/**
	 * An array holding the sequence of objective function values. 
	 */
	private static ArrayList<Double> J = new ArrayList<Double>();
	
	/**
	 * Main entry for the LBFGS algorithm. The matrix variable to
	 * be optimized will be updated in place to a better solution point with
	 * lower objective function value.
	 * 
	 * @param Grad_t gradient at original X_t
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
			
			tol = epsilon * norm(G);
			
			k = 0;
			state = 1;
			
		}
		
		if (state == 1) {
						
			RealMatrix I_k = null;
			RealMatrix I_k_com = null;
			// RealMatrix PG = null;
			
			/*I_k = G_x < 0 | x > 0;
		    I_k_com = not(I_k);
		    PG_x(I_k) = G_x(I_k);
		    PG_x(I_k_com) = 0;*/
			
			/*I_k = or(lt(G, 0), gt(X, 0));
			I_k_com = not(I_k);
			PG = G.copy();
			// logicalIndexingAssignment(PG, I_k, logicalIndexing(G, I_k));
			logicalIndexingAssignment(PG, I_k_com, 0);*/
			
			/*double norm_PGrad = norm(PG);
			if (norm_PGrad < tol) {
				converge = true;
				gradientRequired = false;
				state = 4;
				System.out.printf("PLBFGS converges with norm(PGrad) %f\n", norm_PGrad);
				return new boolean[] {converge, gradientRequired};
			}*/
		    
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
			
			/*RealMatrix HG = r;
			RealMatrix PHG = HG.copy();
			I_k = or(lt(HG, 0), gt(X, 0));
			I_k_com = not(I_k);
			logicalIndexingAssignment(PHG, I_k_com, 0);
			if (innerProduct(PHG, G) <= 0)
				p = uminus(PG);
			else
				p = uminus(PHG);*/
			
			// Transform x_k to z_k
			i = (int)max(X, 1).get("idx").getEntry(0, 0) + 1;
			int n = size(X, 1);
			RealMatrix A = vertcat(horzcat(eye(i - 1), zeros(i - 1, n - i)),
					horzcat(uminus(ones(1, i - 1)), uminus(ones(1, n - i))),
					horzcat(zeros(n - i, i - 1), eye(n - i)));
			
			RealMatrix b_z = vertcat(zeros(i - 1, 1), ones(1, 1), zeros(n - i, 1));
			
			I_z = not(b_z);
			z = logicalIndexing(X, I_z);
			
			G_z = A.transpose().multiply(G);
			I_k = or(lt(G_z, 0), gt(z, 0));
			I_k_com = not(I_k);
			PG_z = G_z.copy();
			logicalIndexingAssignment(PG_z, I_k_com, 0);
			
			double norm_PGrad_z = norm(PG_z);
			if (norm_PGrad_z < tol) {
				converge = true;
				gradientRequired = false;
				state = 4;
				System.out.printf("PLBFGS on simplex converges with norm(PGrad_z) %f\n", norm_PGrad_z);
				return new boolean[] {converge, gradientRequired};
			}
			
			RealMatrix HG_z = A.transpose().multiply(r);
			I_k = or(lt(HG_z, 0), gt(z, 0));
			I_k_com = not(I_k);
			RealMatrix PHG_z = HG_z.copy();
			logicalIndexingAssignment(PHG_z, I_k_com, 0);
			
			if (innerProduct(PHG_z, G_z) <= 0)
				p_z = uminus(PG_z);
			else
				p_z = uminus(PHG_z);
			
			t = 1;
			// z is always less than 0
			// z = innerProduct(G, p);
			// RealMatrix z_t = null;
			while (true) {
				z_t = subplus(plus(z, times(t, p_z)));
				if (sumAll(z_t) <= 1) {
					break;
				} else {
					t = beta * t;
				}
			}
			
			state = 2;
			
			// z_t = subplus(plus(z, times(t, p_z)));
			// setMatrix(X_t, X);
			logicalIndexingAssignment(X_t, I_z, z_t);
			X_t.setEntry(i - 1, 0, 1 - sumAll(z_t));
			
			// X_t.setSubMatrix(subplus(plus(X, times(t, p))).getData(), 0, 0);
			// setMatrix(X_t, subplus(plus(X, times(t, p))));
			
			converge = false;
			gradientRequired = false;
			
			return new boolean[] {converge, gradientRequired};
			
		}
		
		// Backtracking line search
		if (state == 2) {
			
			converge = false;

			if (fval_t <= fval + alpha * innerProduct(G_z, minus(z_t, z))) {
				gradientRequired = true;
				state = 3;
			} else {
				t = beta * t;
				z_t = subplus(plus(z, times(t, p_z)));
				logicalIndexingAssignment(X_t, I_z, z_t);
				X_t.setEntry(i - 1, 0, 1 - sumAll(z_t));
				gradientRequired = false;
				// setMatrix(X_t, subplus(plus(X, times(t, p))));
			}	

			return new boolean[] {converge, gradientRequired};
			
		}
		
		if (state == 3) {
			
			X_pre = X.copy();   
		    G_pre = G.copy();
		    
		    /*if (Math.abs(fval_t - fval) < eps) {
				converge = true;
				gradientRequired = false;
				System.out.printf("Objective function value doesn't decrease, iteration stopped!\n");
				System.out.format("Iter %d, ofv: %g, norm(PGrad_z): %g\n", k + 1, fval, norm(PG_z));
				return new boolean[] {converge, gradientRequired};
		    }*/
	        
		    fval = fval_t;
		    J.add(fval);
		    System.out.format("Iter %d, ofv: %g, norm(PGrad_z): %g\n", k + 1, fval, norm(PG_z));
		    
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
