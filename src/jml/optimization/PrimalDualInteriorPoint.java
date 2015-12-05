package jml.optimization;

import static jml.matlab.Matlab.diag;
import static jml.matlab.Matlab.fprintf;
import static jml.matlab.Matlab.gt;
import static jml.matlab.Matlab.horzcat;
import static jml.matlab.Matlab.innerProduct;
import static jml.matlab.Matlab.lt;
import static jml.matlab.Matlab.mldivide;
import static jml.matlab.Matlab.mtimes;
import static jml.matlab.Matlab.norm;
import static jml.matlab.Matlab.ones;
import static jml.matlab.Matlab.plus;
import static jml.matlab.Matlab.rdivide;
import static jml.matlab.Matlab.setMatrix;
import static jml.matlab.Matlab.sumAll;
import static jml.matlab.Matlab.times;
import static jml.matlab.Matlab.uminus;
import static jml.matlab.Matlab.vertcat;
import static jml.matlab.Matlab.zeros;

import static jml.utils.Time.*;

import java.util.ArrayList;

import org.apache.commons.math.linear.RealMatrix;

/***
 * A Java implementation for the Primal-dual Interior-point
 * method. While general nonlinear objective functions and 
 * nonlinear inequality constraints are supported, only 
 * linear equality constraints are allowed.
 * </p>
 * Example:
 * </p>
 * <code>
 * fval = innerProduct(x, Q.multiply(x)) / 2 + innerProduct(c, x); </br>
 * H_x = Q; </br>
 * DF_x = B; </br>
 * F_x = B.multiply(x).subtract(d); </br>
 * G_f_x = Q.multiply(x).add(c); </br>
 * 
 * boolean flags[] = null; </br>
 * int k = 0; </br>
 * while (true) { </br>
 * &nbsp flags = PrimalDualInteriorPoint.run(A, b, H_x, F_x, DF_x, G_f_x, fval, x, l, v); </br>
 * &nbsp if (flags[0]) </br>
 * &nbsp &nbsp break; </br>
 * &nbsp // Compute the objective function value, if flags[1] is true </br>
 * &nbsp // gradient will also be computed. </br>
 * &nbsp fval = innerProduct(x, Q.multiply(x)) / 2 + innerProduct(c, x); </br>
 * &nbsp F_x = B.multiply(x).subtract(d); </br>
 * &nbsp if (flags[1]) { </br>
 * &nbsp &nbsp k = k + 1; </br>
 * &nbsp &nbsp // Compute the gradient </br>
 * &nbsp &nbsp G_f_x = Q.multiply(x).add(c); </br>
 * &nbsp } </br>
 * } </br>
 * </code>
 * 
 * @author Mingjie Qian
 * @version Feb. 28th, 2013
 */
public class PrimalDualInteriorPoint {

	/**
	 * @param args
	 */
	public static void main(String[] args) {

	}
	
	/**
	 * Unknown variable vector.
	 */
	private static RealMatrix x = null;
	
	/**
	 * Lambda, i.e., the Lagrangian multipliers for inequality 
	 * constraints.
	 */
	private static RealMatrix l = null;
	
	/**
	 * Nu, i.e., the Lagrangian multipliers for equality 
	 * constraints.
	 */
	private static RealMatrix v = null;
	
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
	 * 1: Before ensuring f_i(x) <= 0, i = 1, 2, ..., m
	 * 2: Ensuring f_i(x) <= 0, i = 1, 2, ..., m
	 * 3: Backtracking line search
	 * 4: After backtracking line search
	 * 5: Convergence
	 */
	private static int state = 0;
	
	private static double t = 1;
	
	/**
	 * Iteration counter.
	 */
	private static int k = 0;
	
	/**
	 * Number of unknown variables
	 */
	private static int n = 0;
	
	/**
	 * Number of equality constraints
	 */
	private static int p = 0;
	
	/**
	 * Number of inequality constraints
	 */
	private static int m = 0;
	
	private static double mu = 1.8;
	private static double epsilon = 1e-10;
	private static double epsilon_feas = 1e-10;
	private static double alpha = 0.1;
	private static double beta = 0.98;
	private static double eta_t = 1;
	
	/**
	 * Step length for backtracking line search.
	 */
	private static double s = 1;
	
	private static double residual = 0;
	
	private static RealMatrix r_prim = null;
	private static RealMatrix r_dual = null;
	private static RealMatrix r_cent = null;
	private static RealMatrix Matrix = null;
	private static RealMatrix Vector = null;
	
	private static RealMatrix z_pd = null;
	private static RealMatrix x_nt = null;
	private static RealMatrix l_nt = null;
	private static RealMatrix v_nt = null;
	
	/**
	 * Get Lambda.
	 * 
	 * @return Lambda
	 * 
	 */
	public static RealMatrix getOptimalLambda() {
		return l;
	}
	
	/**
	 * Get Nu.
	 * 
	 * @return Nu
	 * 
	 */
	public static RealMatrix getOptimalNu() {
		return v;
	}
	
	/**
	 * An array holding the sequence of objective function values. 
	 */
	private static ArrayList<Double> J = new ArrayList<Double>();
	
	/**
	 * Run this optimization algorithm.
	 * </p>
	 * Example:
	 * </p>
	 * <code>
	 * fval = innerProduct(x, Q.multiply(x)) / 2 + innerProduct(c, x); </br>
	 * H_x = Q; </br>
	 * DF_x = B; </br>
	 * F_x = B.multiply(x).subtract(d); </br>
	 * G_f_x = Q.multiply(x).add(c); </br>
	 * 
	 * boolean flags[] = null; </br>
	 * int k = 0; </br>
	 * while (true) { </br>
	 * &nbsp flags = PrimalDualInteriorPoint.run(A, b, H_x, F_x, DF_x, G_f_x, fval, x, l, v); </br>
	 * &nbsp if (flags[0]) </br>
	 * &nbsp &nbsp break; </br>
	 * &nbsp // Compute the objective function value, if flags[1] is true </br>
	 * &nbsp // gradient will also be computed. </br>
	 * &nbsp fval = innerProduct(x, Q.multiply(x)) / 2 + innerProduct(c, x); </br>
	 * &nbsp F_x = B.multiply(x).subtract(d); </br>
	 * &nbsp if (flags[1]) { </br>
	 * &nbsp &nbsp k = k + 1; </br>
	 * &nbsp &nbsp // Compute the gradient </br>
	 * &nbsp &nbsp G_f_x = Q.multiply(x).add(c); </br>
	 * &nbsp } </br>
	 * } </br>
	 * </code>
	 * 
	 * @param A matrix for equality constraints
	 * 
	 * @param b the constant vector for equality constraints
	 *  
	 * @param H_x Hessian for the objective function and nonlinear functions for
	 *            inequality constraints
	 *            
	 * @param F_x a vector of nonlinear functions for inequality constraints, 
	 *            F_x(i) is the i-th nonlinear function value for the i-th 
	 *            inequality constraint
	 *            
	 * @param DF_x Derivative for F_x
	 * 
	 * @param G_f_x Gradient of the objective function at the current point x
	 * 
	 * @param fval objective function value at the current point x
	 * 
	 * @param x current point, will be set in place during the procedure
	 * 
	 * @return a {@code boolean} array of two elements: {converge, gradientRequired}
	 * 
	 */
	public static boolean[] run(RealMatrix A, RealMatrix b, RealMatrix H_x, RealMatrix F_x, RealMatrix DF_x, RealMatrix G_f_x, double fval, RealMatrix x) {
		return run(A, b, H_x, F_x, DF_x, G_f_x, fval, x, null, null);
	}
	
	/**
	 * Run this optimization algorithm.
     * </p>
	 * Example:
	 * </p>
	 * <code>
	 * fval = innerProduct(x, Q.multiply(x)) / 2 + innerProduct(c, x); </br>
	 * H_x = Q; </br>
	 * DF_x = B; </br>
	 * F_x = B.multiply(x).subtract(d); </br>
	 * G_f_x = Q.multiply(x).add(c); </br>
	 * 
	 * boolean flags[] = null; </br>
	 * int k = 0; </br>
	 * while (true) { </br>
	 * &nbsp flags = PrimalDualInteriorPoint.run(A, b, H_x, F_x, DF_x, G_f_x, fval, x, l, v); </br>
	 * &nbsp if (flags[0]) </br>
	 * &nbsp &nbsp break; </br>
	 * &nbsp // Compute the objective function value, if flags[1] is true </br>
	 * &nbsp // gradient will also be computed. </br>
	 * &nbsp fval = innerProduct(x, Q.multiply(x)) / 2 + innerProduct(c, x); </br>
	 * &nbsp F_x = B.multiply(x).subtract(d); </br>
	 * &nbsp if (flags[1]) { </br>
	 * &nbsp &nbsp k = k + 1; </br>
	 * &nbsp &nbsp // Compute the gradient </br>
	 * &nbsp &nbsp G_f_x = Q.multiply(x).add(c); </br>
	 * &nbsp } </br>
	 * } </br>
	 * </code>
	 * 
	 * @param A matrix for equality constraints
	 * 
	 * @param b the constant vector for equality constraints
	 *  
	 * @param H_x Hessian for the objective function and nonlinear functions for
	 *            inequality constraints
	 *            
	 * @param F_x a vector of nonlinear functions for inequality constraints, 
	 *            F_x(i) is the i-th nonlinear function value for the i-th 
	 *            inequality constraint
	 *            
	 * @param DF_x Derivative for F_x
	 * 
	 * @param G_f_x Gradient of the objective function at the current point x
	 * 
	 * @param fval objective function value at the current point x
	 * 
	 * @param x_s current point, will be set in place during the procedure
	 * 
	 * @param l_opt optimal Lambda
	 * 
	 * @param v_opt optimal Nu
	 * 
	 * @return a {@code boolean} array of two elements: {converge, gradientRequired}
	 * 
	 */
	public static boolean[] run(RealMatrix A, RealMatrix b, RealMatrix H_x, RealMatrix F_x, RealMatrix DF_x, RealMatrix G_f_x, double fval, RealMatrix x_s, RealMatrix l_opt, RealMatrix v_opt) {
		
		// If the algorithm has converged, we do a new job
		if (state == 5) {
			J.clear();
			state = 0;
		}

		if (state == 0) {
			
			tic();
			
			n = A.getColumnDimension();
			p = A.getRowDimension();
			m = F_x.getRowDimension();
			
			x = x_s.copy();
			l = rdivide(ones(m, 1), m);
			v = zeros(p, 1);
			
			eta_t = - innerProduct(F_x, l);
			
			// System.out.format("Initial ofv: %g\n", fval);
			
			k = 0;
			state = 1;
			
		}
		
		if (state == 1) {
			
			/*if (toc() > 3) {
				int a = 1;
				a = a + 1;
			}*/
			
			double residual_prim = 0;
			double residual_dual = 0;
			
			RealMatrix l_s = null;

			t = mu * m / eta_t;
	    		    	
	    	r_prim = A.multiply(x).subtract(b);
	        r_dual = G_f_x.add(DF_x.transpose().multiply(l)).add(A.transpose().multiply(v));
	        r_cent = uminus(times(l, F_x)).subtract(rdivide(ones(m, 1), t));
	        
	        Matrix = vertcat(
	        			horzcat(H_x, DF_x.transpose(), A.transpose()),
	        			horzcat(uminus(mtimes(diag(l),DF_x)), uminus(diag(F_x)), zeros(m, p)),
	        			horzcat(A, zeros(p, m), zeros(p, p))
	        		);
	        Vector = uminus(vertcat(r_dual, r_cent, r_prim));
	    
	        residual = norm(Vector);
	        residual_prim = norm(r_prim);
	        residual_dual = norm(r_dual);
	        eta_t = -innerProduct(F_x, l);
	        
	        // fprintf("f_x: %g, residual: %g, residual_prim: %g, residual_dual: %g\n", fval, residual, residual_prim, residual_dual);
	        if (residual_prim <= epsilon_feas &&
	        		residual_dual <= epsilon_feas &&
	        		eta_t <= epsilon) {
	        	fprintf("Terminate successfully.\n\n");
	        	if (l_opt != null)
	        		setMatrix(l_opt, l);
	        	if (v_opt != null)
	        		setMatrix(v_opt, v);
	        	converge = true;
	        	gradientRequired = false;
	        	state = 5;
	        	System.out.printf("Primal-dual interior-point algorithm converges.\n");
	        	return new boolean[] {converge, gradientRequired};
	        }

	        z_pd = mldivide(Matrix, Vector);
	        x_nt = z_pd.getSubMatrix(0, n - 1, 0, 0);
	        l_nt = z_pd.getSubMatrix(n, n + m - 1, 0, 0);
	        v_nt = z_pd.getSubMatrix(n + m, n + m + p - 1, 0, 0);
	    	
	        s = 1;
	        // Ensure lambda to be nonnegative
	        while (true) {
	            l_s = plus(l, times(s, l_nt));
	            /*disp(l_s);
	            display(s);*/
	            if (sumAll(lt(l_s, 0)) > 0) {
	                s = beta * s;
	            }else {
	            	/*int a = 1;
	            	a = a + 1;*/
	                break;
	            }
	        }
	        
	        /*// Ensure f_i(x) <= 0, i = 1, 2, ..., m
	        while (true) {
	            x_s = plus(x, times(scale, x_nt));
	            if (sumAll(gt(F_x_s, 0)) > 0)
	                scale = beta * scale;
	            else
	                break;
	        }*/
	        
	        state = 2;
	        
	        setMatrix(x_s, plus(x, times(s, x_nt)));
	        
	        converge = false;
			gradientRequired = false;
			
			return new boolean[] {converge, gradientRequired};
			
		}
		
		if (state == 2) {
			/*if (toc() > 3) {
				int a = 1;
				a = a + 1;
			}*/
			// display(s);
			if (sumAll(gt(F_x, 0)) > 0) {
				s = beta * s;
				setMatrix(x_s, plus(x, times(s, x_nt)));
				converge = false;
				gradientRequired = false;
				return new boolean[] {converge, gradientRequired};
			} else {
				// setMatrix(x, x_s);
				state = 3;
				converge = false;
				gradientRequired = true;
				return new boolean[] {converge, gradientRequired};
			}
		}
		
		if (state == 3) {
			
			/*if (toc() > 3) {
				int a = 1;
				a = a + 1;
			}*/
			
			RealMatrix r_prim_s = null;
	        RealMatrix r_dual_s = null;
			RealMatrix r_cent_s = null;
	        double residual_s = 0;
			
			RealMatrix l_s = null;
			RealMatrix v_s = null;
			
			l_s = plus(l, times(s, l_nt));
        	v_s = plus(v, times(s, v_nt));
			
        	r_prim_s = A.multiply(x_s).subtract(b);
	        r_dual_s = G_f_x.add(DF_x.transpose().multiply(l_s)).add(A.transpose().multiply(v_s));
	        r_cent_s = uminus(times(l_s, F_x)).subtract(rdivide(ones(m, 1), t));
	         
	        residual_s = norm(vertcat(r_dual_s, r_cent_s, r_prim_s));
	        if (residual_s <= (1 - alpha * s) * residual) {
	        	l = l_s;
	        	v = v_s;
	        	x = x_s.copy();
	        	state = 4;
	        } else {
	        	s = beta * s;
	        	converge = false;
	        	gradientRequired = true;
	        	setMatrix(x_s, plus(x, times(s, x_nt)));
	        	return new boolean[] {converge, gradientRequired};
	        }
        	
		}
		
		if (state == 4) {
        	k = k + 1;
		    state = 1;
		}
		
		converge = false;
	    gradientRequired = true;
	    return new boolean[] {converge, gradientRequired};
		
	}

}
