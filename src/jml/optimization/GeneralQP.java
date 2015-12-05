package jml.optimization;

import static jml.matlab.Matlab.diag;
import static jml.matlab.Matlab.disp;
import static jml.matlab.Matlab.eye;
import static jml.matlab.Matlab.fprintf;
import static jml.matlab.Matlab.getRows;
import static jml.matlab.Matlab.horzcat;
import static jml.matlab.Matlab.innerProduct;
import static jml.matlab.Matlab.lt;
import static jml.matlab.Matlab.mldivide;
import static jml.matlab.Matlab.mtimes;
import static jml.matlab.Matlab.norm;
import static jml.matlab.Matlab.ones;
import static jml.matlab.Matlab.plus;
import static jml.matlab.Matlab.rand;
import static jml.matlab.Matlab.rdivide;
import static jml.matlab.Matlab.sumAll;
import static jml.matlab.Matlab.times;
import static jml.matlab.Matlab.uminus;
import static jml.matlab.Matlab.vertcat;
import static jml.matlab.Matlab.zeros;
import static jml.utils.Time.pause;
import static jml.utils.Time.tic;
import static jml.utils.Time.toc;
import jml.data.Data;

import org.apache.commons.math.linear.RealMatrix;

/**
 * General quadratic programming:
 * <p>
 *      min 2 \ x' * Q * x + c' * x </br>
 * s.t. A * x = b </br>
 *      B * x <= d </br>
 * </p>
 * 
 * @author Mingjie Qian
 * @version 1.0 Feb. 27th, 2013
 */
public class GeneralQP {

	/**
	 * @param args
	 */
	public static void main(String[] args) {
		
		/*
		 * Number of unknown variables
		 */
		int n = 5;
		
		/*
		 * Number of inequality constraints
		 */
		int m = 6;
		
		/*
		 * Number of equality constraints
		 */
		int p = 3;

		/*RealMatrix x = rand(n, n);
		RealMatrix Q = x.multiply(x.transpose()).add(times(rand(1), eye(n)));
		RealMatrix c = rand(n, 1);

		double HasEquality = 1;
		RealMatrix A = times(HasEquality, rand(p, n));
		x = rand(n, 1);
		RealMatrix b = A.multiply(x);
		RealMatrix B = rand(m, n);
		double rou = -2;
		RealMatrix d = plus(B.multiply(x), times(rou, ones(m, 1)));*/
		
		RealMatrix x = null;
		RealMatrix Q = null;
		RealMatrix c = null;
		RealMatrix A = null;
		RealMatrix b = null;
		RealMatrix B = null;
		RealMatrix d = null;
		double rou = -2;
		double HasEquality = 1;
		
		boolean generate = false;
		if (generate) {
			x = rand(n, n);
			Q = x.multiply(x.transpose()).add(times(rand(1), eye(n)));
			c = rand(n, 1);

			A = times(HasEquality, rand(p, n));
			x = rand(n, 1);
			b = A.multiply(x);
			B = rand(m, n);
			d = plus(B.multiply(x), times(rou, ones(m, 1)));

			Data.saveMatrix("Q", Q);
			Data.saveMatrix("c", c);
			Data.saveMatrix("A", A);
			Data.saveMatrix("b2", b);
			Data.saveMatrix("B", B);
			Data.saveMatrix("d", d);
		} else {
			Q = Data.loadMatrix("Q");
			c = Data.loadMatrix("c");
			A = Data.loadMatrix("A");
			b = Data.loadMatrix("b2");
			B = Data.loadMatrix("B");
			d = Data.loadMatrix("d");
		}
		
		/*
		 * General quadratic programming:
		 *
		 *      min 2 \ x' * Q * x + c' * x
		 * s.t. A * x = b
		 *      B * x <= d
		 */
		GeneralQP.solve(Q, c, A, b, B, d);

	}
	
	/**
	 * Solve a general quadratic programming problem formulated as
	 * <p>
	 *      min 2 \ x' * Q * x + c' * x </br>
	 * s.t. A * x = b </br>
	 *      B * x <= d </br>
	 * </p>
	 * 
	 * @param Q an n x n positive definite or semi-definite matrix
	 * 
	 * @param c an n x 1 real matrix
	 * 
	 * @param A a p x n real matrix
     * 
     * @param b a p x 1 real matrix
     * 
     * @param B an m x n real matrix
     * 
     * @param d an m x 1 real matrix
     * 
	 * @return a {@code QPSolution} instance if the general QP problems
	 *         is feasible or null otherwise
	 *         
	 */
	public static QPSolution solve(RealMatrix Q, RealMatrix c, RealMatrix A, RealMatrix b, RealMatrix B, RealMatrix d) {

		fprintf("Phase I:\n\n");
		PhaseIResult phaseIResult = phaseI(A, b, B, d);
		if (phaseIResult.feasible) {
			fprintf("Phase II:\n\n");
			RealMatrix x0 = phaseIResult.optimizer;
			return phaseII(Q, c, A, b, B, d, x0);
		} else {
			System.err.println("The QP problem is infeasible!\n");
			return null;
		}
		
	}
	
	/**
	 * We demonstrate the implementation of phase I via primal-dual interior
	 * point method to test whether the following problem is feasible:
	 * </p>
	 *      min f(x) </br>
	 * s.t. A * x = b </br>
	 *      B * x <= d </br>
	 * </p>
	 * We seek the optimizer for the following phase I problem:
	 * </p>
     *      min 1's </br>
     * s.t. A * x = b </br>
     *      B * x - d <= s </br>
     *      s >= 0 </br>
     * </p>     
     * <=> </br>
     *      min cI'y </br>
     * s.t. AI * y = b </br>
     *      BI * y <= dI </br>
     * </p>
     * cI = [zeros(n, 1); ones(m, 1)] </br>
     * AI = [A zeros(p, m)] </br>
     * BI = [B -eye(m); zeros(m, n) -eye(m)] </br>
     * dI = [d; zeros(m, 1)] </br>
     * y = [x; s] </br>
     * 
     * @param A a p x n real matrix
     * 
     * @param b a p x 1 real matrix
     * 
     * @param B an m x n real matrix
     * 
     * @param d an m x 1 real matrix
     *      
	 * @return a {@code PhaseIResult} instance if feasible or null if infeasible
	 * 
	 */
	public static PhaseIResult phaseI(RealMatrix A, RealMatrix b, RealMatrix B, RealMatrix d) {
		
		/*
		 * Number of unknown variables
		 */
		int n = A.getColumnDimension();
		
		/*
		 * Number of equality constraints
		 */
		int p = A.getRowDimension();
		
		/*
		 * Number of inequality constraints
		 */
		int m = B.getRowDimension();
		
		RealMatrix A_ori = A;
		RealMatrix B_ori = B;
		RealMatrix d_ori = d;
		
		RealMatrix c = vertcat(zeros(n, 1), ones(m, 1));
		A = horzcat(A, zeros(p, m));
		B = vertcat(horzcat(B, uminus(eye(m))), horzcat(zeros(m, n), uminus(eye(m))));
		d = vertcat(d, zeros(m, 1));
		
		int n_ori = n;
		int m_ori = m;
		
		n = n + m;
		m = 2 * m;
		
		// RealMatrix x0 = rand(n_ori, 1);
		RealMatrix x0 = ones(n_ori, 1);
		RealMatrix s0 = B_ori.multiply(x0).subtract(d_ori).add(ones(m_ori, 1));
		x0 = vertcat(x0, s0);
		RealMatrix v0 = zeros(p, 1);
		
		// Parameter setting
		
		double mu = 1.8;
		double epsilon = 1e-10;
		double epsilon_feas = 1e-10;
		double alpha = 0.1;
		double beta = 0.98;
		
		tic();
		
		RealMatrix l0 = rdivide(ones(m, 1), m);

		RealMatrix x = x0;
		RealMatrix l = l0;
		RealMatrix v = v0;
		
		RealMatrix F_x_0 = B.multiply(x).subtract(d);
		
		double eta_t = - innerProduct(F_x_0, l0);
		double t = 1;
		double f_x = 0;
		RealMatrix G_f_x = null;
		RealMatrix F_x = null;
		RealMatrix DF_x = null;
		RealMatrix H_x = times(1e-10, eye(n));
		RealMatrix r_prim = null;
		RealMatrix r_dual = null;
		RealMatrix r_cent = null;
		RealMatrix Matrix = null;
		RealMatrix Vector = null;
		
		double residual = 0;
		double residual_prim = 0;
		double residual_dual = 0;
		
		RealMatrix z_pd = null;
		RealMatrix x_nt = null;
		RealMatrix l_nt = null;
		RealMatrix v_nt = null;
		
		RealMatrix x_s = null;
		RealMatrix l_s = null;
		RealMatrix v_s = null;
		
		double s = 0;
		RealMatrix G_f_x_s = null;
        RealMatrix F_x_s = null;
        RealMatrix DF_x_s = null;
        
        RealMatrix r_prim_s = null;
        RealMatrix r_dual_s = null;
		RealMatrix r_cent_s = null;
        double residual_s = 0;
        // int k = 0;
		while (true) {

			t = mu * m / eta_t;
	    	f_x = innerProduct(c, x);
	    	
	    	// Calculate the gradient of f(x)
	    	G_f_x = c;
	    
	    	// Calculate F(x) and DF(x)
	    	F_x = B.multiply(x).subtract(d);
	    	DF_x = B;
	    	
	    	// Calculate the Hessian matrix of f(x) and fi(x)
	    	// H_x = times(1e-10, eye(n));
	    	
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
	        eta_t = - innerProduct(F_x, l);
	        
	        // fprintf("f_x: %g, residual: %g\n", f_x, residual);
	        if (residual_prim <= epsilon_feas &&
	        	residual_dual <= epsilon_feas &&
	        	eta_t <= epsilon) {
	        	fprintf("Terminate successfully.\n\n");
	        	break;
	        }
	    	
	        z_pd = mldivide(Matrix, Vector);
	        /*fprintf("k = %d%n", k++);
	        disp(z_pd.transpose());*/
	        x_nt = z_pd.getSubMatrix(0, n - 1, 0, 0);
	        l_nt = z_pd.getSubMatrix(n, n + m - 1, 0, 0);
	        v_nt = z_pd.getSubMatrix(n + m, n + m + p - 1, 0, 0);
	        
	        // Backtracking line search
	        
	        s = 1;
	        // Ensure lambda to be nonnegative
	        while (true) {
	            l_s = plus(l, times(s, l_nt));
	            if (sumAll(lt(l_s, 0)) > 0)
	                s = beta * s;
	            else
	                break;
	        }
	        
	        // Ensure f_i(x) <= 0, i = 1, 2, ..., m
	        while (true) {
	            x_s = plus(x, times(s, x_nt));
	            if (sumAll(lt(d.subtract(B.multiply(x_s)), 0)) > 0)
	                s = beta * s;
	            else
	                break;
	        }
	        
	        while (true) {
	        	
	        	x_s = plus(x, times(s, x_nt));
	        	l_s = plus(l, times(s, l_nt));
	        	v_s = plus(v, times(s, v_nt));
		        
		        // Template {
		        
		        // Calculate the gradient of f(x_s)
		        G_f_x_s = c;
		        
		        // Calculate F(x_s) and DF(x_s)
		        F_x_s = B.multiply(x_s).subtract(d);
		        DF_x_s = B;
		        
		        // }
		        
		        r_prim_s = A.multiply(x_s).subtract(b);
		        r_dual_s = G_f_x_s.add(DF_x_s.transpose().multiply(l_s)).add(A.transpose().multiply(v_s));
		        r_cent_s = uminus(times(l_s, F_x_s)).subtract(rdivide(ones(m, 1), t));
		         
		        residual_s = norm(vertcat(r_dual_s, r_cent_s, r_prim_s));
		        if (residual_s <= (1 - alpha * s) * residual)
		            break;
		        else
		            s = beta * s;
		        
		    }
	        
	        x = x_s;
	        l = l_s;
	        v = v_s;
	    	
		}
		
		double t_sum_of_inequalities = toc();

		RealMatrix x_opt = getRows(x, 0, n_ori - 1);
		fprintf("x_opt:\n");
		disp(x_opt.transpose());

		RealMatrix s_opt = getRows(x, n_ori, n - 1);
		fprintf("s_opt:\n");
		disp(s_opt.transpose());

		RealMatrix lambda_s = getRows(l, m_ori, m - 1);
		fprintf("lambda for the inequalities s_i >= 0:\n");
		disp(lambda_s.transpose());

		RealMatrix e = B_ori.multiply(x_opt).subtract(d_ori);
		fprintf("B * x - d:\n");
		disp(e.transpose());

		RealMatrix lambda_ineq = getRows(l, 0, m_ori - 1);
		fprintf("lambda for the inequalities fi(x) <= s_i:\n");
		disp(lambda_ineq.transpose());

		RealMatrix v_opt = v;
		fprintf("nu for the equalities A * x = b:\n");
		disp(v_opt.transpose());

		fprintf("residual: %g\n\n", residual);
		fprintf("A * x - b:\n");
		disp(A_ori.multiply(x_opt).subtract(b).transpose());
		fprintf("norm(A * x - b, \"fro\"): %f\n\n", norm(A_ori.multiply(x_opt).subtract(b), "fro"));

		double fval_opt = f_x;
		fprintf("fval_opt: %g\n\n", fval_opt);
		boolean feasible = false;
		if (fval_opt <= epsilon) {
		    feasible = true;
		    fprintf("The problem is feasible.\n\n");
		} else {
		    feasible = false;
		    fprintf("The problem is infeasible.\n\n");
		}
		fprintf("Computation time: %f seconds\n\n", t_sum_of_inequalities);

		/*if (!feasible)
		    return null;*/

		x0 = x_opt;

		int pause_time = 1;
		fprintf("halt execution temporarily in %d seconds...\n\n", pause_time);
		pause(pause_time);
		
		return new PhaseIResult(feasible, x_opt, fval_opt);
		
	}
	
	/**
	 * Phase II for solving a general quadratic programming problem formulated as
	 * <p>
	 *      min 2 \ x' * Q * x + c' * x </br>
	 * s.t. A * x = b </br>
	 *      B * x <= d </br>
	 * </p>
	 * 
	 * @param Q an n x n positive definite or semi-definite matrix
	 * 
	 * @param c an n x 1 real matrix
	 * 
	 * @param A a p x n real matrix
     * 
     * @param b a p x 1 real matrix
     * 
     * @param B an m x n real matrix
     * 
     * @param d an m x 1 real matrix
     * 
     * @param x0 starting point
     * 
	 * @return a {@code QPSolution} instance
	 *         
	 */
	public static QPSolution phaseII(RealMatrix Q, RealMatrix c, RealMatrix A, RealMatrix b, RealMatrix B, RealMatrix d, RealMatrix x0) {
		
		/*
		 * Number of unknown variables
		 */
		int n = A.getColumnDimension();
		
		/*
		 * Number of equality constraints
		 */
		int p = A.getRowDimension();
		
		/*
		 * Number of inequality constraints
		 */
		int m = B.getRowDimension();
		
		
		RealMatrix v0 = zeros(p, 1);
		
		// Parameter setting
		
		double mu = 1.8;
		double epsilon = 1e-10;
		double epsilon_feas = 1e-10;
		double alpha = 0.1;
		double beta = 0.98;
		
		tic();
		
		RealMatrix l0 = rdivide(ones(m, 1), m);

		RealMatrix x = x0;
		RealMatrix l = l0;
		RealMatrix v = v0;
		
		RealMatrix F_x_0 = B.multiply(x).subtract(d);
		
		double eta_t = - innerProduct(F_x_0, l0);
		double t = 1;
		double f_x = 0;
		RealMatrix G_f_x = null;
		RealMatrix F_x = null;
		RealMatrix DF_x = null;
		RealMatrix H_x = Q;
		RealMatrix r_prim = null;
		RealMatrix r_dual = null;
		RealMatrix r_cent = null;
		RealMatrix Matrix = null;
		RealMatrix Vector = null;
		
		double residual = 0;
		double residual_prim = 0;
		double residual_dual = 0;
		
		RealMatrix z_pd = null;
		RealMatrix x_nt = null;
		RealMatrix l_nt = null;
		RealMatrix v_nt = null;
		
		RealMatrix x_s = null;
		RealMatrix l_s = null;
		RealMatrix v_s = null;
		
		double s = 0;
		RealMatrix G_f_x_s = null;
        RealMatrix F_x_s = null;
        RealMatrix DF_x_s = null;
        
        RealMatrix r_prim_s = null;
        RealMatrix r_dual_s = null;
		RealMatrix r_cent_s = null;
        double residual_s = 0;
        
		while (true) {

			t = mu * m / eta_t;
	    	f_x = innerProduct(x, Q.multiply(x)) / 2 + innerProduct(c, x);
	    	
	    	// Calculate the gradient of f(x)
	    	G_f_x = Q.multiply(x).add(c);
	    
	    	// Calculate F(x) and DF(x)
	    	F_x = B.multiply(x).subtract(d);
	    	DF_x = B;
	    	
	    	// Calculate the Hessian matrix of f(x) and fi(x)
	    	// H_x = times(1e-10, eye(n));
	    	
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
	        
	        // fprintf("f_x: %g, residual: %g\n", f_x, residual);
	        if (residual_prim <= epsilon_feas &&
	        	residual_dual <= epsilon_feas &&
	        	eta_t <= epsilon) {
	        	fprintf("Terminate successfully.\n\n");
	        	break;
	        }
	    	
	        z_pd = mldivide(Matrix, Vector);
	        x_nt = z_pd.getSubMatrix(0, n - 1, 0, 0);
	        l_nt = z_pd.getSubMatrix(n, n + m - 1, 0, 0);
	        v_nt = z_pd.getSubMatrix(n + m, n + m + p - 1, 0, 0);
	        
	        // Backtracking line search
	        
	        s = 1;
	        // Ensure lambda to be nonnegative
	        while (true) {
	            l_s = plus(l, times(s, l_nt));
	            if (sumAll(lt(l_s, 0)) > 0)
	                s = beta * s;
	            else
	                break;
	        }
	        
	        // Ensure f_i(x) <= 0, i = 1, 2, ..., m
	        while (true) {
	            x_s = plus(x, times(s, x_nt));
	            if (sumAll(lt(d.subtract(B.multiply(x_s)), 0)) > 0)
	                s = beta * s;
	            else
	                break;
	        }
	        
	        while (true) {
	        	
	        	x_s = plus(x, times(s, x_nt));
	        	l_s = plus(l, times(s, l_nt));
	        	v_s = plus(v, times(s, v_nt));
		        
		        // Template {
		        
		        // Calculate the gradient of f(x_s)
		        G_f_x_s = Q.multiply(x_s).add(c);
		        
		        // Calculate F(x_s) and DF(x_s)
		        F_x_s = B.multiply(x_s).subtract(d);
		        DF_x_s = B;
		        
		        // }
		        
		        r_prim_s = A.multiply(x_s).subtract(b);
		        r_dual_s = G_f_x_s.add(DF_x_s.transpose().multiply(l_s)).add(A.transpose().multiply(v_s));
		        r_cent_s = uminus(times(l_s, F_x_s)).subtract(rdivide(ones(m, 1), t));
		         
		        residual_s = norm(vertcat(r_dual_s, r_cent_s, r_prim_s));
		        if (residual_s <= (1 - alpha * s) * residual)
		            break;
		        else
		            s = beta * s;
		        
		    }
	        
	        x = x_s;
	        l = l_s;
	        v = v_s;
	    	
		}
		
		double t_primal_dual_interior_point = toc();
		
		double fval_primal_dual_interior_point = f_x;
		RealMatrix x_primal_dual_interior_point = x;
		RealMatrix lambda_primal_dual_interior_point = l;
		RealMatrix v_primal_dual_interior_point = v;

		fprintf("residual: %g\n\n", residual);
		fprintf("Optimal objective function value: %g\n\n", fval_primal_dual_interior_point);
		fprintf("Optimizer:\n");
		disp(x_primal_dual_interior_point.transpose());

		RealMatrix e = B.multiply(x).subtract(d);
		fprintf("B * x - d:\n");
		disp(e.transpose());

		fprintf("lambda:\n");
		disp(lambda_primal_dual_interior_point.transpose());

		fprintf("nu:\n");
		disp(v_primal_dual_interior_point.transpose());

		fprintf("norm(A * x - b, \"fro\"): %f\n\n", norm(A.multiply(x_primal_dual_interior_point).subtract(b), "fro"));
		fprintf("Computation time: %f seconds\n\n", t_primal_dual_interior_point);
		
		return new QPSolution(x, l, v, f_x);
		
	}

}
