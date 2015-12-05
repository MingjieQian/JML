package jml.optimization;

import static jml.matlab.Matlab.disp;
import static jml.matlab.Matlab.eye;
import static jml.matlab.Matlab.fprintf;
import static jml.matlab.Matlab.innerProduct;
import static jml.matlab.Matlab.norm;
import static jml.matlab.Matlab.ones;
import static jml.matlab.Matlab.plus;
import static jml.matlab.Matlab.rand;
import static jml.matlab.Matlab.times;
import static jml.utils.Time.*;
import jml.data.Data;
/*import static jml.matlab.Matlab.disp;
import static jml.matlab.Matlab.display;*/

import org.apache.commons.math.linear.BlockRealMatrix;
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
 * @version Feb. 28th, 2013
 */
public class GeneralQPViaPrimalDualInteriorPoint {

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
		GeneralQPViaPrimalDualInteriorPoint.solve(Q, c, A, b, B, d);
		
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
		PhaseIResult phaseIResult = GeneralQP.phaseI(A, b, B, d);
		if (phaseIResult.feasible) {
			fprintf("Phase II:\n\n");
			RealMatrix x0 = phaseIResult.optimizer;
			// GeneralQP.phaseII(Q, c, A, b, B, d, x0);
			return phaseII(Q, c, A, b, B, d, x0);
		} else {
			System.err.println("The QP problem is infeasible!\n");
			return null;
		}
		
	}

	private static QPSolution phaseII(RealMatrix Q, RealMatrix c, RealMatrix A,
			RealMatrix b, RealMatrix B, RealMatrix d, RealMatrix x0) {
		
		RealMatrix x = x0.copy();
		RealMatrix l = new BlockRealMatrix(B.getRowDimension(), 1);
		RealMatrix v = new BlockRealMatrix(A.getRowDimension(), 1);
		RealMatrix H_x = null;
		RealMatrix F_x = null;
		RealMatrix DF_x = null;
		RealMatrix G_f_x = null;
		double fval = 0;
		
		fval = innerProduct(x, Q.multiply(x)) / 2 + innerProduct(c, x);
		H_x = Q;
		DF_x = B;
		F_x = B.multiply(x).subtract(d);
		G_f_x = Q.multiply(x).add(c);
		
		boolean flags[] = null;
		int k = 0;
		// int maxIter = 1000;
		tic();
		while (true) {
			flags = PrimalDualInteriorPoint.run(A, b, H_x, F_x, DF_x, G_f_x, fval, x, l, v);
			/*fprintf("F_x");
			display(F_x.transpose());
			fprintf("DF_x");
			display(DF_x);
			fprintf("G_f_x");
			display(G_f_x.transpose());
			fprintf("x");
			display(x.transpose());*/

			/*if (toc() > 3) {
				int a = 1;
				a = a + 1;
			}*/
			
			if (flags[0])
				break;
			
			/*
			 *  Compute the objective function value, if flags[1] is true
			 *  gradient will also be computed.
			 */
			fval = innerProduct(x, Q.multiply(x)) / 2 + innerProduct(c, x);
			F_x = B.multiply(x).subtract(d);
			// disp(F_x.transpose());
			// G_f_x = Q.multiply(x).add(c);
			if (flags[1]) {
				k = k + 1;
				// Compute the gradient
				/*if (k > maxIter)
					break;*/
				
				G_f_x = Q.multiply(x).add(c);
				
				/*if ( Math.abs(fval_pre - fval) < eps)
					break;
				fval_pre = fval;*/
			}
			
		}
		
		double t_primal_dual_interior_point = toc();
		
		double fval_primal_dual_interior_point = fval;
		RealMatrix x_primal_dual_interior_point = x;
		RealMatrix lambda_primal_dual_interior_point = l;
		RealMatrix v_primal_dual_interior_point = v;

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
		
		return new QPSolution(x, l, v, fval);
		
	}

}
