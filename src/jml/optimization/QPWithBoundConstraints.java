package jml.optimization;

import static jml.matlab.Matlab.zeros;
import static jml.matlab.Matlab.innerProduct;
import static jml.matlab.Matlab.rand;
import static jml.matlab.Matlab.plus;
import static jml.matlab.Matlab.minus;
import static jml.matlab.Matlab.eye;
import static jml.matlab.Matlab.times;
import static jml.matlab.Matlab.fprintf;
import static jml.matlab.Matlab.display;

import org.apache.commons.math.linear.RealMatrix;

/**
 * Quadratic programming with bound constraints:
 * <p>
 *      min 2 \ x' * Q * x + c' * x
 * s.t. l <= x <= u
 * </p>
 * 
 * @author Mingjie Qian
 * @version 1.0, Feb. 25th, 2013
 */
public class QPWithBoundConstraints {

	/**
	 * @param args
	 */
	public static void main(String[] args) {
		
		int n = 5;
		RealMatrix x = rand(n);
		RealMatrix Q = minus(x.multiply(x.transpose()), times(rand(1).getEntry(0, 0), eye(n)));
		RealMatrix c = plus(-2, times(2, rand(n, 1)));
		double l = 0;
		double u = 1;
		double epsilon = 1e-6;
		
		QPSolution S = QPWithBoundConstraints.solve(Q, c, l, u, epsilon);
		
		fprintf("Optimum: %g\n", S.optimum);
		fprintf("Optimizer:\n");
		display(S.optimizer.transpose());
		
	}
	
	/**
	 * Solve this bound constrained QP problem.
	 * 
	 * @param Q the positive semi-definite matrix
	 * 
	 * @param c the linear coefficient vector
	 * 
	 * @param l lower bound
	 * 
	 * @param u upper bound
	 * 
	 * @param epsilon convergence precision
	 * 
	 * @return a {@code QPSolution} instance containing the optimizer
	 *           and the optimum
	 * 
	 */
	public static QPSolution solve(RealMatrix Q, RealMatrix c, double l, double u, double epsilon) {
		return solve(Q, c, l, u, epsilon, null);
	}
	
	/**
	 * Solve this bound constrained QP problem.
	 * 
	 * @param Q the positive semi-definite matrix
	 * 
	 * @param c the linear coefficient vector
	 * 
	 * @param l lower bound
	 * 
	 * @param u upper bound
	 * 
	 * @param epsilon convergence precision
	 * 
	 * @param x0 staring point if not null
	 * 
	 * @return a {@code QPSolution} instance containing the optimizer
	 *           and the optimum
	 * 
	 */
	public static QPSolution solve(RealMatrix Q, RealMatrix c, double l, double u, double epsilon, RealMatrix x0) {
		
		int d = Q.getColumnDimension();
		double fval = 0;
		RealMatrix x = null;
		if (x0 != null) {
			x = x0;
		} else {
			x = plus((l + u) / 2, zeros(d, 1));
		}
		
		/* 
		 * Grad = Q * x + c
		 */
		RealMatrix Grad = Q.multiply(x).add(c);
		fval = innerProduct(x, Q.multiply(x)) / 2 + innerProduct(c, x);
		
		boolean flags[] = null;
		while (true) {
			flags = BoundConstrainedPLBFGS.run(Grad, fval, l, u, epsilon, x);
			if (flags[0])
				break;
			fval = innerProduct(x, Q.multiply(x)) / 2 + innerProduct(c, x);
			if (flags[1])
				Grad = Q.multiply(x).add(c);
		}
		return new QPSolution(x, null, null, fval);
		
	}

}

