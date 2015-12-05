package jml.optimization;

import static jml.matlab.Matlab.norm;
import static jml.matlab.Matlab.times;

import org.apache.commons.math.linear.RealMatrix;

/**
 * Compute proj_tC(X) where C = {X: || X ||_2 <= 1}.
 * 
 * @author Mingjie Qian
 * @version 1.0, Oct. 14th, 2013
 */
public class ProjL2 implements Projection {

	/**
	 * Compute proj_{tC}(X) where C = {X: || X ||_2 <= 1}.
	 * 
	 * @param t a nonnegative real scalar
	 * 
	 * @param X a real matrix
	 * 
	 * @return proj_{tC}(X) where C = {X: || X ||_2 <= 1}
	 * 
	 */
	public RealMatrix compute(double t, RealMatrix X) {
		
		double norm = norm(X, "fro");
		if (norm <= t) {
			return X;
		} else {
			return times(t / norm, X);
		}
		
	}

}
