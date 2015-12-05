package jml.optimization;

import static jml.matlab.Matlab.abs;
import static jml.matlab.Matlab.min;
import static jml.matlab.Matlab.sign;
import static jml.matlab.Matlab.times;

import org.apache.commons.math.linear.RealMatrix;

/**
 * Compute proj_tC(X) where C = {X: || X ||_{\infty} <= 1}.
 * 
 * @author Mingjie Qian
 * @version 1.0, Oct. 14th, 2013
 */
public class ProjLInfinity implements Projection {

	/**
	 * Compute proj_{tC}(X) where C = {X: || X ||_{\infty} <= 1}.
	 * 
	 * @param t a nonnegative real scalar
	 * 
	 * @param X a real matrix
	 * 
	 * @return proj_{tC}(X) where C = {X: || X ||_{\infty} <= 1}
	 * 
	 */
	public RealMatrix compute(double t, RealMatrix X) {
		
		if (t < 0) {
			System.err.println("The first input should be a nonnegative real scalar.");
			System.exit(-1);
		}
		
		if (X.getColumnDimension() > 1) {
			System.err.println("The second input should be a vector.");
			System.exit(-1);
		}
		
		return times(sign(X), min(abs(X), t));
		
	}

}
