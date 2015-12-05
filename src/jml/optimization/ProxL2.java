package jml.optimization;

import static jml.matlab.Matlab.norm;
import static jml.matlab.Matlab.size;
import static jml.matlab.Matlab.times;
import static jml.matlab.Matlab.zeros;

import org.apache.commons.math.linear.RealMatrix;

/**
 * Compute prox_th(X) where h = || X ||_F.
 * 
 * @author Mingjie Qian
 * @version 1.0, Mar. 11th, 2013
 */
public class ProxL2 implements ProximalMapping {

	/**
	 * Compute prox_th(X) where h = || X ||_F. For a 
	 * vector, h(X) is the l_2 norm of X, for a matrix
	 * h(X) is the Frobenius norm of X.
	 * 
	 * @param t a nonnegative real scalar
	 * 
	 * @param X a real matrix
	 * 
	 * @return prox_th(X) where h = || X ||_F
	 * 
	 */
	public RealMatrix compute(double t, RealMatrix X) {
		
		double norm = norm(X, "fro");
		if (norm <= t) {
			return zeros(size(X));
		} else {
			return X.subtract(times(t / norm, X));
		}
		
	}

}
